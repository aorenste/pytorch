// CFG executor: a PEP 523 eval-frame hook that runs each Python frame by
// converting its code object to a CFG (see cfg/) and interpreting that, instead
// of CPython's bytecode loop. Anything it can't run faithfully fails hard with
// an exception rather than falling back to CPython.

pub mod ffi;
mod live;
mod run;
mod trampoline;

use std::cell::Cell;
use std::collections::HashMap;
use std::os::raw::{c_int, c_void};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

use crate::cfg::ir::{BlockId, Cfg, Op, ValueId};
use crate::cfg::opcodes::{self, Out};
use crate::cfg::outline::Outline;

use ffi::*;

static INSTALLED: Mutex<usize> = Mutex::new(0);
static PREV_EVAL_FRAME: AtomicUsize = AtomicUsize::new(0);

thread_local! {
    static ACTIVE: Cell<usize> = const { Cell::new(0) };
}

const CO_GENERATOR: u32 = 0x20;
const CO_COROUTINE: u32 = 0x80;
const CO_ITERABLE_COROUTINE: u32 = 0x100;
const CO_ASYNC_GENERATOR: u32 = 0x200;
const GENERATOR_FLAGS: u32 = CO_GENERATOR | CO_COROUTINE | CO_ITERABLE_COROUTINE | CO_ASYNC_GENERATOR;

// A Cfg with what the executor precomputes for it, and the same for the
// origins of its blocks.
pub struct Prepared {
    pub cfg: Arc<Cfg>,
    // For an instr with several outputs, the Project instrs receiving them:
    // (output position, instr).
    pub projects: Vec<Vec<(u32, ValueId)>>,
    // Number of new outputs of each Op::Bytecode instr.
    pub nouts: Vec<u32>,
    pub live: live::Liveness,
    // By block.
    pub origins: Vec<Option<Prepared>>,
}

impl Prepared {
    fn new(cfg: Arc<Cfg>) -> PyResult<Self> {
        let n = cfg.instrs.len();
        let mut projects = vec![vec![]; n];
        let mut nouts = vec![0u32; n];
        for v in cfg.blocks.iter().flat_map(|b| &b.versions).flat_map(|ver| &ver.body) {
            match &cfg[*v].op {
                Op::Project(k) => projects[cfg[*v].args[0].idx()].push((*k, *v)),
                Op::Bytecode { opcode, arg } => {
                    let sig = opcodes::signature(*opcode, *arg, false).expect("converted op has a signature");
                    nouts[v.idx()] = sig.outputs.iter().filter(|o| **o == Out::New).count() as u32;
                }
                _ => {}
            }
        }
        let live = live::compute(&cfg);
        if let Some(y) = live.bad_yield {
            return Err(pyo3::exceptions::PyNotImplementedError::new_err(format!(
                "CFG executor: a value not on the stack is live across the yield at offset {}",
                cfg[y].origin * 2
            )));
        }
        let origins = cfg
            .blocks
            .iter()
            .map(|b| b.origin.clone().map(Prepared::new).transpose())
            .collect::<PyResult<_>>()?;
        Ok(Prepared {
            cfg,
            projects,
            nouts,
            live,
            origins,
        })
    }
}

// Per-code-object data, cached in co_extra and freed with the code object.
pub struct Compiled {
    pub prep: Prepared,
    pub flags: u32,
    pub consts: Py<PyTuple>,
    pub names: Py<PyTuple>,
    // Where a generator frame resumes, by the code unit its prev_instr points
    // at: each Yield, and RETURN_GENERATOR for a just-created generator. The
    // instr after it is the RESUME's Op::Resume. (block, version, index).
    // Outlining keeps the blocks holding them, so they are all in the
    // top-level Cfg.
    pub resume: HashMap<u32, (BlockId, usize, usize)>,
}

impl Compiled {
    fn new(code: &Bound<'_, PyAny>) -> PyResult<Self> {
        let info = crate::cfg::python::code_info(code)?;
        let outline = *OUTLINE.lock().unwrap();
        let cfg = crate::cfg::build(info, true, outline).map_err(crate::cfg::python::to_pyerr)?;
        let mut resume = HashMap::new();
        let bodies = cfg.block_ids().flat_map(|b| cfg[b].versions.iter().enumerate().map(move |(vi, ver)| (b, vi, ver)));
        for (b, vi, ver) in bodies {
            for (i, v) in ver.body.iter().enumerate() {
                if let Op::Yield
                | Op::Bytecode {
                    opcode: opcodes::RETURN_GENERATOR,
                    ..
                } = &cfg[*v].op
                {
                    if ver.body.get(i + 1).map(|r| &cfg[*r].op) != Some(&Op::Resume) {
                        return Err(pyo3::exceptions::PyNotImplementedError::new_err(format!(
                            "CFG executor: no RESUME after the resume point at offset {}",
                            cfg[*v].origin * 2
                        )));
                    }
                    resume.insert(cfg[*v].origin, (b, vi, i));
                }
            }
        }
        Ok(Compiled {
            resume,
            prep: Prepared::new(Arc::new(cfg))?,
            flags: code.getattr("co_flags")?.extract()?,
            consts: code.getattr("co_consts")?.downcast_into::<PyTuple>()?.unbind(),
            names: code.getattr("co_names")?.downcast_into::<PyTuple>()?.unbind(),
        })
    }
}

// How the executor outlines the CFGs it builds (for testing outlining).
static OUTLINE: Mutex<Option<Outline>> = Mutex::new(None);

/// Outline chains of blocks in the CFGs built from now on: None, "merge" or
/// "origin" (see cfg/outline.rs). Code already run keeps its CFG.
#[pyfunction]
#[pyo3(signature = (mode))]
pub fn _set_cfg_outline(mode: Option<&str>) -> PyResult<()> {
    *OUTLINE.lock().unwrap() = crate::cfg::python::outline_mode(mode)?;
    Ok(())
}

unsafe extern "C" fn free_compiled(p: *mut c_void) {
    if !p.is_null() && p != trampoline::mark() {
        // CPython frees code objects without pyo3 knowing the GIL is held, so
        // attach, or the Compiled's objects would only be released the next
        // time pyo3 attaches (as they are if it can't: while finalizing).
        let c = Box::from_raw(p as *mut Compiled);
        let _ = Python::try_attach(|_| drop(c));
    }
}

pub(crate) fn extra_index() -> isize {
    static INDEX: OnceLock<isize> = OnceLock::new();
    *INDEX.get_or_init(|| unsafe { PyUnstable_Eval_RequestCodeExtraIndex(free_compiled) })
}

enum Lookup<'a> {
    Compiled(&'a Compiled),
    // A trampoline (see trampoline.rs), run by CPython's evaluator.
    Trampoline,
    // A Python exception is set.
    Error,
}

// The cached Compiled for `code`, converting it on first use. The result lives
// as long as the code object.
unsafe fn compiled<'a>(code: *mut pyo3::ffi::PyObject) -> Lookup<'a> {
    let index = extra_index();
    let mut p: *mut c_void = std::ptr::null_mut();
    if index < 0 || PyUnstable_Code_GetExtra(code, index, &mut p) < 0 {
        return Lookup::Error;
    }
    if p == trampoline::mark() {
        return Lookup::Trampoline;
    }
    if !p.is_null() {
        return Lookup::Compiled(&*(p as *const Compiled));
    }
    Python::attach(|py| {
        let code_obj = Bound::from_borrowed_ptr(py, code);
        // A generator resumed with a throw (close(), throw(), or GC finalizing
        // one CPython created) arrives with the exception set; building would
        // clear it.
        let pending = pyo3::PyErr::take(py);
        match Compiled::new(&code_obj) {
            Ok(c) => {
                if let Some(e) = pending {
                    e.restore(py);
                }
                let p = Box::into_raw(Box::new(c));
                if PyUnstable_Code_SetExtra(code, index, p as *mut c_void) < 0 {
                    drop(Box::from_raw(p));
                    return Lookup::Error;
                }
                Lookup::Compiled(&*p)
            }
            Err(e) => {
                e.restore(py);
                Lookup::Error
            }
        }
    })
}

unsafe fn fail(msg: &str, exc: *mut pyo3::ffi::PyObject) {
    let msg = std::ffi::CString::new(msg).unwrap();
    pyo3::ffi::PyErr_SetString(exc, msg.as_ptr());
}

unsafe extern "C" fn eval_frame(
    tstate: *mut pyo3::ffi::PyThreadState,
    frame: *mut Frame,
    throwflag: c_int,
) -> *mut pyo3::ffi::PyObject {
    let prev: EvalFrameFn = std::mem::transmute(PREV_EVAL_FRAME.load(Ordering::Acquire));
    if ACTIVE.get() == 0 {
        return prev(tstate, frame, throwflag);
    }
    // Resuming a generator (created by either engine): the frame is owned by
    // the generator object, and we continue from where it was suspended.
    let gen_frame = tr_frame_owned_by_generator(frame) != 0;
    let code = tr_frame_code(frame);
    let compiled = match compiled(code) {
        Lookup::Compiled(c) => Some(c),
        Lookup::Trampoline => return prev(tstate, frame, throwflag),
        Lookup::Error => None,
    };
    if !gen_frame {
        let Some(c) = compiled else {
            tr_frame_clear_and_pop(tstate, frame);
            return std::ptr::null_mut();
        };
        if c.flags & GENERATOR_FLAGS != 0 {
            // Calling a generator function runs only its prologue, up to
            // RETURN_GENERATOR, which creates the generator object (from
            // CPython internals that aren't exported). The executor runs the
            // body when the generator is resumed.
            return prev(tstate, frame, throwflag);
        }
    }
    if tr_frame_enter(tstate, frame) < 0 || compiled.is_none() {
        return leave(tstate, frame, gen_frame, run::Outcome::Error);
    }
    let c = compiled.unwrap();
    if tr_tracing_active(tstate) != 0 {
        fail(
            "CFG executor: sys.settrace/setprofile/monitoring are not supported",
            pyo3::ffi::PyExc_RuntimeError,
        );
        return leave(tstate, frame, gen_frame, run::Outcome::Error);
    }
    let start = if gen_frame {
        let unit = tr_frame_get_instr(frame) as u32;
        let Some(&(b, ver, i)) = c.resume.get(&unit) else {
            fail(
                &format!("CFG executor: can't resume a generator at offset {}", unit * 2),
                pyo3::ffi::PyExc_RuntimeError,
            );
            return leave(tstate, frame, gen_frame, run::Outcome::Error);
        };
        run::Start::Resume { b, ver, i, throw: throwflag != 0 }
    } else {
        run::Start::Entry
    };
    let outcome = run::run(c, frame, start);
    leave(tstate, frame, gen_frame, outcome)
}

unsafe fn leave(
    tstate: *mut pyo3::ffi::PyThreadState,
    frame: *mut Frame,
    gen_frame: bool,
    outcome: run::Outcome,
) -> *mut pyo3::ffi::PyObject {
    let result = match outcome {
        run::Outcome::Yield { value, unit } => {
            tr_gen_suspend(tstate, frame, unit as c_int);
            return value;
        }
        run::Outcome::Return(v) => v,
        run::Outcome::Error => std::ptr::null_mut(),
        run::Outcome::Leave(_) | run::Outcome::Outer(..) => unreachable!("only an outlined Cfg leaves"),
    };
    if gen_frame {
        tr_gen_exit(tstate, frame);
    } else {
        tr_frame_exit(tstate, frame);
    }
    result
}

struct ExecGuard;

impl ExecGuard {
    fn new() -> Self {
        ACTIVE.set(ACTIVE.get() + 1);
        let mut installed = INSTALLED.lock().unwrap();
        if *installed == 0 {
            unsafe {
                let interp = pyo3::ffi::PyInterpreterState_Get();
                let prev = _PyInterpreterState_GetEvalFrameFunc(interp);
                PREV_EVAL_FRAME.store(prev as usize, Ordering::Release);
                _PyInterpreterState_SetEvalFrameFunc(interp, eval_frame);
            }
        }
        *installed += 1;
        ExecGuard
    }
}

impl Drop for ExecGuard {
    fn drop(&mut self) {
        let mut installed = INSTALLED.lock().unwrap();
        *installed -= 1;
        if *installed == 0 {
            unsafe {
                let prev: EvalFrameFn = std::mem::transmute(PREV_EVAL_FRAME.load(Ordering::Acquire));
                _PyInterpreterState_SetEvalFrameFunc(pyo3::ffi::PyInterpreterState_Get(), prev);
            }
        }
        drop(installed);
        ACTIVE.set(ACTIVE.get() - 1);
    }
}

/// Call `func(*args, **kwargs)`, running every Python frame entered on this
/// thread during the call through the CFG executor.
#[pyfunction]
#[pyo3(signature = (func, /, *args, **kwargs))]
pub fn call_with_cfg_execution<'py>(
    func: &Bound<'py, PyAny>,
    args: &Bound<'py, PyTuple>,
    kwargs: Option<&Bound<'py, PyDict>>,
) -> PyResult<Bound<'py, PyAny>> {
    trampoline::init(func.py())?;
    let _guard = ExecGuard::new();
    func.call(args, kwargs)
}
