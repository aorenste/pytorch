// CFG executor: a PEP 523 eval-frame hook that runs each Python frame by
// converting its code object to a CFG (see cfg/) and interpreting that, instead
// of CPython's bytecode loop. Anything it can't run faithfully fails hard with
// an exception rather than falling back to CPython.

pub mod effects;
pub mod ffi;
mod live;
mod run;
pub mod torch_ops;
mod trampoline;
pub mod types;

use std::cell::Cell;
use std::collections::{HashMap, HashSet};
use std::os::raw::{c_int, c_void};
use std::sync::atomic::{AtomicBool, AtomicPtr, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

use crate::cfg::ir::{Annotations, BlockId, Cfg, Op, Site, ValueId};
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
    // For an instr with several outputs, the Project instr receiving each
    // (by output position).
    pub projects: Vec<Vec<Option<ValueId>>>,
    // Number of new outputs of each Op::Bytecode instr.
    pub nouts: Vec<u32>,
    // By block, then version: the body as run_blocks steps through it.
    pub code: Vec<Vec<run::Code>>,
    // Each LoadGlobal's cache, at its index in `global_slot` (by instr).
    pub globals: Box<[ffi::GilCell<ffi::GlobalCache>]>,
    pub global_slot: Vec<u32>,
    pub live: live::Liveness,
    // By block.
    pub origins: Vec<Option<Prepared>>,
    // What has been observed running the Cfg. Shared with the Prepared of an
    // optimized copy of it, whose values keep their ids.
    pub effects: Arc<Mutex<effects::Table>>,
    // By value: the join of every object seen there.
    pub types: Arc<Mutex<Vec<Option<types::Ty>>>>,
    pub callees: Arc<Mutex<effects::Callees>>,
    // For each Op::Aten instr, the operator overload's C++ binding.
    pub aten: HashMap<ValueId, Py<PyAny>>,
}

impl Prepared {
    // `prev`: the Prepared of the Cfg this one replaces (with the same value
    // ids), whose observations carry over.
    fn new(cfg: Arc<Cfg>, prev: Option<&Prepared>) -> PyResult<Self> {
        let n = cfg.instrs.len();
        let mut projects: Vec<Vec<Option<ValueId>>> = vec![vec![]; n];
        let mut nouts = vec![0u32; n];
        let mut flags = vec![0u8; n];
        let body = || cfg.blocks.iter().flat_map(|b| &b.versions).flat_map(|ver| &ver.body);
        for v in body() {
            let ins = &cfg[*v];
            match &ins.op {
                Op::Project(k) => {
                    let ps = &mut projects[ins.args[0].idx()];
                    if ps.len() <= *k as usize {
                        ps.resize(*k as usize + 1, None);
                    }
                    ps[*k as usize] = Some(*v);
                }
                Op::Bytecode { opcode, arg } => {
                    let sig = opcodes::signature(*opcode, *arg, false).expect("converted op has a signature");
                    nouts[v.idx()] = sig.outputs.iter().filter(|o| **o == Out::New).count() as u32;
                }
                _ => {}
            }
            flags[v.idx()] = match ins.op {
                Op::Yield => run::FLAG_YIELD,
                Op::Call { .. } => run::FLAG_CALL,
                Op::Bytecode {
                    opcode: opcodes::CALL_FUNCTION_EX,
                    ..
                } => run::FLAG_CALL_EX,
                _ => 0,
            } | if ins.op.can_raise() { run::FLAG_RAISE } else { 0 };
        }
        // A multi-output instr's values are its Projects'.
        for v in body() {
            if cfg[*v].op.produces_value() && projects[v.idx()].is_empty() {
                flags[v.idx()] |= run::FLAG_VALUE;
            }
        }
        let mut aten = HashMap::new();
        for v in cfg.blocks.iter().flat_map(|b| &b.versions).flat_map(|ver| &ver.body) {
            if let Op::Aten(name) = &cfg[*v].op {
                aten.insert(*v, resolve_aten(name)?);
            }
        }
        let live = live::compute(&cfg);
        let code = run::Code::build(&cfg, &live, &flags);
        let mut global_slot = vec![u32::MAX; n];
        let mut nglobals = 0;
        for v in body() {
            if let Op::LoadGlobal(_) = cfg[*v].op {
                global_slot[v.idx()] = nglobals;
                nglobals += 1;
            }
        }
        let globals = (0..nglobals).map(|_| ffi::GilCell::new(ffi::GlobalCache::default())).collect();
        if let Some(y) = live.bad_yield {
            return Err(pyo3::exceptions::PyNotImplementedError::new_err(format!(
                "CFG executor: a value not on the stack is live across the yield at offset {}",
                cfg[y].origin * 2
            )));
        }
        // An origin the previous Cfg already had is the same Cfg.
        let origins = cfg
            .blocks
            .iter()
            .map(|b| {
                b.origin
                    .clone()
                    .map(|o| {
                        let same = prev.and_then(|p| p.find(&o));
                        Prepared::new(o, same)
                    })
                    .transpose()
            })
            .collect::<PyResult<_>>()?;
        let (effects, types, callees) = match prev {
            Some(p) => {
                p.types.lock().unwrap().resize(n, None);
                (p.effects.clone(), p.types.clone(), p.callees.clone())
            }
            None => (Arc::default(), Arc::new(Mutex::new(vec![None; n])), Arc::default()),
        };
        Ok(Prepared {
            cfg,
            projects,
            nouts,
            code,
            globals,
            global_slot,
            live,
            origins,
            effects,
            types,
            callees,
            aten,
        })
    }

    // The Prepared of `cfg` here or among the origins.
    fn find(&self, cfg: &Arc<Cfg>) -> Option<&Prepared> {
        if Arc::ptr_eq(&self.cfg, cfg) {
            return Some(self);
        }
        self.origins.iter().flatten().find_map(|o| o.find(cfg))
    }
}

// torch.ops.<ns>.<op>.<overload>._op for "ns::op.overload" (or "ns::op").
fn resolve_aten(name: &str) -> PyResult<Py<PyAny>> {
    let err = || pyo3::exceptions::PyValueError::new_err(format!("not an operator name: {name}"));
    let (ns, rest) = name.split_once("::").ok_or_else(err)?;
    let (op, overload) = rest.split_once('.').unwrap_or((rest, "default"));
    Python::attach(|py| {
        let ops = py.import("torch")?.getattr("ops")?;
        Ok(ops.getattr(ns)?.getattr(op)?.getattr(overload)?.getattr("_op")?.unbind())
    })
}

// The classes whose Python operators run as plain aten ops: torch.Tensor and
// nn.Parameter (whose __torch_function__ is disabled).
fn plain_tensor_types() -> [usize; 2] {
    static TYPES: OnceLock<[usize; 2]> = OnceLock::new();
    *TYPES.get_or_init(|| {
        Python::attach(|py| {
            let get = |m: &str, n: &str| py.import(m).and_then(|m| m.getattr(n)).map_or(0, |t| t.as_ptr() as usize);
            [get("torch", "Tensor"), get("torch.nn", "Parameter")]
        })
    })
}

pub(crate) unsafe fn is_plain_tensor(obj: *mut pyo3::ffi::PyObject) -> bool {
    plain_tensor_types().contains(&(pyo3::ffi::Py_TYPE(obj) as usize))
}

// What a frame of the code runs: the current Cfg, prepared.
pub struct Plan {
    pub prep: Prepared,
    // Where a generator frame resumes, by the code unit its prev_instr points
    // at: each Yield, and RETURN_GENERATOR for a just-created generator. The
    // instr after it is the RESUME's Op::Resume. (block, version, index).
    // Outlining and optimization keep the blocks holding them, so they are all
    // in the top-level Cfg, and a generator suspended under one Plan resumes
    // at the same place under its replacement.
    pub resume: HashMap<u32, (BlockId, usize, usize)>,
}

impl Plan {
    fn new(cfg: Cfg, prev: Option<&Plan>) -> PyResult<Self> {
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
        Ok(Plan {
            resume,
            prep: Prepared::new(Arc::new(cfg), prev.map(|p| &p.prep))?,
        })
    }
}

// Per-code-object data, cached in co_extra and freed with the code object.
pub struct Compiled {
    // The current Plan, the last of `plans`. Replaced when the Cfg is
    // optimized; frames already running keep the Plan they started with, so
    // every Plan lives as long as the Compiled.
    plan: AtomicPtr<Plan>,
    // Boxed so a Plan doesn't move when the Vec grows.
    #[allow(clippy::vec_box)]
    plans: Mutex<Vec<Box<Plan>>>,
    // Whether the code was optimized after its first run (see OPTIMIZE).
    auto_optimized: AtomicBool,
    pub flags: u32,
    pub consts: Py<PyTuple>,
    pub names: Py<PyTuple>,
}

impl Compiled {
    fn new(code: &Bound<'_, PyAny>) -> PyResult<Self> {
        let info = crate::cfg::python::code_info(code)?;
        let outline = *OUTLINE.lock().unwrap();
        let cfg = crate::cfg::build(info, true, outline).map_err(crate::cfg::python::to_pyerr)?;
        let mut plan = Box::new(Plan::new(cfg, None)?);
        Ok(Compiled {
            plan: AtomicPtr::new(&mut *plan),
            plans: Mutex::new(vec![plan]),
            auto_optimized: AtomicBool::new(false),
            flags: code.getattr("co_flags")?.extract()?,
            consts: code.getattr("co_consts")?.downcast_into::<PyTuple>()?.unbind(),
            names: code.getattr("co_names")?.downcast_into::<PyTuple>()?.unbind(),
        })
    }

    pub fn plan(&self) -> &Plan {
        unsafe { &*self.plan.load(Ordering::Acquire) }
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

// Whether the executor optimizes every block of a code's CFG after the first
// time a frame of it returns (for testing the optimizer).
static OPTIMIZE: AtomicBool = AtomicBool::new(false);

/// Optimize every block of each code's CFG after a frame of it first returns.
#[pyfunction]
pub fn _set_cfg_optimize(on: bool) {
    OPTIMIZE.store(on, Ordering::Relaxed);
}

// Whether the executor records side effects and value types (see
// cfg_observed). The optimizer works from them.
pub static RECORD: AtomicBool = AtomicBool::new(true);

/// Turn recording of side effects and value types on or off for executor
/// calls started from now on.
#[pyfunction]
pub fn _set_cfg_record(on: bool) {
    RECORD.store(on, Ordering::Relaxed);
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
    effects::frame_entered(frame, gen_frame);
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
    let plan = c.plan();
    let start = if gen_frame {
        let unit = tr_frame_get_instr(frame) as u32;
        let Some(&(b, ver, i)) = plan.resume.get(&unit) else {
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
    let outcome = run::run(c, &plan.prep, frame, start);
    if matches!(outcome, run::Outcome::Return(_))
        && OPTIMIZE.load(Ordering::Relaxed)
        && !OPTIMIZING.get()
        && !c.auto_optimized.swap(true, Ordering::Relaxed)
    {
        let py = Python::assume_attached();
        if let Err(e) = optimize(py, c, None) {
            e.write_unraisable(py, None);
        }
    }
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

/// The CFG the executor runs for `code` (a code object, or a function via
/// `__code__`), with what was observed running it so far: the side effects at
/// each site (Python frames entered, torch ops run) and the type of each value.
#[pyfunction]
#[pyo3(signature = (code, effects=true, types=true))]
pub fn cfg_observed(code: &Bound<'_, PyAny>, effects: bool, types: bool) -> PyResult<String> {
    let c = unsafe { lookup(code)? };
    let plan = c.plan();
    let mut by_cfg = HashMap::new();
    let mut work = vec![&plan.prep];
    while let Some(p) = work.pop() {
        by_cfg.insert(Arc::as_ptr(&p.cfg), p);
        work.extend(p.origins.iter().flatten());
    }
    let observed = Observed { by_cfg, effects, types };
    Ok(plan.prep.cfg.render_with(false, false, &observed))
}

/// Optimize block `block` of the CFG the executor runs for `code` (a code
/// object, or a function via `__code__`); later calls run the result. See
/// cfg/optimize.rs.
#[pyfunction]
pub fn optimize_block(code: &Bound<'_, PyAny>, block: u32) -> PyResult<()> {
    let c = unsafe { lookup(code)? };
    optimize(code.py(), c, Some(BlockId(block)))
}

thread_local! {
    // Set while optimizing, which runs Python code (see in_graph) that
    // mustn't be optimized itself meanwhile.
    static OPTIMIZING: Cell<bool> = const { Cell::new(false) };
}

// Optimizes `block` of c's Cfg, or every block it can (others are left as
// they are), and swaps in the result. Blocks are renumbered, so block-site
// effects move with their blocks.
fn optimize(py: Python<'_>, c: &Compiled, block: Option<BlockId>) -> PyResult<()> {
    struct Reset(bool);
    impl Drop for Reset {
        fn drop(&mut self) {
            OPTIMIZING.set(self.0);
        }
    }
    let _reset = Reset(OPTIMIZING.replace(true));
    let plan = c.plan();
    let blocks: Vec<BlockId> = match block {
        Some(b) => vec![b],
        None => plan.prep.cfg.block_ids().collect(),
    };
    let (movable, aten) = observed_ops(py, &plan.prep, &blocks);
    let mut cfg = (*plan.prep.cfg).clone();
    let renumbered = crate::cfg::optimize::optimize_blocks(&mut cfg, &blocks, &movable, &aten, block.is_some())
        .map_err(pyo3::exceptions::PyValueError::new_err)?;
    crate::cfg::verify::verify(&cfg).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("CFG verification failed: {e}")))?;
    let mut plans = c.plans.lock().unwrap();
    if !std::ptr::eq(c.plan(), plan) {
        return Err(pyo3::exceptions::PyRuntimeError::new_err("the CFG was optimized concurrently"));
    }
    plan.prep.effects.lock().unwrap().renumber_blocks(&renumbered);
    let mut new = Box::new(Plan::new(cfg, Some(plan))?);
    c.plan.store(&mut *new, Ordering::Release);
    plans.push(new);
    Ok(())
}

// The instrs of `blocks` that, as observed, run no code that could look at
// the frame or care when a local's old value is released, so they may sit in
// a pure block (see cfg/optimize.rs): roughly what Dynamo would put in a
// graph. Arithmetic, attribute and subscript ops count when their operands
// were only ever builtin scalars, strings, tensors, tuples or lists (or, for
// attributes, modules) and no Python frame was entered there; calls when they
// always called the same function, one Dynamo puts in its graphs. Also returns
// the operators that may be replaced by the aten op they ran: every visit ran
// that one op, the left operand was always a plain tensor (a reflected
// operator's aten args don't follow the operand order) and each operand was
// always a plain tensor or one builtin scalar type.
fn observed_ops(
    py: Python<'_>,
    prep: &Prepared,
    blocks: &[BlockId],
) -> (HashSet<ValueId>, HashMap<ValueId, crate::cfg::optimize::AtenOp>) {
    use crate::cfg::ir::Builtin;
    use crate::cfg::optimize::{Arg, AtenOp};
    use types::{Kind, Ty};
    let mut aten = HashMap::new();
    let cfg = &prep.cfg;
    // Calls to judge with is_in_graph, which runs Python code (observed by the
    // executor, so not while holding the tables' locks).
    let mut calls: Vec<(ValueId, Py<PyAny>)> = vec![];
    let mut out = HashSet::new();
    {
        let types = prep.types.lock().unwrap();
        let effects = prep.effects.lock().unwrap();
        let callees = prep.callees.lock().unwrap();
        let plain = |v: &ValueId, module_ok: bool| match types[v.idx()].as_ref() {
            Some(Ty::Union(ks)) => ks.iter().all(|k| match k {
                Kind::None
                | Kind::Bool(_)
                | Kind::Int(_)
                | Kind::Float
                | Kind::Str(_)
                | Kind::Tensor(_)
                | Kind::Tuple(_)
                | Kind::VarTuple(_)
                | Kind::List(_) => true,
                Kind::Object(c) => module_ok && c.is(&raw const pyo3::ffi::PyModule_Type),
                _ => false,
            }),
            _ => false,
        };
        let bodies = blocks.iter().filter(|b| b.idx() < cfg.blocks.len()).flat_map(|b| cfg[*b].versions.first());
        for v in bodies.flat_map(|ver| &ver.body) {
            let ins = &cfg[*v];
            let no_frames = || {
                effects
                    .get(Site::Instr(*v))
                    .iter()
                    .all(|(seq, _)| !seq.iter().any(|e| matches!(e, effects::Effect::Frame { .. })))
            };
            let ok = match &ins.op {
                Op::LoadLocal { .. }
                | Op::LoadLocalChecked { .. }
                | Op::StoreLocal { .. }
                | Op::DeleteLocal { .. }
                | Op::Resume
                | Op::EvalBreaker
                | Op::LoadGlobal(_) => true,
                _ if callees.contains_key(v) => {
                    if let (Some(f), true) = (callees[v].get(py), no_frames()) {
                        calls.push((*v, f.unbind()));
                    }
                    false
                }
                op if !op.can_raise() => true,
                Op::LoadAttr(_) | Op::LoadMethod(_) => no_frames() && plain(&ins.args[0], true),
                Op::Binary(_) | Op::Compare(_) | Op::ToBool | Op::GetIter | Op::Bytecode { .. } => {
                    no_frames() && ins.args.iter().all(|a| plain(a, false))
                }
                _ => false,
            };
            if ok {
                out.insert(*v);
            }
            let replaceable = matches!(
                ins.op,
                Op::Binary(_)
                    | Op::Compare(_)
                    | Op::Bytecode {
                        opcode: opcodes::UNARY_NEGATIVE | opcodes::UNARY_INVERT,
                        ..
                    }
            );
            let arg = |a: &ValueId| match types[a.idx()].as_ref() {
                Some(Ty::Union(ks)) => match ks.as_slice() {
                    [Kind::Tensor(t)] if plain_tensor_types().iter().any(|p| t.class().is(*p as *const _)) => {
                        Some(Arg::PlainTensor)
                    }
                    [Kind::Int(_)] => Some(Arg::Scalar(Builtin::Int)),
                    [Kind::Float] => Some(Arg::Scalar(Builtin::Float)),
                    [Kind::Bool(_)] => Some(Arg::Scalar(Builtin::Bool)),
                    _ => None,
                },
                _ => None,
            };
            if let (true, [(seq, _)]) = (replaceable, effects.get(Site::Instr(*v))) {
                if let [effects::Effect::TorchOp { op, .. }] = seq.as_slice() {
                    let args: Option<Vec<Arg>> = ins.args.iter().map(arg).collect();
                    if let Some(args) = args.filter(|a| a.first() == Some(&Arg::PlainTensor)) {
                        aten.insert(*v, AtenOp { name: op.clone(), args });
                    }
                }
            }
        }
    }
    let mut in_graph: HashMap<*mut pyo3::ffi::PyObject, bool> = HashMap::new();
    for (v, f) in calls {
        if *in_graph.entry(f.as_ptr()).or_insert_with(|| is_in_graph(f.bind(py))) {
            out.insert(v);
        }
    }
    (out, aten)
}

// Whether Dynamo puts calls to `f` in its graphs.
fn is_in_graph(f: &Bound<'_, PyAny>) -> bool {
    f.py()
        .import("torch._dynamo.trace_rules")
        .and_then(|m| m.call_method1("lookup", (f,)))
        .and_then(|r| r.getattr("__name__"))
        .and_then(|n| n.extract::<String>())
        .is_ok_and(|n| n == "TorchInGraphFunctionVariable")
}

// The Compiled of `code` (or of a function's `__code__`), which lives as long
// as the code object.
unsafe fn lookup<'a>(code: &Bound<'_, PyAny>) -> PyResult<&'a Compiled> {
    let code = if code.hasattr("__code__")? {
        code.getattr("__code__")?
    } else {
        code.clone()
    };
    let mut p: *mut c_void = std::ptr::null_mut();
    let index = extra_index();
    if index < 0 || PyUnstable_Code_GetExtra(code.as_ptr(), index, &mut p) < 0 {
        return Err(PyErr::fetch(code.py()));
    }
    if p.is_null() || p == trampoline::mark() {
        return Err(pyo3::exceptions::PyValueError::new_err("the CFG executor hasn't run this code"));
    }
    Ok(&*(p as *const Compiled))
}

struct Observed<'a> {
    by_cfg: HashMap<*const Cfg, &'a Prepared>,
    effects: bool,
    types: bool,
}

impl Annotations for Observed<'_> {
    fn notes(&self, cfg: &Cfg, site: Site) -> Vec<String> {
        if !self.effects {
            return vec![];
        }
        let prep = self.by_cfg[&(cfg as *const Cfg)];
        let mut lines = vec![];
        if let Site::Instr(v) = site {
            // Describing runs Python code, so not under the lock.
            let py = unsafe { Python::assume_attached() };
            let callee = prep.callees.lock().unwrap().get(&v).map(|c| match c {
                effects::Callee::One { ptr, weak, obj } => effects::Callee::One {
                    ptr: *ptr,
                    weak: *weak,
                    obj: obj.clone_ref(py),
                },
                effects::Callee::Many => effects::Callee::Many,
            });
            if let Some(c) = callee {
                lines.push(format!("callee {}", c.describe(py)));
            }
        }
        let table = prep.effects.lock().unwrap();
        for (seq, n) in table.get(site) {
            let s = if *n == 1 { "" } else { "s" };
            lines.push(format!("effects [{n} visit{s}]:"));
            lines.extend(seq.iter().map(|e| format!("  {e}")));
        }
        lines
    }

    fn value(&self, cfg: &Cfg, v: ValueId) -> Option<String> {
        if !self.types {
            return None;
        }
        let table = self.by_cfg[&(cfg as *const Cfg)].types.lock().unwrap();
        table[v.idx()].as_ref().map(|t| t.to_string())
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
    let _ops = (RECORD.load(Ordering::Relaxed) || torch_ops::printing()).then(torch_ops::OpHookGuard::new);
    func.call(args, kwargs)
}
