// Side effects seen while executing a Cfg, by the site that caused them: the
// Python frames entered and the torch ops run (see torch_ops.rs). An effect is
// recorded by the innermost executor activation only, so a call to a Python
// function shows up as one frame at the caller's Call, and what the function
// does at the callee's own sites. Frames CPython runs for us (trampolines,
// generator prologues) don't record, so their effects land on the caller.

use std::cell::Cell;
use std::collections::HashMap;
use std::fmt;
use std::sync::Mutex;

use pyo3::ffi::{self, PyObject};
use pyo3::prelude::*;

use crate::cfg::ir::{BlockId, Site, ValueId};

use super::ffi::{tr_bound_function, tr_frame_code, tr_frame_func, Frame};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum How {
    // The function the site called.
    Direct,
    // Called on the site's behalf by C code (a builtin, a type slot, ...).
    Indirect,
    // A generator or coroutine resumed (or thrown into).
    Resume,
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub enum Effect {
    Frame { func: String, how: How },
    TorchOp { op: String, args: String },
}

impl fmt::Display for Effect {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Effect::Frame { func, how: How::Direct } => write!(f, "call {func}"),
            Effect::Frame { func, how: How::Indirect } => write!(f, "call {func} (indirect)"),
            Effect::Frame { func, how: How::Resume } => write!(f, "resume {func}"),
            Effect::TorchOp { op, args } => write!(f, "torch {op}({args})"),
        }
    }
}

// For each site, every distinct sequence of effects one visit to it produced,
// with the number of visits that produced it. Visits with no effects aren't
// counted.
#[derive(Default)]
pub struct Table(HashMap<Site, Vec<(Vec<Effect>, u64)>>);

impl Table {
    pub fn get(&self, s: Site) -> &[(Vec<Effect>, u64)] {
        self.0.get(&s).map_or(&[], |v| v.as_slice())
    }

    // After the Cfg's blocks were renumbered (`map`: old id -> new id).
    pub fn renumber_blocks(&mut self, map: &[Option<BlockId>]) {
        self.0 = std::mem::take(&mut self.0)
            .into_iter()
            .filter_map(|(site, seqs)| match site {
                Site::Block(b) => map.get(b.idx()).copied().flatten().map(|n| (Site::Block(n), seqs)),
                s => Some((s, seqs)),
            })
            .collect();
    }
}

// The callable each Call site called (a bound method's function): the same
// object on every visit, or several. The optimizer identifies calls by it.
// Only callables that hold no user data are identified, held so that they
// can still be freed: functions, classes and builtins not bound to an object
// through a weakref, and method descriptors of static types (which can't be
// weakly referenced) strongly. Holding anything else, like a builtin bound to
// an object, would keep that object alive. Other calls get no entry.
pub enum Callee {
    One { ptr: usize, weak: bool, obj: Py<PyAny> },
    Many,
}

pub type Callees = HashMap<ValueId, Callee>;

impl Callee {
    // The callable, unless it's gone.
    pub fn get<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyAny>> {
        let Callee::One { weak, obj, .. } = self else {
            return None;
        };
        if !weak {
            return Some(obj.bind(py).clone());
        }
        let f = unsafe { ffi::PyWeakref_GetObject(obj.as_ptr()) };
        (f != unsafe { ffi::Py_None() }).then(|| unsafe { Bound::from_borrowed_ptr(py, f) })
    }

    pub fn describe(&self, py: Python<'_>) -> String {
        let Some(f) = self.get(py) else {
            return if matches!(self, Callee::Many) { "several".into() } else { "(gone)".into() };
        };
        let attr = |n: &str| f.getattr(n).and_then(|a| a.extract::<String>()).ok();
        match (attr("__module__"), attr("__qualname__")) {
            (Some(m), Some(q)) => format!("{m}.{q}"),
            (None, Some(q)) => q,
            _ => f.repr().map_or_else(|_| "?".into(), |r| r.to_string()),
        }
    }
}

pub unsafe fn record_callee(table: &Mutex<Callees>, v: ValueId, callable: *mut PyObject) {
    let f = tr_bound_function(callable);
    let weak = if ffi::PyFunction_Check(f) != 0 || ffi::PyType_Check(f) != 0 {
        true
    } else if ffi::PyCFunction_Check(f) != 0 {
        let owner = ffi::PyCFunction_GET_SELF(f);
        if !(owner.is_null() || ffi::PyModule_Check(owner) != 0 || ffi::PyType_Check(owner) != 0) {
            return;
        }
        true
    } else {
        let tp = ffi::Py_TYPE(f);
        let descr = tp == &raw mut ffi::PyMethodDescr_Type
            || tp == &raw mut ffi::PyClassMethodDescr_Type
            || tp == &raw mut ffi::PyWrapperDescr_Type;
        // PyDescrObject's d_type.
        if !descr || ffi::PyType_HasFeature((*(f as *mut ffi::PyDescrObject)).d_type, ffi::Py_TPFLAGS_HEAPTYPE) != 0 {
            return;
        }
        false
    };
    let py = Python::assume_attached();
    let mut table = table.lock().unwrap();
    match table.get(&v) {
        Some(Callee::Many) => {}
        Some(c @ Callee::One { ptr, .. }) if *ptr == f as usize && c.get(py).is_some() => {}
        Some(Callee::One { .. }) => {
            table.insert(v, Callee::Many);
        }
        None => {
            let obj = if weak { ffi::PyWeakref_NewRef(f, std::ptr::null_mut()) } else { ffi::Py_NewRef(f) };
            if obj.is_null() {
                ffi::PyErr_Clear();
                return;
            }
            table.insert(v, Callee::One { ptr: f as usize, weak, obj: Py::from_owned_ptr(py, obj) });
        }
    }
}

// One executor activation's view: the site it is at and the effects of the
// current visit to it.
pub struct Recorder {
    table: *const Mutex<Table>,
    site: Site,
    pending: Vec<Effect>,
    // The callable of the Call being run, until the frame it enters directly.
    callee: *mut PyObject,
    prev: *mut Recorder,
}

thread_local! {
    static CURRENT: Cell<*mut Recorder> = const { Cell::new(std::ptr::null_mut()) };
}

impl Recorder {
    fn commit(&mut self) {
        if self.pending.is_empty() {
            return;
        }
        let seq = std::mem::take(&mut self.pending);
        let mut table = unsafe { &*self.table }.lock().unwrap();
        let seqs = table.0.entry(self.site).or_default();
        match seqs.iter_mut().find(|(s, _)| *s == seq) {
            Some((_, n)) => *n += 1,
            None => seqs.push((seq, 1)),
        }
    }

    pub fn at(&mut self, site: Site) {
        if self.site != site {
            self.commit();
            self.site = site;
        }
    }

    pub fn set_callee(&mut self, callee: *mut PyObject) {
        self.callee = callee;
    }
}

struct Scope(*mut Recorder);

impl Drop for Scope {
    fn drop(&mut self) {
        unsafe {
            (*self.0).commit();
            CURRENT.set((*self.0).prev);
        }
    }
}

// Runs `f` with a recorder for `table`, starting at `site`, as the innermost
// one on this thread.
pub fn scope<R>(table: &Mutex<Table>, site: Site, f: impl FnOnce(*mut Recorder) -> R) -> R {
    let mut rec = Recorder {
        table,
        site,
        pending: vec![],
        callee: std::ptr::null_mut(),
        prev: CURRENT.get(),
    };
    let p: *mut Recorder = &mut rec;
    CURRENT.set(p);
    let _scope = Scope(p);
    f(p)
}

// Before the eval-frame hook runs `frame`, holding the GIL.
pub unsafe fn frame_entered(frame: *mut Frame, resumed: bool) {
    let rec = CURRENT.get();
    if rec.is_null() {
        return;
    }
    let func = tr_frame_func(frame);
    let how = if resumed {
        How::Resume
    } else if !(*rec).callee.is_null() && tr_bound_function((*rec).callee) == func {
        (*rec).callee = std::ptr::null_mut();
        How::Direct
    } else {
        How::Indirect
    };
    let func = func_name(func, tr_frame_code(frame));
    (*rec).pending.push(Effect::Frame { func, how });
}

// "module.qualname". A throw into a generator enters with the exception set,
// so it's put aside around the lookups.
unsafe fn func_name(func: *mut PyObject, code: *mut PyObject) -> String {
    let saved = ffi::PyErr_GetRaisedException();
    let py = Python::assume_attached();
    let code = Bound::from_borrowed_ptr(py, code);
    let qualname = code
        .getattr("co_qualname")
        .and_then(|q| q.extract::<String>())
        .unwrap_or_else(|_| "?".into());
    let module = (!func.is_null())
        .then(|| Bound::from_borrowed_ptr(py, func).getattr("__module__").and_then(|m| m.extract::<String>()).ok())
        .flatten();
    ffi::PyErr_SetRaisedException(saved);
    match module {
        Some(m) => format!("{m}.{qualname}"),
        None => qualname,
    }
}

// From the PreDispatch hook, maybe without the GIL.
pub fn torch_op(op: &str, args: &str) {
    let rec = CURRENT.get();
    if !rec.is_null() {
        let op = Effect::TorchOp {
            op: op.into(),
            args: args.into(),
        };
        unsafe { (*rec).pending.push(op) };
    }
}
