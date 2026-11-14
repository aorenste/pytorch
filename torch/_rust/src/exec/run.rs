// Interprets one activation of a Compiled CFG on a CPython frame. Locals live
// in the frame (localsplus); SSA values are owned references in `vals`.

use std::os::raw::c_int;

use pyo3::ffi::{self, PyObject};

use crate::cfg::ir::*;
use crate::cfg::opcodes;

use super::ffi::*;
use super::{Compiled, Prepared};

// How an exception was raised: Error adds a traceback entry for this frame
// (the default loop's `error:`); Reraise doesn't (`exception_unwind:`).
#[derive(Clone, Copy, PartialEq, Eq)]
enum Exc {
    Error,
    Reraise,
}

type R = Result<(), Exc>;

pub enum Outcome {
    Return(*mut PyObject),
    // A generator suspended at the Yield with origin `unit`, its stack saved
    // in the frame.
    Yield { value: *mut PyObject, unit: u32 },
    // An exception is set.
    Error,
    // Only from an outlined Cfg: it left with these values as its owner's
    // outputs, or unwound to its owner's handlers[k] with these args (an
    // exception is set).
    Leave(Vec<*mut PyObject>),
    Outer(usize, Vec<*mut PyObject>),
}

// Where an exception goes from a raise point.
enum Unwound {
    Block(BlockId),
    Outer(usize, Vec<*mut PyObject>),
    Propagate,
}

pub enum Start {
    Entry,
    // A generator frame resumed (or thrown into) at body[i] of version `ver`
    // of block b: a Yield, or RETURN_GENERATOR on first resume. The sent value
    // is on top of the frame's stack, above the stack saved at suspension.
    Resume { b: BlockId, ver: usize, i: usize, throw: bool },
}

fn status(rc: c_int) -> R {
    match rc {
        0 => Ok(()),
        -2 => Err(Exc::Reraise),
        _ => Err(Exc::Error),
    }
}

fn check(p: *mut PyObject) -> Result<*mut PyObject, Exc> {
    if p.is_null() {
        Err(Exc::Error)
    } else {
        Ok(p)
    }
}

unsafe fn xincref(p: *mut PyObject) -> *mut PyObject {
    ffi::Py_XINCREF(p);
    p
}

unsafe fn new_bool(b: bool) -> *mut PyObject {
    xincref(if b { ffi::Py_True() } else { ffi::Py_False() })
}

struct Act<'a> {
    p: &'a Prepared,
    cfg: &'a Cfg,
    f: *mut Frame,
    locals: *mut *mut PyObject,
    consts: *mut PyObject,
    names: *mut PyObject,
    vals: Vec<*mut PyObject>,
    tstate: *mut ffi::PyThreadState,
}

// Runs the frame, which the caller has linked.
pub unsafe fn run(c: &Compiled, f: *mut Frame, start: Start) -> Outcome {
    let mut act = Act {
        p: &c.prep,
        cfg: &c.prep.cfg,
        f,
        locals: tr_frame_localsplus(f),
        consts: c.consts.as_ptr(),
        names: c.names.as_ptr(),
        vals: vec![std::ptr::null_mut(); c.prep.cfg.instrs.len()],
        tstate: ffi::PyThreadState_Get(),
    };
    let result = match start {
        Start::Entry => act.run_blocks(ENTRY, None),
        Start::Resume { b, ver, i, throw } => match act.resume(b, ver, i, throw) {
            Some(Ok(h)) => act.run_blocks(h, None),
            Some(Err(next)) => act.run_blocks(b, Some((ver, next))),
            None => Outcome::Error,
        },
    };
    for v in act.vals.drain(..) {
        ffi::Py_XDECREF(v);
    }
    result
}

impl Act<'_> {
    // Moves the frame's saved stack back into SSA values. Returns where to
    // continue: Ok(the handler a throw unwound to), or Err(the index after the
    // resume point); None if the frame is done (an exception is set).
    unsafe fn resume(&mut self, b: BlockId, ver: usize, i: usize, throw: bool) -> Option<Result<BlockId, usize>> {
        let cfg = self.cfg;
        let v = cfg[b].versions[ver].body[i];
        let sent = tr_frame_pop(self.f);
        let saved: &[ValueId] = match cfg[v].op {
            Op::Yield => &cfg[v].args[..cfg[v].args.len() - 1],
            _ => &[],
        };
        if tr_frame_stack_depth(self.f) as usize != saved.len() {
            ffi::Py_DECREF(sent);
            super::fail(
                "CFG executor: unexpected stack depth in a suspended generator",
                ffi::PyExc_RuntimeError,
            );
            return None;
        }
        for a in saved.iter().rev() {
            let obj = tr_frame_pop(self.f);
            self.set(*a, obj);
        }
        if throw {
            ffi::Py_DECREF(sent);
            return match self.unwind(b, ver, i, Exc::Error) {
                Unwound::Block(h) => Some(Ok(h)),
                Unwound::Propagate => None,
                Unwound::Outer(..) => unreachable!("resume points are in the top-level Cfg"),
            };
        }
        self.set(v, sent);
        self.drop_all(&self.p.live.versions[b.idx()][ver].after_kill[i]);
        Some(Err(i + 1))
    }

    fn val(&self, v: ValueId) -> *mut PyObject {
        self.vals[v.idx()]
    }

    unsafe fn set(&mut self, v: ValueId, obj: *mut PyObject) {
        let old = std::mem::replace(&mut self.vals[v.idx()], obj);
        ffi::Py_XDECREF(old);
    }

    fn args(&self, ins: &Instr) -> Vec<*mut PyObject> {
        ins.args.iter().map(|a| self.val(*a)).collect()
    }

    unsafe fn local(&self, slot: u32) -> *mut *mut PyObject {
        self.locals.add(slot as usize)
    }

    unsafe fn name(&self, i: u32) -> *mut PyObject {
        ffi::PyTuple_GetItem(self.names, i as ffi::Py_ssize_t)
    }

    unsafe fn konst(&self, i: u32) -> *mut PyObject {
        ffi::PyTuple_GetItem(self.consts, i as ffi::Py_ssize_t)
    }

    // Moves the outputs of a multi-output instr into its Project instrs.
    unsafe fn distribute(&mut self, v: ValueId, outs: &[*mut PyObject]) {
        let projects = &self.p.projects[v.idx()];
        let mut used = vec![false; outs.len()];
        for (k, p) in projects {
            used[*k as usize] = true;
            let obj = outs[*k as usize];
            self.set(*p, obj);
        }
        // Outputs nobody reads (simplify never drops Projects, but be safe).
        for (k, obj) in outs.iter().enumerate() {
            if !used[k] {
                ffi::Py_XDECREF(*obj);
            }
        }
    }

    unsafe fn drop_all(&mut self, vs: &[ValueId]) {
        for v in vs {
            self.set(*v, std::ptr::null_mut());
        }
    }

    // Drops what is left of a version's values when control leaves the block.
    unsafe fn clear(&mut self, b: BlockId, ver: usize) {
        let blk = &self.cfg[b];
        for v in blk.params.iter().chain(&blk.versions[ver].body) {
            self.set(*v, std::ptr::null_mut());
        }
    }

    // Enters `target` (after clear()), with owned values for its explicit
    // params in order and `implicit` supplying the others.
    unsafe fn enter(
        &mut self,
        target: BlockId,
        explicit: Vec<*mut PyObject>,
        mut implicit: impl FnMut(ParamKind) -> *mut PyObject,
    ) {
        let mut explicit = explicit.into_iter();
        for p in &self.cfg[target].params {
            let kind = self.cfg.param_kind(*p).expect("block params are Params");
            let obj = if kind.is_implicit() {
                implicit(kind)
            } else {
                explicit.next().expect("arg count checked by the verifier")
            };
            self.set(*p, obj);
        }
    }

    // The version of `b` to run: the first whose guard holds (no Check kinds
    // exist yet, so only an empty guard holds). None: run its origin.
    unsafe fn select(&mut self, b: BlockId) -> Option<usize> {
        let ver = self.cfg[b].versions.iter().position(|v| v.guard.checks.is_empty())?;
        self.drop_all(&self.p.live.versions[b.idx()][ver].entry_kill);
        Some(ver)
    }

    // No version of `b` matched: run its origin on b's params, then continue
    // as b would with the outputs it leaves with.
    unsafe fn miss(&mut self, b: BlockId) -> Result<BlockId, Outcome> {
        let origin = self.p.origins[b.idx()].as_ref().expect("verified: a block without a matching version has an origin");
        let mut sub = Act {
            p: origin,
            cfg: &origin.cfg,
            vals: vec![std::ptr::null_mut(); origin.cfg.instrs.len()],
            ..*self
        };
        for (p, q) in self.cfg[b].params.iter().zip(&origin.cfg[ENTRY].params) {
            let obj = std::mem::replace(&mut self.vals[p.idx()], std::ptr::null_mut());
            sub.set(*q, obj);
        }
        let result = sub.run_blocks(ENTRY, None);
        for v in sub.vals.drain(..) {
            ffi::Py_XDECREF(v);
        }
        match result {
            Outcome::Leave(outs) => {
                let next = self.take_exit(b, &outs, None);
                for o in outs {
                    ffi::Py_XDECREF(o);
                }
                next
            }
            Outcome::Outer(k, args) => {
                let h = self.cfg[b].handlers[k];
                self.enter_handler(h, args);
                Ok(h)
            }
            Outcome::Error => Err(Outcome::Error),
            Outcome::Return(_) | Outcome::Yield { .. } => {
                unreachable!("verified: an outlined Cfg leaves rather than returns, and has no resume points")
            }
        }
    }

    // Takes b's exit for its outputs `outs` (borrowed): Ok(the next block, or
    // the handler a raise went to), or the frame's (or outlined Cfg's)
    // outcome. `clear` is the version that ran, whose values are dropped once
    // the args are taken.
    unsafe fn take_exit(&mut self, b: BlockId, outs: &[*mut PyObject], clear: Option<usize>) -> Result<BlockId, Outcome> {
        let blk = &self.cfg[b];
        let out = |o: &OutIdx| outs[*o as usize];
        let how = match &blk.term {
            Terminator::Raise { args, .. } => {
                tr_frame_set_instr(self.f, blk.term_origin as c_int);
                let exc = args.first().map_or(std::ptr::null_mut(), out);
                let cause = args.get(1).map_or(std::ptr::null_mut(), out);
                Some(status(tr_do_raise(exc, cause)).unwrap_err())
            }
            Terminator::Reraise { exc, lasti, .. } => {
                let instr = lasti.map_or(blk.term_origin as c_int, |l| ffi::PyLong_AsLong(out(&l)) as c_int);
                tr_frame_set_instr(self.f, instr);
                ffi::PyErr_SetRaisedException(xincref(out(exc)));
                Some(Exc::Reraise)
            }
            _ => None,
        };
        if let (Some(how), Some(unwind)) = (how, blk.term.unwind()) {
            if how == Exc::Error {
                tr_traceback_here(self.f);
            }
            let take = |os: &[OutIdx]| -> Vec<*mut PyObject> { os.iter().map(|o| xincref(out(o))).collect() };
            let TermUnwind::Block(k, args) = unwind else {
                return Err(Outcome::Error);
            };
            let (h, args) = (blk.handlers[*k], take(args));
            if let Some(vi) = clear {
                self.clear(b, vi);
            }
            self.enter_handler(h, args);
            return Ok(h);
        }
        let exit = match &blk.term {
            Terminator::Jump(e) => e,
            Terminator::Branch { cond, then, else_ } => {
                if outs[*cond as usize] == ffi::Py_True() {
                    then
                } else {
                    else_
                }
            }
            Terminator::Raise { .. } | Terminator::Reraise { .. } => unreachable!("handled above"),
            Terminator::Unreachable => unreachable!("verified CFGs have no unreachable terminators"),
        };
        let take = |os: &[OutIdx]| -> Vec<*mut PyObject> { os.iter().map(|o| xincref(outs[*o as usize])).collect() };
        match exit {
            Exit::Block(edge) => {
                let args = take(&edge.args);
                if let Some(vi) = clear {
                    self.clear(b, vi);
                }
                self.enter(edge.target, args, |_| unreachable!("edges have no implicit params"));
                Ok(edge.target)
            }
            Exit::Return(o) => {
                // A finished frame's f_lineno is its return's line.
                tr_frame_set_instr(self.f, blk.term_origin as c_int);
                Err(Outcome::Return(xincref(outs[*o as usize])))
            }
            Exit::Leave(os) => Err(Outcome::Leave(take(os))),
        }
    }

    // Where CPython checks its eval breaker (RESUME, after calls,
    // JUMP_BACKWARD), run it; it can raise (e.g. KeyboardInterrupt).
    unsafe fn breaker(&self) -> R {
        if tr_eval_breaker(self.tstate) != 0 && !super::trampoline::eval_breaker() {
            return Err(Exc::Error);
        }
        Ok(())
    }

    // An exception is set; find where the segment covering body[at] of the
    // version sends it, and enter the handler if it's in this Cfg.
    unsafe fn unwind(&mut self, b: BlockId, ver: usize, at: usize, how: Exc) -> Unwound {
        if how == Exc::Error {
            // A failure replaces the pending exception, which is what CPython
            // does too.
            tr_traceback_here(self.f);
        }
        let cfg = self.cfg;
        let blk = &cfg[b];
        let take = |args: &[ValueId]| -> Vec<*mut PyObject> { args.iter().map(|a| xincref(self.val(*a))).collect() };
        let (h, args) = match super::live::segment_at(&blk.versions[ver], at).map(|s| &s.handler) {
            None | Some(Handler::Propagate) => return Unwound::Propagate,
            Some(Handler::Block(k, args)) => (blk.handlers[*k], take(args)),
            Some(Handler::Outer(k, args)) => {
                let args = take(args);
                self.clear(b, ver);
                return Unwound::Outer(*k, args);
            }
        };
        self.clear(b, ver);
        self.enter_handler(h, args);
        Unwound::Block(h)
    }

    // Enters handler block h with `args` (owned) for its explicit params and
    // the pending exception (and lasti) for the implicit ones.
    unsafe fn enter_handler(&mut self, h: BlockId, args: Vec<*mut PyObject>) {
        let lasti = ffi::PyLong_FromLong(tr_frame_get_instr(self.f) as _);
        let exc = ffi::PyErr_GetRaisedException();
        let (mut lasti, mut exc) = (Some(lasti), Some(exc));
        self.enter(h, args, |kind| match kind {
            ParamKind::Lasti => lasti.take().expect("one lasti param"),
            ParamKind::Exc => exc.take().expect("one exc param"),
            ParamKind::Stack(_) => unreachable!("stack params are explicit"),
        });
        // A handler without a lasti param doesn't take it.
        if let Some(l) = lasti {
            ffi::Py_XDECREF(l);
        }
    }

    // Runs from block b: from the start of the version select() picks, or at
    // `resume` = (version, index).
    unsafe fn run_blocks(&mut self, mut b: BlockId, mut resume: Option<(usize, usize)>) -> Outcome {
        let cfg = self.cfg;
        'blocks: loop {
            let blk = &cfg[b];
            let (vi, first) = match resume.take() {
                Some(r) => r,
                None => match self.select(b) {
                    Some(vi) => (vi, 0),
                    None => match self.miss(b) {
                        Ok(next) => {
                            b = next;
                            continue 'blocks;
                        }
                        Err(outcome) => return outcome,
                    },
                },
            };
            let ver = &blk.versions[vi];
            let live = &self.p.live.versions[b.idx()][vi];
            for (i, v) in ver.body.iter().enumerate().skip(first) {
                let ins = &cfg[*v];
                if ins.op == Op::Yield {
                    let (value, stack) = ins.args.split_last().expect("Yield has the yielded value");
                    for a in stack {
                        tr_frame_push(self.f, xincref(self.val(*a)));
                    }
                    return Outcome::Yield {
                        value: xincref(self.val(*value)),
                        unit: ins.origin,
                    };
                }
                if ins.op.can_raise() {
                    tr_frame_set_instr(self.f, ins.origin as c_int);
                }
                let calls = matches!(
                    ins.op,
                    Op::Call { .. }
                        | Op::Bytecode {
                            opcode: opcodes::CALL_FUNCTION_EX,
                            ..
                        }
                );
                if let Err(how) = self.exec(*v, ins).and_then(|()| if calls { self.breaker() } else { Ok(()) }) {
                    match self.unwind(b, vi, i, how) {
                        Unwound::Block(h) => {
                            b = h;
                            continue 'blocks;
                        }
                        Unwound::Outer(k, args) => return Outcome::Outer(k, args),
                        Unwound::Propagate => return Outcome::Error,
                    }
                }
                self.drop_all(&live.after_kill[i]);
            }
            let outs: Vec<*mut PyObject> = ver.outputs.iter().map(|o| self.val(*o)).collect();
            match self.take_exit(b, &outs, Some(vi)) {
                Ok(next) => b = next,
                Err(outcome) => return outcome,
            }
        }
    }

    unsafe fn exec(&mut self, v: ValueId, ins: &Instr) -> R {
        let a = |k: usize| self.val(ins.args[k]);
        let obj = match &ins.op {
            Op::Const(i) => xincref(self.konst(*i)),
            // Written by the instr that produces it.
            Op::Project(_) => return Ok(()),
            Op::ToBool => {
                let r = ffi::PyObject_IsTrue(a(0));
                if r < 0 {
                    return Err(Exc::Error);
                }
                new_bool(r != 0)
            }
            Op::IsNone => new_bool(a(0) == ffi::Py_None()),
            Op::LoadLocal { slot } => xincref(*self.local(*slot)),
            Op::LoadLocalChecked { slot } => {
                let x = *self.local(*slot);
                if x.is_null() {
                    tr_unbound_local(self.f, *slot as c_int);
                    return Err(Exc::Error);
                }
                xincref(x)
            }
            Op::StoreLocal { slot } => {
                let p = self.local(*slot);
                let old = std::mem::replace(&mut *p, xincref(a(0)));
                ffi::Py_XDECREF(old);
                return Ok(());
            }
            Op::DeleteLocal { slot } => {
                let p = self.local(*slot);
                if (*p).is_null() {
                    tr_unbound_local(self.f, *slot as c_int);
                    return Err(Exc::Error);
                }
                let old = std::mem::replace(&mut *p, std::ptr::null_mut());
                ffi::Py_DECREF(old);
                return Ok(());
            }
            Op::LoadGlobal(n) => check(tr_load_global(self.f, *n as c_int))?,
            Op::LoadAttr(n) => check(ffi::PyObject_GetAttr(a(0), self.name(*n)))?,
            Op::LoadMethod(n) => {
                let mut out = [std::ptr::null_mut(); 2];
                status(tr_load_method(self.f, *n as c_int, a(0), out.as_mut_ptr()))?;
                self.distribute(v, &out);
                return Ok(());
            }
            Op::StoreAttr(n) => {
                if ffi::PyObject_SetAttr(a(1), self.name(*n), a(0)) < 0 {
                    return Err(Exc::Error);
                }
                return Ok(());
            }
            Op::Call { argc, kw_names } => {
                let kw = kw_names.map_or(std::ptr::null_mut(), |k| self.konst(k));
                check(tr_call(self.args(ins).as_ptr(), *argc as c_int, kw))?
            }
            Op::Binary(k) => check(tr_binary_op(*k as c_int, a(0), a(1)))?,
            Op::Compare(k) => check(ffi::PyObject_RichCompare(a(0), a(1), *k as c_int))?,
            Op::BuildTuple(n) => {
                let t = check(ffi::PyTuple_New(*n as ffi::Py_ssize_t))?;
                for (i, x) in ins.args.iter().enumerate() {
                    ffi::PyTuple_SetItem(t, i as ffi::Py_ssize_t, xincref(self.val(*x)));
                }
                t
            }
            Op::GetIter => check(ffi::PyObject_GetIter(a(0)))?,
            Op::MakeFunction(flags) => check(tr_make_function(self.f, *flags as c_int, self.args(ins).as_ptr()))?,
            Op::MakeCell { slot } => {
                let p = self.local(*slot);
                let cell = check(PyCell_New(*p))?;
                let old = std::mem::replace(&mut *p, cell);
                ffi::Py_XDECREF(old);
                return Ok(());
            }
            Op::CopyFreeVars(n) => {
                let closure = PyFunction_GetClosure(tr_frame_func(self.f));
                let offset = tr_frame_nlocalsplus(self.f) as usize - *n as usize;
                for i in 0..*n as usize {
                    let cell = ffi::PyTuple_GetItem(closure, i as ffi::Py_ssize_t);
                    *self.locals.add(offset + i) = xincref(cell);
                }
                return Ok(());
            }
            Op::LoadDeref { slot } => {
                let x = PyCell_Get(*self.local(*slot));
                if x.is_null() {
                    tr_unbound_deref(self.f, *slot as c_int);
                    return Err(Exc::Error);
                }
                x
            }
            Op::StoreDeref { slot } => {
                if PyCell_Set(*self.local(*slot), a(0)) < 0 {
                    return Err(Exc::Error);
                }
                return Ok(());
            }
            Op::DeleteDeref { slot } => {
                let cell = *self.local(*slot);
                let old = PyCell_Get(cell);
                if old.is_null() {
                    tr_unbound_deref(self.f, *slot as c_int);
                    return Err(Exc::Error);
                }
                ffi::Py_DECREF(old);
                if PyCell_Set(cell, std::ptr::null_mut()) < 0 {
                    return Err(Exc::Error);
                }
                return Ok(());
            }
            Op::LoadFromDictOrDeref { slot } => {
                check(tr_load_from_dict_or_deref(self.f, a(0), *slot as c_int))?
            }
            Op::Bytecode { opcode, arg } if super::trampoline::needed(*opcode, *arg) => {
                check(super::trampoline::call(*opcode, *arg, &self.args(ins)))?
            }
            Op::Bytecode { opcode, arg } => {
                let n = self.p.nouts[v.idx()] as usize;
                let mut out = vec![std::ptr::null_mut(); n.max(1)];
                status(tr_generic(
                    self.f,
                    *opcode as c_int,
                    *arg as c_int,
                    self.args(ins).as_ptr(),
                    out.as_mut_ptr(),
                ))?;
                match n {
                    0 => return Ok(()),
                    1 => out[0],
                    _ => {
                        self.distribute(v, &out);
                        return Ok(());
                    }
                }
            }
            Op::Yield => unreachable!("handled by run_blocks"),
            Op::Resume | Op::EvalBreaker => return self.breaker(),
            Op::IterNext => {
                let item = ffi::PyIter_Next(a(0));
                if item.is_null() && !ffi::PyErr_Occurred().is_null() {
                    return Err(Exc::Error);
                }
                self.distribute(v, &[item, new_bool(!item.is_null())]);
                return Ok(());
            }
            Op::Send => {
                let mut out = std::ptr::null_mut();
                let yielded = match tr_send(a(0), a(1), &mut out) {
                    0 => true,
                    1 => false,
                    _ => return Err(Exc::Error),
                };
                self.distribute(v, &[out, new_bool(yielded)]);
                return Ok(());
            }
            Op::Null | Op::Unknown | Op::Param(_) | Op::Removed => {
                unreachable!("not an instruction")
            }
        };
        self.set(v, obj);
        Ok(())
    }
}
