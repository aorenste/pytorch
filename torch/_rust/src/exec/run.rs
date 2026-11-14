// Interprets one activation of a Compiled CFG on a CPython frame. Locals live
// in the frame (localsplus); SSA values are owned references in `vals`.

use std::os::raw::c_int;

use pyo3::ffi::{self, PyObject};

use crate::cfg::ir::*;

use super::effects::{self, Recorder};
use super::types;
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
    // Null when not recording (see RECORD).
    rec: *mut Recorder,
    // The frame's prev_instr, the code units it points into, and the
    // interpreter's eval-breaker flag.
    prev_instr: *mut *mut u16,
    code_units: *mut u16,
    breaker_flag: *const std::sync::atomic::AtomicI32,
    // Scratch, reused to avoid allocating per instr or block: an instr's
    // args, a version's outputs, an edge's args.
    args: Vec<*mut PyObject>,
    outs: Vec<*mut PyObject>,
    edge: Vec<*mut PyObject>,
}

// An activation's buffers, pooled per thread so a call allocates nothing.
// Pooled `vals` are all null: an activation drops every value it holds before
// it ends (run_blocks drops what's left of the version it leaves by).
#[derive(Default)]
struct Bufs {
    vals: Vec<*mut PyObject>,
    args: Vec<*mut PyObject>,
    outs: Vec<*mut PyObject>,
    edge: Vec<*mut PyObject>,
}

thread_local! {
    static POOL: std::cell::RefCell<Vec<Bufs>> = const { std::cell::RefCell::new(Vec::new()) };
}

impl Bufs {
    // With at least `n` null values.
    fn take(n: usize) -> Bufs {
        let mut b = POOL.with_borrow_mut(|p| p.pop()).unwrap_or_default();
        if b.vals.len() < n {
            b.vals.resize(n, std::ptr::null_mut());
        }
        b
    }

    fn give(self) {
        debug_assert!(self.vals.iter().all(|v| v.is_null()), "an activation ended holding values");
        POOL.with_borrow_mut(|p| p.push(self));
    }
}

// A version's body, laid out for run_blocks.
pub struct Code {
    pub steps: Vec<Step>,
    // The values each step drops after it runs (Step::kill ranges).
    pub kills: Vec<ValueId>,
    // What's left when the version runs to its end: the values no step drops
    // (those its outputs or only its unwind segments use).
    pub exit_kill: Vec<ValueId>,
    // For each exit of the terminator (Jump: one, Branch: then and else) that
    // goes to a block, how the version's values pass to its params.
    pub edges: Vec<Option<Transfer>>,
}

pub struct Transfer {
    pub target: BlockId,
    // (source value, target param, whether the source moves rather than
    // being shared: its last use, when the version holds it).
    pub moves: Vec<(ValueId, ValueId, bool)>,
    // exit_kill minus the moved values.
    pub rest: Vec<ValueId>,
}

pub struct Step {
    pub v: ValueId,
    pub op: Fast,
    pub flags: u8,
    pub kill: (u32, u32),
}

// The ops run_blocks runs inline when nothing is recorded: they can't raise
// or enter a frame.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Fast {
    Const(u32),
    LoadLocal(u32),
    // (slot, value)
    StoreLocal(u32, ValueId),
    No,
}

impl Code {
    // By block, then version. `flags` by instr (FLAG_*).
    pub fn build(cfg: &Cfg, live: &super::live::Liveness, flags: &[u8]) -> Vec<Vec<Code>> {
        // Only params and instrs with FLAG_VALUE ever hold an object.
        let holds = |v: &ValueId| flags[v.idx()] & FLAG_VALUE != 0 || matches!(cfg[*v].op, Op::Param(_));
        let version = |blk: &Block, ver: &Version, lv: &super::live::VersionLive| {
            let mut kills = vec![];
            let steps = ver
                .body
                .iter()
                .zip(&lv.after_kill)
                .map(|(v, after)| {
                    let start = kills.len() as u32;
                    kills.extend(after.iter().filter(|k| holds(k)));
                    let op = match cfg[*v].op {
                        Op::Const(k) => Fast::Const(k),
                        Op::LoadLocal { slot } => Fast::LoadLocal(slot),
                        Op::StoreLocal { slot } => Fast::StoreLocal(slot, cfg[*v].args[0]),
                        _ => Fast::No,
                    };
                    Step { v: *v, op, flags: flags[v.idx()], kill: (start, kills.len() as u32) }
                })
                .collect();
            let dropped: std::collections::HashSet<&ValueId> = lv.entry_kill.iter().chain(&kills).collect();
            let mut exit_kill: Vec<ValueId> =
                blk.params.iter().chain(&ver.body).filter(|v| holds(v) && !dropped.contains(v)).copied().collect();
            exit_kill.sort();
            let exits: Vec<&Exit> = match &blk.term {
                Terminator::Jump(e) => vec![e],
                Terminator::Branch { then, else_, .. } => vec![then, else_],
                _ => vec![],
            };
            let edges = exits
                .into_iter()
                .map(|e| {
                    let Exit::Block(edge) = e else { return None };
                    let srcs: Vec<ValueId> = edge.args.iter().map(|o| ver.outputs[*o as usize]).collect();
                    let owned = |v: &ValueId| exit_kill.contains(v);
                    let moves: Vec<(ValueId, ValueId, bool)> = srcs
                        .iter()
                        .zip(&cfg[edge.target].params)
                        .enumerate()
                        .map(|(i, (src, dst))| (*src, *dst, owned(src) && !srcs[i + 1..].contains(src)))
                        .collect();
                    let rest = exit_kill.iter().filter(|v| !moves.iter().any(|(s, _, m)| *m && s == *v)).copied().collect();
                    Some(Transfer { target: edge.target, moves, rest })
                })
                .collect();
            Code { steps, kills, exit_kill, edges }
        };
        cfg.blocks
            .iter()
            .zip(&live.versions)
            .map(|(blk, lvs)| blk.versions.iter().zip(lvs).map(|(ver, lv)| version(blk, ver, lv)).collect())
            .collect()
    }
}

// How run_blocks continues after a step.
enum Next {
    Step,
    // A raise went to this handler (entered).
    Block(BlockId),
    Done(Outcome),
}

pub const FLAG_RAISE: u8 = 1;
// Its own id holds a value (not a multi-output instr's), observed when it's
// defined.
pub const FLAG_VALUE: u8 = 2;
// A call whose callable is args[0] (or args[1] when args[0] is NULL).
pub const FLAG_CALL: u8 = 4;
// CALL_FUNCTION_EX, whose callable is args[1].
pub const FLAG_CALL_EX: u8 = 8;
pub const FLAG_YIELD: u8 = 16;

// Runs `f` with a recorder for `table` at `site`, or with none if recording is off.
fn recording<R>(table: &std::sync::Mutex<effects::Table>, site: Site, f: impl FnOnce(*mut Recorder) -> R) -> R {
    if super::RECORD.load(std::sync::atomic::Ordering::Relaxed) {
        effects::scope(table, site, f)
    } else {
        f(std::ptr::null_mut())
    }
}

// Runs the frame, which the caller has linked.
pub unsafe fn run(c: &Compiled, prep: &Prepared, f: *mut Frame, start: Start) -> Outcome {
    let site = match start {
        Start::Entry => Site::Block(ENTRY),
        Start::Resume { b, .. } => Site::Block(b),
    };
    recording(&prep.effects, site, |rec| {
        let Bufs { vals, args, outs, edge } = Bufs::take(prep.cfg.instrs.len());
        let mut act = Act {
            p: prep,
            cfg: &prep.cfg,
            f,
            locals: tr_frame_localsplus(f),
            consts: c.consts.as_ptr(),
            names: c.names.as_ptr(),
            vals,
            tstate: ffi::PyThreadState_Get(),
            rec,
            prev_instr: tr_frame_prev_instr_slot(f),
            code_units: tr_frame_code_units(f),
            breaker_flag: std::sync::atomic::AtomicI32::from_ptr(tr_eval_breaker_flag(ffi::PyThreadState_Get())),
            args,
            outs,
            edge,
        };
        let result = match start {
            Start::Entry => act.run_blocks(ENTRY, None),
            Start::Resume { b, ver, i, throw } => match act.resume(b, ver, i, throw) {
                Some(Ok(h)) => act.run_blocks(h, None),
                Some(Err(next)) => act.run_blocks(b, Some((ver, next))),
                None => Outcome::Error,
            },
        };
        act.into_bufs().give();
        result
    })
}

impl Act<'_> {
    // Points the frame at code unit `unit`, as if that instruction is executing.
    #[inline]
    unsafe fn set_instr(&self, unit: c_int) {
        *self.prev_instr = self.code_units.add(unit as usize);
    }

    fn into_bufs(self) -> Bufs {
        Bufs { vals: self.vals, args: self.args, outs: self.outs, edge: self.edge }
    }

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
        self.observe(v);
        let code = &self.p.code[b.idx()][ver];
        let (k0, k1) = code.steps[i].kill;
        self.drop_all(&code.kills[k0 as usize..k1 as usize]);
        Some(Err(i + 1))
    }

    fn val(&self, v: ValueId) -> *mut PyObject {
        self.vals[v.idx()]
    }

    unsafe fn set(&mut self, v: ValueId, obj: *mut PyObject) {
        let old = std::mem::replace(&mut self.vals[v.idx()], obj);
        ffi::Py_XDECREF(old);
    }

    // Joins v's current object into its type, if recording.
    #[inline]
    unsafe fn observe(&self, v: ValueId) {
        if !self.rec.is_null() {
            self.observe_slow(v);
        }
    }

    #[inline(never)]
    unsafe fn observe_slow(&self, v: ValueId) {
        let ty = types::observe(self.val(v));
        let mut table = self.p.types.lock().unwrap();
        let slot = &mut table[v.idx()];
        *slot = Some(match slot.take() {
            Some(t) => t.join(ty),
            None => ty,
        });
    }

    // The args' objects, in self.args (valid until the next call).
    fn args(&mut self, ins: &Instr) -> *const *mut PyObject {
        self.args.clear();
        self.args.extend(ins.args.iter().map(|a| self.vals[a.idx()]));
        self.args.as_ptr()
    }

    unsafe fn local(&self, slot: u32) -> *mut *mut PyObject {
        self.locals.add(slot as usize)
    }

    unsafe fn name(&self, i: u32) -> *mut PyObject {
        ffi::PyTuple_GET_ITEM(self.names, i as ffi::Py_ssize_t)
    }

    unsafe fn konst(&self, i: u32) -> *mut PyObject {
        ffi::PyTuple_GET_ITEM(self.consts, i as ffi::Py_ssize_t)
    }

    // Moves the outputs of a multi-output instr into its Project instrs.
    unsafe fn distribute(&mut self, v: ValueId, outs: &[*mut PyObject]) {
        let p = self.p;
        for (k, obj) in outs.iter().enumerate() {
            match p.projects[v.idx()].get(k).copied().flatten() {
                Some(proj) => self.set(proj, *obj),
                // An output nobody reads (simplify never drops Projects, but be safe).
                None => ffi::Py_XDECREF(*obj),
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

    // Leaves a version that ran to its end by a normal edge.
    unsafe fn transfer(&mut self, t: &Transfer) {
        let mut args = std::mem::take(&mut self.edge);
        for (src, _, moved) in &t.moves {
            let p = &mut self.vals[src.idx()];
            args.push(if *moved { std::mem::replace(p, std::ptr::null_mut()) } else { xincref(*p) });
        }
        self.drop_all(&t.rest);
        for ((_, dst, _), obj) in t.moves.iter().zip(args.drain(..)) {
            self.set(*dst, obj);
            self.observe(*dst);
        }
        self.edge = args;
    }

    // Drops what is left of a version's values when it ran to the end.
    unsafe fn clear_at_exit(&mut self, b: BlockId, ver: usize) {
        let p = self.p;
        self.drop_all(&p.code[b.idx()][ver].exit_kill);
    }

    // Enters `target` (after clear()), with owned values for its explicit
    // params in order and `implicit` supplying the others.
    unsafe fn enter(
        &mut self,
        target: BlockId,
        explicit: impl IntoIterator<Item = *mut PyObject>,
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
            self.observe(*p);
        }
    }

    // The version of `b` to run: the first whose guard holds. None: run its
    // origin.
    unsafe fn select(&mut self, b: BlockId) -> Option<usize> {
        let vers = &self.cfg[b].versions;
        let ver = if vers.len() == 1 && vers[0].guard.checks.is_empty() {
            0
        } else {
            vers.iter().position(|v| v.guard.checks.iter().all(|c| self.check(b, c)))?
        };
        self.drop_all(&self.p.live.versions[b.idx()][ver].entry_kill);
        Some(ver)
    }

    unsafe fn check(&self, b: BlockId, c: &Check) -> bool {
        let param = |i: &u32| self.val(self.cfg[b].params[*i as usize]);
        match c {
            Check::PlainTensor(i) => super::is_plain_tensor(param(i)),
            Check::Exact(i, t) => {
                let tp = ffi::Py_TYPE(param(i));
                tp == match t {
                    Builtin::Int => &raw mut ffi::PyLong_Type,
                    Builtin::Float => &raw mut ffi::PyFloat_Type,
                    Builtin::Bool => &raw mut ffi::PyBool_Type,
                }
            }
            Check::NoTorchFunctionMode => !crate::bindings::ffi::torch_function_mode_active(),
        }
    }

    // No version of `b` matched: run its origin on b's params, then continue
    // as b would with the outputs it leaves with.
    unsafe fn miss(&mut self, b: BlockId) -> Result<BlockId, Outcome> {
        let origin = self.p.origins[b.idx()].as_ref().expect("verified: a block without a matching version has an origin");
        let params: Vec<*mut PyObject> =
            self.cfg[b].params.iter().map(|p| std::mem::replace(&mut self.vals[p.idx()], std::ptr::null_mut())).collect();
        let result = recording(&origin.effects, Site::Block(ENTRY), |rec| {
            let Bufs { vals, args, outs, edge } = Bufs::take(origin.cfg.instrs.len());
            let mut sub = Act {
                p: origin,
                cfg: &origin.cfg,
                vals,
                rec,
                args,
                outs,
                edge,
                locals: self.locals,
                consts: self.consts,
                names: self.names,
                f: self.f,
                tstate: self.tstate,
                prev_instr: self.prev_instr,
                code_units: self.code_units,
                breaker_flag: self.breaker_flag,
            };
            for (obj, q) in params.into_iter().zip(&origin.cfg[ENTRY].params) {
                sub.set(*q, obj);
            }
            let result = sub.run_blocks(ENTRY, None);
            sub.into_bufs().give();
            result
        });
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
                self.set_instr(blk.term_origin as c_int);
                let exc = args.first().map_or(std::ptr::null_mut(), out);
                let cause = args.get(1).map_or(std::ptr::null_mut(), out);
                Some(status(tr_do_raise(exc, cause)).unwrap_err())
            }
            Terminator::Reraise { exc, lasti, .. } => {
                let instr = lasti.map_or(blk.term_origin as c_int, |l| ffi::PyLong_AsLong(out(&l)) as c_int);
                self.set_instr(instr);
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
                if let Some(vi) = clear {
                    self.clear_at_exit(b, vi);
                }
                return Err(Outcome::Error);
            };
            let (h, args) = (blk.handlers[*k], take(args));
            if let Some(vi) = clear {
                self.clear_at_exit(b, vi);
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
                let mut args = std::mem::take(&mut self.edge);
                args.extend(edge.args.iter().map(|o| xincref(outs[*o as usize])));
                if let Some(vi) = clear {
                    self.clear_at_exit(b, vi);
                }
                self.enter(edge.target, args.drain(..), |_| unreachable!("edges have no implicit params"));
                self.edge = args;
                Ok(edge.target)
            }
            Exit::Return(o) => {
                // A finished frame's f_lineno is its return's line.
                self.set_instr(blk.term_origin as c_int);
                let r = xincref(outs[*o as usize]);
                if let Some(vi) = clear {
                    self.clear_at_exit(b, vi);
                }
                Err(Outcome::Return(r))
            }
            Exit::Leave(os) => {
                let r = take(os);
                if let Some(vi) = clear {
                    self.clear_at_exit(b, vi);
                }
                Err(Outcome::Leave(r))
            }
        }
    }

    // Where CPython checks its eval breaker (RESUME, after calls,
    // JUMP_BACKWARD), run it; it can raise (e.g. KeyboardInterrupt).
    #[inline]
    unsafe fn breaker(&self) -> R {
        if (*self.breaker_flag).load(std::sync::atomic::Ordering::Relaxed) != 0 && !super::trampoline::eval_breaker() {
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
            None | Some(Handler::Propagate) => {
                self.clear(b, ver);
                return Unwound::Propagate;
            }
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
            if let Some(rec) = self.rec.as_mut() {
                rec.at(Site::Block(b));
            }
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
            let code = &self.p.code[b.idx()][vi];
            let fast = self.rec.is_null();
            let mut i = first;
            while i < code.steps.len() {
                // Steps' kill ranges are within `kills`, and instrs index
                // values within `vals`.
                let step = code.steps.get_unchecked(i);
                let kills = code.kills.get_unchecked(step.kill.0 as usize..step.kill.1 as usize);
                i += 1;
                if fast && step.op != Fast::No {
                    // A value is null until its instr defines it.
                    let vals = self.vals.as_mut_ptr();
                    match step.op {
                        Fast::Const(k) => {
                            debug_assert!(self.vals[step.v.idx()].is_null());
                            *vals.add(step.v.idx()) = xincref(self.konst(k));
                        }
                        Fast::LoadLocal(slot) => {
                            debug_assert!(self.vals[step.v.idx()].is_null());
                            *vals.add(step.v.idx()) = xincref(*self.local(slot));
                        }
                        Fast::StoreLocal(slot, x) => {
                            let p = self.local(slot);
                            let old = std::mem::replace(&mut *p, xincref(*vals.add(x.idx())));
                            ffi::Py_XDECREF(old);
                        }
                        Fast::No => unreachable!(),
                    }
                    for k in kills {
                        let p = vals.add(k.idx());
                        let old = std::mem::replace(&mut *p, std::ptr::null_mut());
                        ffi::Py_XDECREF(old);
                    }
                    continue;
                }
                match self.slow_step(b, vi, i - 1, step, kills) {
                    Next::Step => {}
                    Next::Block(h) => {
                        b = h;
                        continue 'blocks;
                    }
                    Next::Done(outcome) => return outcome,
                }
            }
            let edge = match &blk.term {
                Terminator::Jump(_) => code.edges[0].as_ref(),
                Terminator::Branch { cond, .. } => {
                    let taken = self.vals[ver.outputs[*cond as usize].idx()] == ffi::Py_True();
                    code.edges[if taken { 0 } else { 1 }].as_ref()
                }
                _ => None,
            };
            if let Some(t) = edge {
                self.transfer(t);
                b = t.target;
                continue;
            }
            let mut outs = std::mem::take(&mut self.outs);
            outs.extend(ver.outputs.iter().map(|o| self.vals[o.idx()]));
            let next = self.take_exit(b, &outs, Some(vi));
            outs.clear();
            self.outs = outs;
            match next {
                Ok(next) => b = next,
                Err(outcome) => return outcome,
            }
        }
    }

    // Runs body[i] of version `vi` of block b (step) and drops `kills`.
    #[inline(always)]
    unsafe fn slow_step(&mut self, b: BlockId, vi: usize, i: usize, step: &Step, kills: &[ValueId]) -> Next {
        let v = &step.v;
        let ins = &self.cfg[*v];
        let flags = step.flags;
        if flags & FLAG_YIELD != 0 {
            let (value, stack) = ins.args.split_last().expect("Yield has the yielded value");
            for a in stack {
                tr_frame_push(self.f, xincref(self.val(*a)));
            }
            let value = xincref(self.val(*value));
            self.clear(b, vi);
            return Next::Done(Outcome::Yield { value, unit: ins.origin });
        }
        if flags & FLAG_RAISE != 0 {
            self.set_instr(ins.origin as c_int);
        }
        // The callable a call's frame (if any) would be entered for directly.
        let callee = if flags & FLAG_CALL != 0 {
            let c = self.val(ins.args[0]);
            Some(if c.is_null() { self.val(ins.args[1]) } else { c })
        } else if flags & FLAG_CALL_EX != 0 {
            Some(self.val(ins.args[1]))
        } else {
            None
        };
        let rec = self.rec;
        if !rec.is_null() {
            if let Some(c) = callee.filter(|c| !c.is_null()) {
                effects::record_callee(&self.p.callees, *v, c);
            }
            (*rec).at(Site::Instr(*v));
            (*rec).set_callee(callee.unwrap_or(std::ptr::null_mut()));
        }
        let done = self.exec(*v, ins);
        if !rec.is_null() {
            (*rec).set_callee(std::ptr::null_mut());
            (*rec).at(Site::Block(b));
        }
        if let Err(how) = done.and_then(|()| if callee.is_some() { self.breaker() } else { Ok(()) }) {
            return match self.unwind(b, vi, i, how) {
                Unwound::Block(h) => Next::Block(h),
                Unwound::Outer(k, args) => Next::Done(Outcome::Outer(k, args)),
                Unwound::Propagate => Next::Done(Outcome::Error),
            };
        }
        if flags & FLAG_VALUE != 0 {
            self.observe(*v);
        }
        self.drop_all(kills);
        Next::Step
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
            Op::LoadGlobal(n) => {
                let cache = self.p.globals[self.p.global_slot[v.idx()] as usize].get();
                check(tr_load_global(self.f, *n as c_int, cache))?
            }
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
                check(tr_call(self.args(ins), *argc as c_int, kw))?
            }
            Op::Binary(k) => check(tr_binary_op(*k as c_int, a(0), a(1)))?,
            Op::Aten(_) => {
                let f = self.p.aten[&v].as_ptr();
                let args = self.args(ins);
                check(ffi::PyObject_Vectorcall(f, args, ins.args.len(), std::ptr::null_mut()))?
            }
            Op::Compare(k) => check(ffi::PyObject_RichCompare(a(0), a(1), *k as c_int))?,
            Op::BuildTuple(n) => {
                let t = check(ffi::PyTuple_New(*n as ffi::Py_ssize_t))?;
                for (i, x) in ins.args.iter().enumerate() {
                    ffi::PyTuple_SetItem(t, i as ffi::Py_ssize_t, xincref(self.val(*x)));
                }
                t
            }
            Op::GetIter => check(ffi::PyObject_GetIter(a(0)))?,
            Op::MakeFunction(flags) => check(tr_make_function(self.f, *flags as c_int, self.args(ins)))?,
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
                let args = self.args(ins);
                check(super::trampoline::call(*opcode, *arg, std::slice::from_raw_parts(args, ins.args.len())))?
            }
            Op::Bytecode { opcode, arg } => {
                let n = self.p.nouts[v.idx()] as usize;
                let mut small = [std::ptr::null_mut(); 4];
                let mut big = vec![];
                let out: &mut [*mut PyObject] = if n <= small.len() {
                    &mut small[..n.max(1)]
                } else {
                    big.resize(n, std::ptr::null_mut());
                    &mut big
                };
                let args = self.args(ins);
                status(tr_generic(self.f, *opcode as c_int, *arg as c_int, args, out.as_mut_ptr()))?;
                match n {
                    0 => return Ok(()),
                    1 => out[0],
                    _ => {
                        self.distribute(v, out);
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
