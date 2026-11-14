// Optimization of a Cfg's blocks, following Dynamo's structure: the frame's
// locals are read and written only in glue code, around pure blocks that take
// the values they need as params and produce their results as outputs, the way
// Dynamo's bytecode loads locals, calls a graph and stores its outputs.
//
// Optimizing a block splits it at its barriers (ops the caller didn't judge
// movable, the places Dynamo would graph-break) into glue blocks and pure
// blocks. A stretch of movable ops becomes a pure block; its first load of
// each local moves to the glue before it, later loads reuse the value last
// loaded or stored, and its stores move to the glue after it (dropped when
// nothing follows but a return). Inside the stretch, as under Dynamo, the
// frame's locals are stale and old values are released late. If a pure block
// raises while stores are pending, it unwinds to a landing block that writes
// them and re-raises into the original handler, so a handler or a traceback
// sees the locals as CPython would have left them.
//
// Each pure block is then outlined into itself: it keeps the original code as
// its origin (what runs when no version's guard holds, and what a later
// optimization can start again from) and gets the optimized code as its
// version, where Python-level operators are replaced by the aten ops they ran
// (`a + b` by aten::add.Tensor). Replacing is valid when the operator would
// dispatch straight to that op: the guard checks that each tensor param is a
// plain tensor (no __torch_function__ or operator overrides), each scalar
// param exactly its builtin type, and that no __torch_function__ mode is
// active, as Dynamo guards its graph inputs and torch-function state. An
// operand computed in the block counts only if it's a constant or the result
// of a replaced op (an aten op on plain tensors returns a plain tensor), since
// nothing checks values in the middle of a version. Dtypes, devices and
// shapes don't matter here: the same op runs whatever they are.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

use super::ir::{
    Block, BlockId, Builtin, Cfg, Check, Edge, Exit, Handler, Op, ParamKind, TermUnwind, Terminator, UnwindSegment,
    ValueId, Version,
};
use super::outline::{self, Outline};

// What the caller observed about an operator it may replace: the one aten op
// every visit ran, with the operator's args in order, and what each arg was.
pub struct AtenOp {
    pub name: String,
    pub args: Vec<Arg>,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Arg {
    PlainTensor,
    Scalar(Builtin),
}

// Optimizes `blocks` (each split, and its pure blocks outlined into
// themselves), then renumbers the blocks. Returns each old block's new id.
// `movable`: the instrs that may sit in a pure block; `aten`: the operators
// that may be replaced. With `strict`, a block that can't be optimized is an
// error; otherwise it's left as it is.
pub fn optimize_blocks(
    cfg: &mut Cfg,
    blocks: &[BlockId],
    movable: &HashSet<ValueId>,
    aten: &HashMap<ValueId, AtenOp>,
    strict: bool,
) -> Result<Vec<Option<BlockId>>, String> {
    let mut pure = vec![];
    for &b in blocks {
        if b.idx() >= cfg.blocks.len() {
            return Err(format!("no block bb{}", b.0));
        }
        if !outline::outlinable(cfg, b) {
            if strict {
                return Err(format!(
                    "bb{} can't be optimized: it already has an origin or several versions, or a generator suspends in it",
                    b.0
                ));
            }
            continue;
        }
        pure.extend(split(cfg, b, movable));
    }
    let map = cfg.compact_blocks();
    for p in pure {
        let p = map[p.idx()].expect("a pure block is reachable");
        outline::outline(cfg, &[p], Outline::Merge);
        use_aten(cfg, p, aten);
    }
    Ok(map)
}

// Replaces the operators of pure block p's version with aten ops where the
// guard can make that valid, and sets the guard.
fn use_aten(cfg: &mut Cfg, p: BlockId, aten: &HashMap<ValueId, AtenOp>) {
    let params = cfg[p].params.clone();
    let body = cfg[p].versions[0].body.clone();
    let mut checks: BTreeSet<Check> = BTreeSet::new();
    // Results of replaced ops: plain tensors whenever the guard holds.
    let mut tensors: HashSet<ValueId> = HashSet::new();
    for v in body {
        let Some(a) = aten.get(&v) else {
            continue;
        };
        if cfg[v].args.len() != a.args.len() {
            continue;
        }
        let mut need = vec![];
        let ok = cfg[v].args.iter().zip(&a.args).all(|(x, kind)| match params.iter().position(|q| q == x) {
            Some(i) => {
                need.push(match kind {
                    Arg::PlainTensor => Check::PlainTensor(i as u32),
                    Arg::Scalar(t) => Check::Exact(i as u32, *t),
                });
                true
            }
            None => match kind {
                Arg::PlainTensor => tensors.contains(x),
                Arg::Scalar(_) => matches!(cfg[*x].op, Op::Const(_)),
            },
        });
        if ok {
            checks.extend(need);
            cfg[v].op = Op::Aten(a.name.clone());
            tensors.insert(v);
        }
    }
    if !tensors.is_empty() {
        checks.insert(Check::NoTorchFunctionMode);
        cfg[p].versions[0].guard.checks = checks.into_iter().collect();
    }
}

enum Class {
    Load(u32),
    Store(u32),
    Compute,
    Barrier,
}

fn classify(cfg: &Cfg, v: ValueId, movable: &HashSet<ValueId>) -> Class {
    match cfg[v].op {
        Op::LoadLocal { slot } => Class::Load(slot),
        Op::StoreLocal { slot } => Class::Store(slot),
        // They read or write the frame in ways that must stay in place.
        Op::LoadLocalChecked { .. }
        | Op::DeleteLocal { .. }
        | Op::MakeCell { .. }
        | Op::CopyFreeVars(_)
        | Op::LoadDeref { .. }
        | Op::StoreDeref { .. }
        | Op::DeleteDeref { .. }
        | Op::LoadFromDictOrDeref { .. } => Class::Barrier,
        // Written by its producer when that runs, so it stays with it.
        Op::Project(_) => classify(cfg, cfg[v].args[0], movable),
        _ if movable.contains(&v) => Class::Compute,
        _ => Class::Barrier,
    }
}

// (slot, value): stores a pure block hasn't written to the frame yet.
type Pending = Vec<(u32, ValueId)>;

// A raising op of a pure block: where it unwinds in the original block, and
// the stores pending when it runs.
struct Raise {
    at: ValueId,
    handler: Handler,
    pending: Pending,
}

enum Part {
    Glue(Vec<ValueId>),
    Pure(Vec<ValueId>, Vec<Raise>),
}

impl Part {
    fn instrs(&self) -> &[ValueId] {
        match self {
            Part::Glue(is) | Part::Pure(is, _) => is,
        }
    }
}

fn handler_args(h: &Handler) -> Vec<ValueId> {
    match h {
        Handler::Block(_, args) | Handler::Outer(_, args) => args.clone(),
        Handler::Propagate => vec![],
    }
}

fn map_handler(h: &Handler, f: impl Fn(ValueId) -> ValueId) -> Handler {
    match h {
        Handler::Block(k, args) => Handler::Block(*k, args.iter().map(|a| f(*a)).collect()),
        Handler::Outer(k, args) => Handler::Outer(*k, args.iter().map(|a| f(*a)).collect()),
        Handler::Propagate => Handler::Propagate,
    }
}

// Splits block b (one version, no origin) into a chain of glue and pure
// blocks; b stays the first glue block. Returns the pure blocks (new ids, not
// yet renumbered), none if b has no stretch worth splitting out.
fn split(cfg: &mut Cfg, b: BlockId, movable: &HashSet<ValueId>) -> Vec<BlockId> {
    let ver = cfg[b].versions[0].clone();
    let returns = matches!(cfg[b].term, Terminator::Jump(Exit::Return(_)));
    let handler_at = |i: usize| -> Handler {
        ver.unwind.iter().rev().find(|s| s.from <= i).map_or(Handler::Propagate, |s| s.handler.clone())
    };

    // The parts, in the original block's namespace.
    let mut parts: Vec<Part> = vec![];
    let mut glue: Vec<ValueId> = vec![];
    let mut subst: HashMap<ValueId, ValueId> = HashMap::new();
    let get = |subst: &HashMap<ValueId, ValueId>, v: ValueId| subst.get(&v).copied().unwrap_or(v);
    let mut i = 0;
    while i < ver.body.len() {
        let start = i;
        while i < ver.body.len() && !matches!(classify(cfg, ver.body[i], movable), Class::Barrier) {
            i += 1;
        }
        let worth = ver.body[start..i].iter().any(|v| {
            matches!(classify(cfg, *v, movable), Class::Compute)
                && cfg[*v].op.can_raise()
                && !matches!(cfg[*v].op, Op::Resume | Op::EvalBreaker)
        });
        if worth {
            let mut known: HashMap<u32, ValueId> = HashMap::new();
            let mut pending: BTreeMap<u32, (ValueId, u32)> = BTreeMap::new();
            let (mut body, mut raises) = (vec![], vec![]);
            for k in start..i {
                let v = ver.body[k];
                match classify(cfg, v, movable) {
                    Class::Load(slot) => match known.get(&slot) {
                        Some(&x) => {
                            subst.insert(v, x);
                            cfg[v].op = Op::Removed;
                        }
                        None => {
                            known.insert(slot, v);
                            glue.push(v);
                        }
                    },
                    Class::Store(slot) => {
                        let x = get(&subst, cfg[v].args[0]);
                        known.insert(slot, x);
                        pending.insert(slot, (x, cfg[v].origin));
                        cfg[v].op = Op::Removed;
                        cfg[v].args.clear();
                    }
                    Class::Compute => {
                        if cfg[v].op.can_raise() {
                            let pending = pending.iter().map(|(s, (x, _))| (*s, *x)).collect();
                            raises.push(Raise { at: v, handler: handler_at(k), pending });
                        }
                        body.push(v);
                    }
                    Class::Barrier => unreachable!("runs hold no barriers"),
                }
            }
            parts.push(Part::Glue(std::mem::take(&mut glue)));
            parts.push(Part::Pure(body, raises));
            if !(returns && i == ver.body.len()) {
                for (slot, (x, origin)) in pending {
                    glue.push(cfg.add(Op::StoreLocal { slot }, vec![x], origin));
                }
            }
        } else {
            glue.extend(&ver.body[start..i]);
        }
        if i < ver.body.len() {
            glue.push(ver.body[i]);
            i += 1;
        }
    }
    if parts.is_empty() {
        return vec![];
    }
    parts.push(Part::Glue(glue));

    // Forwarded loads, everywhere they're used.
    for part in &parts {
        for v in part.instrs() {
            let args = cfg[*v].args.iter().map(|a| get(&subst, *a)).collect();
            cfg[*v].args = args;
        }
    }
    for part in &mut parts {
        if let Part::Pure(_, raises) = part {
            for r in raises {
                r.handler = map_handler(&r.handler, |a| get(&subst, a));
                r.pending.iter_mut().for_each(|(_, x)| *x = get(&subst, *x));
            }
        }
    }
    let outputs: Vec<ValueId> = ver.outputs.iter().map(|o| get(&subst, *o)).collect();
    let orig_index: HashMap<ValueId, usize> = ver.body.iter().enumerate().map(|(k, v)| (*v, k)).collect();
    // A glue instr's handler, in the original namespace with loads forwarded.
    let glue_handler = |v: ValueId| -> Option<Handler> {
        let oi = orig_index.get(&v)?;
        Some(map_handler(&handler_at(*oi), |a| get(&subst, a)))
    };

    // The values each part needs from earlier ones, which become its params.
    let n = parts.len();
    let mut live_in: Vec<Vec<ValueId>> = vec![vec![]; n];
    let mut live: HashSet<ValueId> = outputs.iter().copied().filter(|v| !v.is_reserved()).collect();
    for k in (0..n).rev() {
        let part = &parts[k];
        let mut uses: Vec<ValueId> = part.instrs().iter().flat_map(|v| cfg[*v].args.clone()).collect();
        match part {
            Part::Glue(is) => {
                for v in is.iter().filter(|v| cfg[**v].op.can_raise()) {
                    uses.extend(glue_handler(*v).map(|h| handler_args(&h)).unwrap_or_default());
                }
            }
            Part::Pure(_, raises) => {
                for r in raises {
                    uses.extend(handler_args(&r.handler));
                    uses.extend(r.pending.iter().map(|(_, x)| *x));
                }
            }
        }
        live.extend(uses.into_iter().filter(|v| !v.is_reserved()));
        for v in part.instrs() {
            live.remove(v);
        }
        let mut l: Vec<ValueId> = live.iter().copied().collect();
        l.sort();
        live_in[k] = l;
    }
    debug_assert!(live_in[0].iter().all(|v| cfg[b].params.contains(v)));

    // b is part 0; the other parts follow it, then the landing blocks.
    let orig_handlers = cfg[b].handlers.clone();
    let term = cfg[b].term.clone();
    let term_origin = cfg[b].term_origin;
    let base = cfg.blocks.len();
    let id_of = |k: usize| if k == 0 { b } else { BlockId((base + k - 1) as u32) };
    let mut new_blocks: Vec<Block> = vec![];
    let mut landings: Vec<((Handler, Pending), Block)> = vec![];
    let mut pure_ids = vec![];
    for (k, part) in parts.iter().enumerate() {
        let mut blk = Block::new();
        let mut local: HashMap<ValueId, ValueId> = HashMap::new();
        if k == 0 {
            blk.params = cfg[b].params.clone();
        } else {
            for (j, v) in live_in[k].iter().enumerate() {
                let p = cfg.add(Op::Param(ParamKind::Stack(j as u32)), vec![], term_origin);
                local.insert(*v, p);
                blk.params.push(p);
            }
        }
        let loc = |v: ValueId| local.get(&v).copied().unwrap_or(v);
        for v in part.instrs() {
            let args = cfg[*v].args.iter().map(|a| loc(*a)).collect();
            cfg[*v].args = args;
        }
        let mut ver = Version {
            body: part.instrs().to_vec(),
            ..Version::default()
        };
        let push_seg = |ver: &mut Version, from: usize, handler: Handler| {
            if ver.unwind.last().map_or(handler != Handler::Propagate, |s| s.handler != handler) {
                ver.unwind.push(UnwindSegment { from, handler });
            }
        };
        let index_of = |handlers: &mut Vec<BlockId>, h: BlockId| match handlers.iter().position(|x| *x == h) {
            Some(i) => i,
            None => {
                handlers.push(h);
                handlers.len() - 1
            }
        };
        let mut handlers: Vec<BlockId> = vec![];
        match part {
            Part::Glue(is) => {
                handlers.clone_from(&orig_handlers);
                for (pos, v) in is.iter().enumerate() {
                    if let (true, Some(h)) = (cfg[*v].op.can_raise(), glue_handler(*v)) {
                        push_seg(&mut ver, pos, map_handler(&h, loc));
                    }
                }
            }
            Part::Pure(body, raises) => {
                pure_ids.push(id_of(k));
                for r in raises {
                    let pos = body.iter().position(|v| *v == r.at).expect("the raise is in the body");
                    let handler = if r.pending.is_empty() {
                        match &r.handler {
                            Handler::Block(h, args) => Handler::Block(
                                index_of(&mut handlers, orig_handlers[*h]),
                                args.iter().map(|a| loc(*a)).collect(),
                            ),
                            h => h.clone(),
                        }
                    } else {
                        let key = (r.handler.clone(), r.pending.clone());
                        let li = match landings.iter().position(|(k, _)| *k == key) {
                            Some(li) => li,
                            None => {
                                let origin = cfg[r.at].origin;
                                landings.push((key, landing(cfg, &r.pending, &r.handler, &orig_handlers, origin)));
                                landings.len() - 1
                            }
                        };
                        let target = BlockId((base + n - 1 + li) as u32);
                        let args = r.pending.iter().map(|(_, x)| *x).chain(handler_args(&r.handler));
                        Handler::Block(index_of(&mut handlers, target), args.map(loc).collect())
                    };
                    push_seg(&mut ver, pos, handler);
                }
            }
        }
        blk.handlers = handlers;
        if k + 1 < n {
            ver.outputs = live_in[k + 1].iter().map(|v| loc(*v)).collect();
            blk.term = Terminator::Jump(Exit::Block(Edge {
                target: id_of(k + 1),
                args: (0..ver.outputs.len() as u32).collect(),
            }));
        } else {
            ver.outputs = outputs.iter().map(|v| loc(*v)).collect();
            blk.term = term.clone();
        }
        blk.term_origin = term_origin;
        blk.versions.push(ver);
        if k == 0 {
            cfg[b] = blk;
        } else {
            new_blocks.push(blk);
        }
    }
    cfg.blocks.extend(new_blocks);
    cfg.blocks.extend(landings.into_iter().map(|(_, l)| l));
    for k in std::iter::once(b.idx()).chain(base..cfg.blocks.len()) {
        cfg.normalize_unwind(BlockId(k as u32));
    }
    pure_ids
}

// A block that writes `pending` to the frame and re-raises into `handler` (an
// index into `handlers`, the original block's) with the same lasti, so the
// handler or a traceback sees what CPython would have. Its params: the pending
// values, the handler's args, then lasti and the exception.
fn landing(cfg: &mut Cfg, pending: &[(u32, ValueId)], handler: &Handler, handlers: &[BlockId], origin: u32) -> Block {
    let mut blk = Block::new();
    let mut param = |cfg: &mut Cfg, kind: ParamKind| {
        let p = cfg.add(Op::Param(kind), vec![], origin);
        blk.params.push(p);
        p
    };
    let n_args = handler_args(handler).len();
    let stored: Vec<(u32, ValueId)> =
        pending.iter().enumerate().map(|(j, (slot, _))| (*slot, param(cfg, ParamKind::Stack(j as u32)))).collect();
    let args: Vec<ValueId> = (0..n_args).map(|j| param(cfg, ParamKind::Stack((pending.len() + j) as u32))).collect();
    let lasti = param(cfg, ParamKind::Lasti);
    let exc = param(cfg, ParamKind::Exc);
    let mut ver = Version::default();
    for (slot, p) in stored {
        ver.body.push(cfg.add(Op::StoreLocal { slot }, vec![p], origin));
    }
    ver.outputs = [exc, lasti].into_iter().chain(args).collect();
    let unwind = match handler {
        Handler::Block(h, _) => {
            blk.handlers = vec![handlers[*h]];
            TermUnwind::Block(0, (2..2 + n_args as u32).collect())
        }
        Handler::Propagate => TermUnwind::Propagate,
        Handler::Outer(..) => unreachable!("a top-level block has no outer handlers"),
    };
    blk.term = Terminator::Reraise {
        exc: 0,
        lasti: Some(1),
        unwind,
    };
    blk.term_origin = origin;
    blk.versions.push(ver);
    blk
}
