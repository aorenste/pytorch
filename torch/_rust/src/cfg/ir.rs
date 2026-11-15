// SSA control-flow-graph IR for CPython 3.12 bytecode. See cfg/mod.rs for the
// pipeline. Every instruction is its own value: ValueId(n) is cfg.instrs[n].
// SSA values model only the bytecode value stack; fast locals (and cells) stay
// in the frame and are accessed with explicit Load/Store ops, so they keep
// CPython's lifetimes and stay visible to frame introspection.

use std::collections::HashMap;
use std::fmt::{self, Write as _};
use std::ops::{Index, IndexMut};
use std::sync::Arc;

use super::opcodes;

#[derive(Copy, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Debug)]
pub struct ValueId(pub u32);

#[derive(Copy, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Debug)]
pub struct BlockId(pub u32);

// Where code runs in a Cfg: an instr, or a block outside its instrs (entering
// and leaving it, its terminator).
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub enum Site {
    Instr(ValueId),
    Block(BlockId),
}

// Extra information to print with a Cfg (or with an origin in it, passed as
// `cfg`).
pub trait Annotations {
    // Lines to print under a site.
    fn notes(&self, _cfg: &Cfg, _site: Site) -> Vec<String> {
        vec![]
    }

    // Printed after the value (an instr result or a block param) it defines.
    fn value(&self, _cfg: &Cfg, _v: ValueId) -> Option<String> {
        None
    }
}

impl Annotations for () {}

// Invariant, not a field: blocks are numbered in RPO, so the entry is always 0.
pub const ENTRY: BlockId = BlockId(0);

// Reserved values. They belong to no block and dominate everything.
pub const NULL: ValueId = ValueId(0);
pub const UNKNOWN: ValueId = ValueId(1);
pub const NUM_RESERVED: u32 = 2;

impl ValueId {
    pub fn idx(self) -> usize {
        self.0 as usize
    }
    pub fn is_reserved(self) -> bool {
        self.0 < NUM_RESERVED
    }
}

impl BlockId {
    pub fn idx(self) -> usize {
        self.0 as usize
    }
}

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum ParamKind {
    Stack(u32),
    Exc,
    Lasti,
}

impl ParamKind {
    // Implicit params are supplied by the raise, so incoming edges and
    // handler segments carry no arg for them.
    pub fn is_implicit(self) -> bool {
        matches!(self, ParamKind::Exc | ParamKind::Lasti)
    }
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub enum Op {
    Null,
    Unknown,
    Param(ParamKind),
    Const(u32),
    Project(u32),
    ToBool,
    IsNone,
    // Frame slot access. LoadLocal doesn't check the slot is bound (CPython's
    // compiler guarantees it, or the value may be NULL for LOAD_FAST_AND_CLEAR).
    LoadLocal { slot: u32 },
    LoadLocalChecked { slot: u32 },
    StoreLocal { slot: u32 },
    DeleteLocal { slot: u32 },
    LoadGlobal(u32),
    // A global (Cfg::names[name]) read from the dicts Cfg::pins[globals]
    // and [builtins] instead of the frame's: code inlined from another
    // function. Uncached.
    LoadGlobalIn { name: u32, globals: u32, builtins: u32 },
    LoadAttr(u32),
    LoadMethod(u32),
    StoreAttr(u32),
    Call { argc: u32, kw_names: Option<u32> },
    Binary(u32),
    Compare(u32),
    BuildTuple(u32),
    GetIter,
    MakeFunction(u32),
    // Args are the whole value stack, bottom first; the last is the yielded
    // value.
    Yield,
    // RESUME: runs CPython's eval breaker (signals, pending calls, GC, GIL
    // drop requests), so it can raise. A resumed generator continues here.
    Resume,
    // FOR_ITER's next(): Project 0 is the item (NULL when exhausted), Project 1
    // whether there was one.
    IterNext,
    // SEND (receiver, value): Project 0 is the yielded or returned value,
    // Project 1 whether it was yielded.
    Send,
    // JUMP_BACKWARD's eval breaker check (see Resume).
    EvalBreaker,
    // Wraps the slot's current value in a new cell, in place.
    MakeCell { slot: u32 },
    // Copies the function's closure cells into the last n slots.
    CopyFreeVars(u32),
    LoadDeref { slot: u32 },
    StoreDeref { slot: u32 },
    DeleteDeref { slot: u32 },
    LoadFromDictOrDeref { slot: u32 },
    Bytecode { opcode: u8, arg: u32 },
    // Calls an aten operator overload ("aten::add.Tensor") on the args: in
    // schema order, the last kwnames.len() by those names, other args taking
    // their defaults. An optimized version's stand-in for the Python-level op
    // that ran it (see cfg/optimize.rs), or an op of a traced graph. An
    // overload returning several values (a tuple or Tensor[]) hands them to
    // its Projects.
    Aten { name: String, kwnames: Vec<String> },
    // Cfg::literals[i].
    Lit(u32),
    // The object Cfg::pins[i] resolved to when it was pinned (a version's
    // guard checks it still does).
    Pinned(u32),
    // Checks that its arg is True (true) or False, and that it is (by
    // identity) the object Cfg::pins[i]. A failing check aborts the version
    // (see Version::restart). In-body counterparts of a guard, for values
    // the version computes.
    Expect(bool),
    ExpectIs(u32),
    // Checks that its arg's class runs Cfg::pins[pin] for Cfg::names[name]
    // (looked up on the class, as for a special method like __call__).
    ExpectTypeAttr { name: u32, pin: u32 },
    // Checks that loading Cfg::names[name] from its arg falls back to its
    // class's __getattr__, which is Cfg::pins[pin]: the class has the
    // generic __getattribute__ and nothing by that name, and neither has the
    // instance's __dict__.
    ExpectGetattr { name: u32, pin: u32 },
    // A guard check on its arg instead of a param (the check's param index
    // is unused): a training version's checks on what its read prefix read.
    ExpectCheck(Box<Check>),
    // Runs Cfg::trainings[i] on the block's params (its args), handing each
    // block output to its Project: the forward of a pure block, with its own
    // autograd node (see exec/autograd.rs).
    Autograd(u32),
    Removed,
}

impl Op {
    pub fn can_raise(&self) -> bool {
        match self {
            Op::Null
            | Op::Unknown
            | Op::Param(_)
            | Op::Const(_)
            | Op::Lit(_)
            | Op::Pinned(_)
            | Op::Project(_)
            | Op::IsNone
            | Op::LoadLocal { .. }
            | Op::StoreLocal { .. }
            | Op::BuildTuple(_)
            | Op::MakeFunction(_)
            | Op::MakeCell { .. }
            | Op::CopyFreeVars(_)
            | Op::StoreDeref { .. }
            | Op::Removed => false,
            Op::ToBool
            | Op::LoadLocalChecked { .. }
            | Op::DeleteLocal { .. }
            | Op::LoadGlobal(_)
            | Op::LoadGlobalIn { .. }
            | Op::LoadAttr(_)
            | Op::LoadMethod(_)
            | Op::StoreAttr(_)
            | Op::Call { .. }
            | Op::Binary(_)
            | Op::Compare(_)
            | Op::GetIter
            | Op::Yield
            | Op::Resume
            | Op::IterNext
            | Op::Send
            | Op::EvalBreaker
            | Op::LoadDeref { .. }
            | Op::DeleteDeref { .. }
            | Op::LoadFromDictOrDeref { .. }
            | Op::Aten { .. }
            | Op::Expect(_)
            | Op::ExpectIs(_)
            | Op::ExpectCheck(_)
            | Op::ExpectTypeAttr { .. }
            | Op::ExpectGetattr { .. }
            | Op::Autograd(_) => true,
            Op::Bytecode { opcode, .. } => opcodes::generic_can_raise(*opcode),
        }
    }

    // Whether the instruction's own id is a usable value (used by the printer).
    pub fn produces_value(&self) -> bool {
        match self {
            Op::StoreAttr(_)
            | Op::StoreLocal { .. }
            | Op::DeleteLocal { .. }
            | Op::MakeCell { .. }
            | Op::CopyFreeVars(_)
            | Op::StoreDeref { .. }
            | Op::DeleteDeref { .. }
            | Op::Resume
            | Op::EvalBreaker
            | Op::Expect(_)
            | Op::ExpectIs(_)
            | Op::ExpectCheck(_)
            | Op::ExpectTypeAttr { .. }
            | Op::ExpectGetattr { .. }
            | Op::Removed => false,
            Op::Bytecode { opcode, arg } => opcodes::signature(*opcode, *arg, false)
                .is_some_and(|s| s.outputs.iter().any(|o| matches!(o, opcodes::Out::New))),
            _ => true,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Instr {
    pub op: Op,
    pub args: Vec<ValueId>,
    // Code unit of the bytecode instruction this was lowered from (the byte
    // offset / 2). Used for tracebacks, f_lineno and handler lasti; a pass that
    // creates an instr copies the origin of the instr it derives from.
    pub origin: u32,
}

// Index into a version's outputs. Every version of a block produces the same
// number of outputs, and the block's terminator reads them by index.
pub type OutIdx = u32;

// The instrs of a version, in order. Straight-line: control flow is only in
// the block's terminator.
pub type InstrSeq = Vec<ValueId>;

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Edge {
    pub target: BlockId,
    // One per explicit param of the target.
    pub args: Vec<OutIdx>,
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub enum Exit {
    Block(Edge),
    Return(OutIdx),
    // Only in an outlined Cfg: leaves it, the args becoming the owner block's
    // outputs (which its terminator then dispatches on).
    Leave(Vec<OutIdx>),
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub enum Handler {
    // Block::handlers[k], with one arg per explicit param of that block.
    Block(usize, Vec<ValueId>),
    // Only in an outlined Cfg: the owner block's handlers[k].
    Outer(usize, Vec<ValueId>),
    // The exception propagates out of the function.
    Propagate,
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct UnwindSegment {
    // Applies to raising ops at body[from..next.from].
    pub from: usize,
    pub handler: Handler,
}

// Conditions checked before any of a version runs, on the block's params (by
// position) or on global state.
#[derive(Clone, PartialEq, Eq, Ord, PartialOrd, Debug)]
pub enum Check {
    // A tensor whose Python operators run as plain aten ops: exactly
    // torch.Tensor or nn.Parameter (no __torch_function__ of its own).
    PlainTensor(u32),
    // Exactly this builtin type (not a subclass).
    Exact(u32, Builtin),
    // No __torch_function__ mode would intercept torch calls.
    NoTorchFunctionMode,
    // Grad mode is on (true) or off.
    GradEnabled(bool),
    // A tensor param requires grad (true) or doesn't.
    RequiresGrad(u32, bool),
    // A plain strided tensor with exactly this dtype, device, sizes and strides.
    TensorMeta(u32, Box<TensorSpec>),
    // Exactly this scalar: same type, and the same value (floats bitwise).
    Value(u32, Lit),
    // No forward-mode AD level is open.
    NoForwardAD,
    // No saved_tensors_hooks would pack tensors saved for backward.
    NoSavedTensorHooks,
    // Cfg::pins[i]'s path still resolves to the object it was pinned to.
    Pinned(u32),
    // A param is (by identity) the object Cfg::pins[i] was pinned to.
    Is(u32, u32),
    // A param is CPython's NULL (a LoadMethod's empty self slot).
    Null(u32),
    // A param is exactly a tuple (true) or list of exactly these ints.
    Ints(u32, bool, Vec<i64>),
}

impl Check {
    // The param it checks, if any.
    pub fn param(&self) -> Option<u32> {
        match self {
            Check::PlainTensor(p)
            | Check::Exact(p, _)
            | Check::RequiresGrad(p, _)
            | Check::TensorMeta(p, _)
            | Check::Value(p, _)
            | Check::Is(p, _)
            | Check::Null(p)
            | Check::Ints(p, _, _) => Some(*p),
            Check::NoTorchFunctionMode
            | Check::GradEnabled(_)
            | Check::NoForwardAD
            | Check::NoSavedTensorHooks
            | Check::Pinned(_) => None,
        }
    }
}

// c10's ScalarType and DeviceType enums.
#[derive(Clone, PartialEq, Eq, Ord, PartialOrd, Debug)]
pub struct TensorSpec {
    pub dtype: i8,
    pub device_type: i8,
    pub device_index: i8,
    pub sizes: Vec<i64>,
    pub strides: Vec<i64>,
}

// An object an optimized version holds (see cfg/optimize.rs), named for
// printing, and where it was read from if that can be checked again.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Pin {
    pub name: String,
    pub path: Option<PinPath>,
}

// A global (in the frame's globals, then builtins, or in the dicts
// Cfg::pins[ns]) and the attributes read from it in turn, each through a
// module's __dict__: `F.linear` is (F, [linear]). Names are indices into
// Cfg::names.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct PinPath {
    pub global: u32,
    pub attrs: Vec<u32>,
    pub ns: Option<(u32, u32)>,
}

// A Python value an op takes as is.
#[derive(Clone, PartialEq, Eq, Hash, Ord, PartialOrd, Debug)]
pub enum Lit {
    None,
    Bool(bool),
    Int(i64),
    // An f64's bits, so that Lit is Eq and a guard compares bitwise.
    Float(u64),
    Str(String),
    IntList(Vec<i64>),
    FloatList(Vec<u64>),
    BoolList(Vec<bool>),
    // By name: torch.<name> for dtypes, layouts and memory formats, and
    // torch.device(<name>).
    Dtype(String),
    Layout(String),
    MemoryFormat(String),
    Device(String),
}

impl fmt::Display for Lit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Lit::None => write!(f, "None"),
            Lit::Bool(b) => write!(f, "{}", if *b { "True" } else { "False" }),
            Lit::Int(i) => write!(f, "{i}"),
            Lit::Float(bits) => write!(f, "{:?}", f64::from_bits(*bits)),
            Lit::Str(x) => write!(f, "{x:?}"),
            Lit::IntList(xs) => write!(f, "{xs:?}"),
            Lit::FloatList(xs) => write!(f, "{:?}", xs.iter().map(|b| f64::from_bits(*b)).collect::<Vec<_>>()),
            Lit::BoolList(xs) => {
                let xs: Vec<&str> = xs.iter().map(|b| if *b { "True" } else { "False" }).collect();
                write!(f, "[{}]", xs.join(", "))
            }
            Lit::Dtype(x) | Lit::Layout(x) | Lit::MemoryFormat(x) => write!(f, "torch.{x}"),
            Lit::Device(x) => write!(f, "device({x:?})"),
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Ord, PartialOrd, Debug)]
pub enum Builtin {
    Int,
    Float,
    Bool,
}

#[derive(Clone, Default, PartialEq, Eq, Debug)]
pub struct Guard {
    // All must hold.
    pub checks: Vec<Check>,
}

// One way to run a block: when its guard holds, the body runs and produces the
// outputs the block's terminator reads. Values are local to the version.
#[derive(Clone, Debug, Default)]
pub struct Version {
    pub guard: Guard,
    pub body: InstrSeq,
    pub outputs: Vec<ValueId>,
    // Ordered by `from`.
    pub unwind: Vec<UnwindSegment>,
    // Whether anything going wrong (a raise, a failed Expect) abandons the
    // version and runs the block's origin from the start instead, which
    // gives CPython's exact exception, traceback and locals: for a version
    // with no effects a second run would repeat (see cfg/optimize.rs). Its
    // unwind segments are unused, and it runs no eval breaker after calls
    // (a signal handled there would be lost by the abort).
    pub restart: bool,
}

// Where a raising terminator's exception goes: like a segment's Handler, but
// with args read from the outputs.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum TermUnwind {
    Block(usize, Vec<OutIdx>),
    Propagate,
}

// Terminators only transfer control, to a successor or (Raise, Reraise) a
// handler; they have no other effects. They are shared by all versions of a
// block, so they give the CFG's shape: a block that raises does so whichever
// version runs.
#[derive(Clone, Debug)]
pub enum Terminator {
    Jump(Exit),
    Branch {
        cond: OutIdx,
        then: Exit,
        else_: Exit,
    },
    // RAISE_VARARGS n: args.len() == n.
    Raise {
        args: Vec<OutIdx>,
        unwind: TermUnwind,
    },
    // RERAISE; lasti for RERAISE 1/2.
    Reraise {
        exc: OutIdx,
        lasti: Option<OutIdx>,
        unwind: TermUnwind,
    },
    Unreachable,
}

impl Terminator {
    pub fn exits(&self) -> Vec<&Exit> {
        match self {
            Terminator::Jump(e) => vec![e],
            Terminator::Branch { then, else_, .. } => vec![then, else_],
            Terminator::Raise { .. } | Terminator::Reraise { .. } | Terminator::Unreachable => vec![],
        }
    }

    pub fn unwind(&self) -> Option<&TermUnwind> {
        match self {
            Terminator::Raise { unwind, .. } | Terminator::Reraise { unwind, .. } => Some(unwind),
            _ => None,
        }
    }

    pub fn unwind_mut(&mut self) -> Option<&mut TermUnwind> {
        match self {
            Terminator::Raise { unwind, .. } | Terminator::Reraise { unwind, .. } => Some(unwind),
            _ => None,
        }
    }

    pub fn edges(&self) -> Vec<&Edge> {
        self.exits()
            .into_iter()
            .filter_map(|e| match e {
                Exit::Block(edge) => Some(edge),
                Exit::Return(_) | Exit::Leave(_) => None,
            })
            .collect()
    }

    pub fn edges_mut(&mut self) -> Vec<&mut Edge> {
        let exits = match self {
            Terminator::Jump(e) => vec![e],
            Terminator::Branch { then, else_, .. } => vec![then, else_],
            Terminator::Raise { .. } | Terminator::Reraise { .. } | Terminator::Unreachable => vec![],
        };
        exits
            .into_iter()
            .filter_map(|e| match e {
                Exit::Block(edge) => Some(edge),
                Exit::Return(_) | Exit::Leave(_) => None,
            })
            .collect()
    }

    // Normal successors in field order.
    pub fn successors(&self) -> Vec<BlockId> {
        self.edges().iter().map(|e| e.target).collect()
    }

    // Every output index the terminator reads.
    pub fn outputs_mut(&mut self) -> Vec<&mut OutIdx> {
        let mut v: Vec<&mut OutIdx> = vec![];
        let (exits, unwind) = match self {
            Terminator::Jump(e) => (vec![e], None),
            Terminator::Branch { cond, then, else_ } => {
                v.push(cond);
                (vec![then, else_], None)
            }
            Terminator::Raise { args, unwind } => {
                v.extend(args.iter_mut());
                (vec![], Some(unwind))
            }
            Terminator::Reraise { exc, lasti, unwind } => {
                v.push(exc);
                v.extend(lasti.iter_mut());
                (vec![], Some(unwind))
            }
            Terminator::Unreachable => (vec![], None),
        };
        if let Some(TermUnwind::Block(_, args)) = unwind {
            v.extend(args.iter_mut());
        }
        for e in exits {
            match e {
                Exit::Block(Edge { args, .. }) | Exit::Leave(args) => v.extend(args.iter_mut()),
                Exit::Return(o) => v.push(o),
            }
        }
        v
    }

    pub fn outputs(&self) -> Vec<OutIdx> {
        self.clone().outputs_mut().into_iter().map(|o| *o).collect()
    }
}

#[derive(Clone, Debug)]
pub struct Block {
    pub params: Vec<ValueId>,
    // The first version whose guard holds runs.
    pub versions: Vec<Version>,
    pub term: Terminator,
    // Like Instr::origin, for the terminator (a finished frame's f_lineno).
    pub term_origin: u32,
    // Exception targets, shared by all versions and the origin
    // (Handler::Block and Handler::Outer index it).
    pub handlers: Vec<BlockId>,
    // The region this block was outlined from, run when no version's guard
    // holds. Its ENTRY params correspond one-to-one to this block's params.
    pub origin: Option<Arc<Cfg>>,
    pub removed: bool,
}

impl Block {
    pub fn new() -> Self {
        Block {
            params: vec![],
            versions: vec![],
            term: Terminator::Unreachable,
            term_origin: 0,
            handlers: vec![],
            origin: None,
            removed: false,
        }
    }

    // RPO-rule successor order: normal successors, then handlers.
    pub fn successors(&self) -> Vec<BlockId> {
        let mut s = self.term.successors();
        s.extend(&self.handlers);
        s
    }

    pub fn n_outputs(&self) -> usize {
        match (self.versions.first(), &self.origin) {
            (Some(v), _) => v.outputs.len(),
            (None, Some(o)) => o.boundary.as_ref().map_or(0, |b| b.n_outputs),
            (None, None) => 0,
        }
    }
}

// What an outlined Cfg's owner block expects of it (lets it verify on its own).
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Boundary {
    // The arity of every Leave.
    pub n_outputs: usize,
    // The number of args each Outer(k) passes.
    pub handler_arity: Vec<usize>,
}

#[derive(Clone, Debug)]
pub struct Cfg {
    pub instrs: Vec<Instr>,
    pub blocks: Vec<Block>,
    // Some for an outlined Cfg (see Block::origin).
    pub boundary: Option<Boundary>,
    pub slot_names: Vec<String>,
    pub const_reprs: Vec<String>,
    pub names: Vec<String>,
    // What Op::Lit instrs hold.
    pub literals: Vec<Lit>,
    // What Op::Pinned instrs hold and Check::Pinned checks.
    pub pins: Vec<Pin>,
    // Consts past the code's own (const_reprs): Op::Const(const_reprs.len()
    // + i) is the object Cfg::pins[const_pins[i]] (code inlined from another
    // function). Likewise names past the code's own co_names are only
    // strings here.
    pub const_pins: Vec<u32>,
    // What Op::Autograd instrs run.
    pub trainings: Vec<Arc<Training>>,
}

// A pure block's forward and backward, split from its joint graph (see
// cfg/partition.rs), for an Op::Autograd.
#[derive(Debug)]
pub struct Training {
    // Graphs of aten ops (see exec/graph.rs). The forward takes the block's
    // tensor params and leaves with its tensor outputs, then the values saved
    // for the backward; the backward takes those saved values, then the
    // gradients of the outputs given the node, and leaves with the gradients
    // of the params that require grad.
    pub forward: Arc<Cfg>,
    pub backward: Arc<Cfg>,
    // The block's params that are tensors, in the forward's order.
    pub tensor_params: Vec<u32>,
    // Which of those require grad.
    pub requires_grad: Vec<bool>,
    // By forward output: whether it's given the node (it requires grad).
    pub wrap: Vec<bool>,
    // How each block output is made.
    pub outputs: Vec<Template>,
}

#[derive(Clone, Debug)]
pub enum Template {
    Param(u32),
    // A forward output.
    Out(u32),
    Lit(Lit),
    Tuple(Vec<Template>),
    // CPython's NULL (an elided LoadMethod's empty self slot).
    Null,
}

impl Index<ValueId> for Cfg {
    type Output = Instr;
    fn index(&self, v: ValueId) -> &Instr {
        &self.instrs[v.idx()]
    }
}

impl IndexMut<ValueId> for Cfg {
    fn index_mut(&mut self, v: ValueId) -> &mut Instr {
        &mut self.instrs[v.idx()]
    }
}

impl Index<BlockId> for Cfg {
    type Output = Block;
    fn index(&self, b: BlockId) -> &Block {
        &self.blocks[b.idx()]
    }
}

impl IndexMut<BlockId> for Cfg {
    fn index_mut(&mut self, b: BlockId) -> &mut Block {
        &mut self.blocks[b.idx()]
    }
}

impl Cfg {
    pub fn new(
        slot_names: Vec<String>,
        const_reprs: Vec<String>,
        names: Vec<String>,
    ) -> Self {
        let reserved = [Op::Null, Op::Unknown];
        Cfg {
            instrs: reserved
                .into_iter()
                .map(|op| Instr {
                    op,
                    args: vec![],
                    origin: 0,
                })
                .collect(),
            blocks: vec![],
            boundary: None,
            slot_names,
            const_reprs,
            names,
            literals: vec![],
            pins: vec![],
            const_pins: vec![],
            trainings: vec![],
        }
    }

    pub fn add(&mut self, op: Op, args: Vec<ValueId>, origin: u32) -> ValueId {
        let v = ValueId(self.instrs.len() as u32);
        self.instrs.push(Instr { op, args, origin });
        v
    }

    pub fn block_ids(&self) -> impl Iterator<Item = BlockId> + '_ {
        (0..self.blocks.len())
            .map(|i| BlockId(i as u32))
            .filter(|b| !self[*b].removed)
    }

    pub fn param_kind(&self, v: ValueId) -> Option<ParamKind> {
        match self[v].op {
            Op::Param(k) => Some(k),
            _ => None,
        }
    }

    pub fn explicit_params(&self, b: BlockId) -> Vec<ValueId> {
        self[b]
            .params
            .iter()
            .copied()
            .filter(|p| !self.param_kind(*p).is_some_and(|k| k.is_implicit()))
            .collect()
    }

    pub fn remove_block(&mut self, b: BlockId) {
        let blk = std::mem::replace(&mut self[b], Block::new());
        for v in blk.versions.iter().flat_map(|v| &v.body).chain(&blk.params) {
            self[*v].op = Op::Removed;
            self[*v].args.clear();
        }
        self[b].removed = true;
    }

    // Canonical segment form, per version: no segment covering no raising op,
    // no adjacent identical segments, and the first segment never propagates.
    // Then drops the handlers no segment unwinds to.
    pub fn normalize_unwind(&mut self, b: BlockId) {
        let instrs = &self.instrs;
        let blk = &mut self.blocks[b.idx()];
        for ver in &mut blk.versions {
            loop {
                let n = ver.unwind.len();
                let covers_raise = |i: usize| {
                    let to = ver.unwind.get(i + 1).map_or(ver.body.len(), |s| s.from);
                    let from = ver.unwind[i].from.min(to);
                    ver.body[from..to].iter().any(|v| instrs[v.idx()].op.can_raise())
                };
                let drop = (0..n).find(|&i| {
                    !covers_raise(i)
                        || (i == 0 && ver.unwind[0].handler == Handler::Propagate)
                        || (i > 0 && ver.unwind[i].handler == ver.unwind[i - 1].handler)
                });
                match drop {
                    Some(i) => {
                        ver.unwind.remove(i);
                    }
                    None => break,
                }
            }
        }
        // The origin refers to the handlers by index too.
        if blk.origin.is_some() {
            return;
        }
        // Renumber the kept handlers in first-use order (the RPO rule's).
        let mut map = vec![None; blk.handlers.len()];
        let mut kept = vec![];
        let segs = blk.versions.iter().flat_map(|v| &v.unwind).filter_map(|s| match s.handler {
            Handler::Block(k, _) => Some(k),
            _ => None,
        });
        let term = match blk.term.unwind() {
            Some(TermUnwind::Block(k, _)) => Some(*k),
            _ => None,
        };
        for k in segs.chain(term) {
            if map[k].is_none() {
                map[k] = Some(kept.len());
                kept.push(blk.handlers[k]);
            }
        }
        for seg in blk.versions.iter_mut().flat_map(|v| &mut v.unwind) {
            if let Handler::Block(k, _) = &mut seg.handler {
                *k = map[*k].expect("referenced handler is kept");
            }
        }
        if let Some(TermUnwind::Block(k, _)) = blk.term.unwind_mut() {
            *k = map[*k].expect("referenced handler is kept");
        }
        blk.handlers = kept;
    }

    // Removes the outputs the terminator doesn't read.
    pub fn prune_outputs(&mut self, b: BlockId) {
        let blk = &mut self.blocks[b.idx()];
        let mut used = vec![false; blk.n_outputs()];
        for o in blk.term.outputs() {
            used[o as usize] = true;
        }
        if used.iter().all(|u| *u) {
            return;
        }
        let renumber: Vec<OutIdx> = used.iter().scan(0, |n, u| {
            *n += *u as OutIdx;
            Some(*n - 1)
        }).collect();
        for o in blk.term.outputs_mut() {
            *o = renumber[*o as usize];
        }
        for ver in &mut blk.versions {
            let mut k = 0;
            ver.outputs.retain(|_| {
                k += 1;
                used[k - 1]
            });
        }
    }

    // DFS from ENTRY visiting successors in reverse RPO-rule order, so the
    // first-listed successor gets the lower id.
    pub fn rpo(&self) -> Vec<BlockId> {
        rpo_by(self.blocks.len(), ENTRY.idx(), |b| {
            if self.blocks[b].removed {
                vec![]
            } else {
                self.blocks[b].successors().iter().map(|s| s.idx()).collect()
            }
        })
        .into_iter()
        .map(|i| BlockId(i as u32))
        .collect()
    }

    pub fn reachable(&self) -> Vec<bool> {
        let mut r = vec![false; self.blocks.len()];
        for b in self.rpo() {
            r[b.idx()] = true;
        }
        r
    }

    // Renumber the reachable blocks densely in RPO and drop the rest. Returns
    // each old block's new id (None if dropped).
    pub fn compact_blocks(&mut self) -> Vec<Option<BlockId>> {
        let order = self.rpo();
        let mut map = vec![None; self.blocks.len()];
        for (new, old) in order.iter().enumerate() {
            map[old.idx()] = Some(BlockId(new as u32));
        }
        let unreached: Vec<BlockId> = self
            .block_ids()
            .filter(|b| map[b.idx()].is_none())
            .collect();
        for b in unreached {
            self.remove_block(b);
        }
        let mut old_blocks: Vec<Option<Block>> =
            std::mem::take(&mut self.blocks).into_iter().map(Some).collect();
        for old in &order {
            let mut blk = old_blocks[old.idx()].take().expect("block visited twice");
            for e in blk.term.edges_mut() {
                e.target = map[e.target.idx()].expect("successor of a live block was dropped");
            }
            for h in &mut blk.handlers {
                *h = map[h.idx()].expect("handler was dropped");
            }
            self.blocks.push(blk);
        }
        map
    }

    pub fn render(&self, raw_ids: bool, origins: bool) -> String {
        self.render_with(raw_ids, origins, &())
    }

    pub fn render_with(&self, raw_ids: bool, origins: bool, ann: &dyn Annotations) -> String {
        let mut out = String::new();
        Printer::new(self, raw_ids, origins, ann).render(&mut out, "");
        out
    }
}

// Iterative DFS postorder, reversed. `succs(n)` lists successors in rule order.
pub fn rpo_by(n: usize, start: usize, succs: impl Fn(usize) -> Vec<usize>) -> Vec<usize> {
    let mut visited = vec![false; n];
    let mut post = Vec::with_capacity(n);
    let mut stack: Vec<(usize, Vec<usize>)> = vec![];
    visited[start] = true;
    let mut children = succs(start);
    children.reverse();
    stack.push((start, children));
    while let Some((node, children)) = stack.last_mut() {
        // Visit in reverse rule order: `children` is reversed, so pop from the front.
        if let Some(pos) = children.iter().position(|c| !visited[*c]) {
            let c = children[pos];
            children.drain(..=pos);
            visited[c] = true;
            let mut cc = succs(c);
            cc.reverse();
            stack.push((c, cc));
        } else {
            post.push(*node);
            stack.pop();
        }
    }
    post.reverse();
    post
}

const BINARY_OPS: [&str; 26] = [
    "+", "&", "//", "<<", "@", "*", "%", "|", "**", ">>", "-", "/", "^", "+=", "&=", "//=", "<<=",
    "@=", "*=", "%=", "|=", "**=", ">>=", "-=", "/=", "^=",
];
const COMPARE_OPS: [&str; 6] = ["<", "<=", "==", "!=", ">", ">="];

struct Printer<'a> {
    cfg: &'a Cfg,
    raw: bool,
    origins: bool,
    ann: &'a dyn Annotations,
    numbers: HashMap<ValueId, u32>,
}

impl<'a> Printer<'a> {
    fn new(cfg: &'a Cfg, raw: bool, origins: bool, ann: &'a dyn Annotations) -> Self {
        let mut numbers = HashMap::new();
        let mut next = 0;
        for b in cfg.block_ids() {
            for v in &cfg[b].params {
                numbers.insert(*v, next);
                next += 1;
            }
            for v in cfg[b].versions.iter().flat_map(|ver| &ver.body) {
                if cfg[*v].op.produces_value() {
                    numbers.insert(*v, next);
                    next += 1;
                }
            }
        }
        Printer {
            cfg,
            raw,
            origins,
            ann,
            numbers,
        }
    }

    fn val(&self, v: ValueId) -> String {
        if self.raw {
            return format!("%{}", v.0);
        }
        match v {
            NULL => return "null".into(),
            UNKNOWN => return "unknown".into(),
            _ => {}
        }
        match self.numbers.get(&v) {
            Some(n) => format!("%{n}"),
            None => format!("%?{}", v.0),
        }
    }

    fn vals(&self, vs: &[ValueId]) -> String {
        vs.iter().map(|v| self.val(*v)).collect::<Vec<_>>().join(", ")
    }

    fn param_name(&self, p: ValueId) -> String {
        match self.cfg.param_kind(p) {
            Some(ParamKind::Stack(i)) => format!("s{i}"),
            Some(ParamKind::Exc) => "exc".into(),
            Some(ParamKind::Lasti) => "lasti".into(),
            None => "?".into(),
        }
    }

    fn call(&self, target: BlockId, args: &[String]) -> String {
        if args.is_empty() {
            return format!("bb{}", target.0);
        }
        let params = if target.idx() < self.cfg.blocks.len() {
            self.cfg.explicit_params(target)
        } else {
            vec![]
        };
        let args: Vec<String> = args
            .iter()
            .enumerate()
            .map(|(i, a)| match params.get(i) {
                Some(p) => format!("{}={a}", self.param_name(*p)),
                None => a.clone(),
            })
            .collect();
        format!("bb{}({})", target.0, args.join(", "))
    }

    fn name(&self, i: u32) -> &str {
        self.cfg.names.get(i as usize).map_or("?", |s| s.as_str())
    }

    fn pin(&self, i: u32) -> &str {
        self.cfg.pins.get(i as usize).map_or("?", |p| p.name.as_str())
    }

    fn slot(&self, i: u32) -> &str {
        self.cfg.slot_names.get(i as usize).map_or("?", |s| s.as_str())
    }

    // " @offset" (bytes, as dis prints) for ops that can raise.
    fn at(&self, raises: bool, origin: u32) -> String {
        if self.origins && raises {
            format!(" @{}", origin * 2)
        } else {
            String::new()
        }
    }

    fn op(&self, op: &Op) -> String {
        match op {
            Op::Null => "Null".into(),
            Op::Unknown => "Unknown".into(),
            Op::Param(_) => "Param".into(),
            Op::Const(i) => format!(
                "Const {}",
                match self.cfg.const_reprs.get(*i as usize) {
                    Some(s) => s.as_str(),
                    None => self.cfg.const_pins.get(*i as usize - self.cfg.const_reprs.len()).map_or("?", |p| self.pin(*p)),
                }
            ),
            Op::Project(i) => format!("Project {i}"),
            Op::ToBool => "ToBool".into(),
            Op::IsNone => "IsNone".into(),
            Op::LoadLocal { slot } => format!("LoadLocal {}", self.slot(*slot)),
            Op::LoadLocalChecked { slot } => format!("LoadLocalChecked {}", self.slot(*slot)),
            Op::StoreLocal { slot } => format!("StoreLocal {}", self.slot(*slot)),
            Op::DeleteLocal { slot } => format!("DeleteLocal {}", self.slot(*slot)),
            Op::LoadGlobal(n) => format!("LoadGlobal {}", self.name(*n)),
            Op::LoadGlobalIn { name, globals, .. } => format!("LoadGlobalIn {} {}", self.pin(*globals), self.name(*name)),
            Op::LoadAttr(n) => format!("LoadAttr {}", self.name(*n)),
            Op::LoadMethod(n) => format!("LoadMethod {}", self.name(*n)),
            Op::StoreAttr(n) => format!("StoreAttr {}", self.name(*n)),
            Op::Call { argc, kw_names } => match kw_names {
                Some(k) => format!(
                    "Call {argc} kw={}",
                    self.cfg.const_reprs.get(*k as usize).map_or("?", |s| s.as_str())
                ),
                None => format!("Call {argc}"),
            },
            Op::Binary(k) => format!(
                "BinaryOp {}",
                BINARY_OPS.get(*k as usize).copied().unwrap_or("?")
            ),
            Op::Compare(k) => format!(
                "CompareOp {}",
                COMPARE_OPS.get(*k as usize).copied().unwrap_or("?")
            ),
            Op::BuildTuple(n) => format!("BuildTuple {n}"),
            Op::GetIter => "GetIter".into(),
            Op::MakeFunction(f) => format!("MakeFunction {f:#x}"),
            Op::Yield => "Yield".into(),
            Op::Resume => "Resume".into(),
            Op::IterNext => "IterNext".into(),
            Op::Send => "Send".into(),
            Op::EvalBreaker => "EvalBreaker".into(),
            Op::MakeCell { slot } => format!("MakeCell {}", self.slot(*slot)),
            Op::CopyFreeVars(n) => format!("CopyFreeVars {n}"),
            Op::LoadDeref { slot } => format!("LoadDeref {}", self.slot(*slot)),
            Op::StoreDeref { slot } => format!("StoreDeref {}", self.slot(*slot)),
            Op::DeleteDeref { slot } => format!("DeleteDeref {}", self.slot(*slot)),
            Op::LoadFromDictOrDeref { slot } => format!("LoadFromDictOrDeref {}", self.slot(*slot)),
            Op::Bytecode { opcode, arg } => format!("{} {arg}", opcodes::name(*opcode)),
            Op::Aten { name, kwnames } if kwnames.is_empty() => format!("Aten {name}"),
            Op::Aten { name, kwnames } => format!("Aten {name} kw[{}]", kwnames.join(", ")),
            Op::Lit(i) => format!("Lit {}", self.cfg.literals.get(*i as usize).map_or("?".into(), |l| l.to_string())),
            Op::Pinned(i) => format!("Pinned {}", self.pin(*i)),
            Op::Expect(b) => format!("Expect {}", if *b { "True" } else { "False" }),
            Op::ExpectIs(i) => format!("ExpectIs {}", self.pin(*i)),
            Op::ExpectCheck(c) => format!("ExpectCheck {}", self.check(&[], c)),
            Op::ExpectTypeAttr { name, pin } => format!("ExpectTypeAttr {} {}", self.name(*name), self.pin(*pin)),
            Op::ExpectGetattr { name, pin } => format!("ExpectGetattr {} {}", self.name(*name), self.pin(*pin)),
            Op::Autograd(i) => format!("Autograd t{i}"),
            Op::Removed => "Removed".into(),
        }
    }

    // `out` renders an output index: the value itself when the block has a
    // single unguarded version, else `oN`.
    fn exit(&self, e: &Exit, out: &dyn Fn(OutIdx) -> String) -> String {
        match e {
            Exit::Block(edge) => {
                let args: Vec<String> = edge.args.iter().map(|o| out(*o)).collect();
                self.call(edge.target, &args)
            }
            Exit::Return(o) => format!("return {}", out(*o)),
            Exit::Leave(args) => {
                let args: Vec<String> = args.iter().map(|o| out(*o)).collect();
                format!("leave({})", args.join(", "))
            }
        }
    }

    // " -> handler(args)" for a raising terminator, nothing if it propagates.
    fn term_unwind(&self, blk: &Block, u: &TermUnwind, out: &dyn Fn(OutIdx) -> String) -> String {
        let args = |a: &[OutIdx]| a.iter().map(|o| out(*o)).collect::<Vec<_>>();
        match u {
            TermUnwind::Block(k, a) => format!(" -> {}", self.call(blk.handlers[*k], &args(a))),
            TermUnwind::Propagate => String::new(),
        }
    }

    fn term(&self, blk: &Block, out: &dyn Fn(OutIdx) -> String) -> String {
        let t = &blk.term;
        let arm = |e: &Exit| match e {
            Exit::Block(_) => self.exit(e, out),
            Exit::Return(_) | Exit::Leave(_) => format!("({})", self.exit(e, out)),
        };
        match t {
            Terminator::Jump(e @ Exit::Block(_)) => format!("jump {}", self.exit(e, out)),
            Terminator::Jump(e) => self.exit(e, out),
            Terminator::Branch { cond, then, else_ } => format!("branch {}, {}, {}", out(*cond), arm(then), arm(else_)),
            Terminator::Raise { args, unwind } => {
                let args: Vec<String> = args.iter().map(|o| out(*o)).collect();
                let sep = if args.is_empty() { "" } else { " " };
                format!("raise{sep}{}{}", args.join(", "), self.term_unwind(blk, unwind, out))
            }
            Terminator::Reraise { exc, lasti, unwind } => {
                let lasti = lasti.map_or(String::new(), |l| format!(", lasti={}", out(l)));
                format!("reraise {}{lasti}{}", out(*exc), self.term_unwind(blk, unwind, out))
            }
            Terminator::Unreachable => "unreachable".into(),
        }
    }

    fn body(&self, out: &mut String, ver: &Version, indent: &str) {
        for v in &ver.body {
            let ins = &self.cfg[*v];
            let mut rhs = self.op(&ins.op);
            if !ins.args.is_empty() {
                rhs = format!("{rhs} {}", self.vals(&ins.args));
            }
            rhs += &self.at(ins.op.can_raise(), ins.origin);
            if ins.op.produces_value() {
                let _ = writeln!(out, "{indent}{}{} = {}", self.val(*v), self.ty(*v), rhs);
            } else {
                let _ = writeln!(out, "{indent}{rhs}");
            }
            self.notes(out, Site::Instr(*v), indent);
        }
    }

    // ": <annotation>" for a value that has one.
    fn ty(&self, v: ValueId) -> String {
        self.ann.value(self.cfg, v).map_or(String::new(), |t| format!(": {t}"))
    }

    fn notes(&self, out: &mut String, site: Site, indent: &str) {
        for line in self.ann.notes(self.cfg, site) {
            let _ = writeln!(out, "{indent}    {line}");
        }
    }

    // " [guard: ...]" for a non-empty guard.
    fn guard(&self, blk: &Block, g: &Guard) -> String {
        if g.checks.is_empty() {
            return String::new();
        }
        let checks: Vec<String> = g.checks.iter().map(|c| self.check(&blk.params, c)).collect();
        format!(" [guard: {}]", checks.join(", "))
    }

    // A check on `params` (by position; "_" past them: an ExpectCheck's arg).
    fn check(&self, params: &[ValueId], c: &Check) -> String {
        let param = |i: &u32| params.get(*i as usize).map_or("_".into(), |p| self.param_name(*p));
        match c {
            Check::PlainTensor(i) => format!("plain_tensor({})", param(i)),
            Check::Exact(i, t) => format!("{}({})", format!("{t:?}").to_lowercase(), param(i)),
            Check::NoTorchFunctionMode => "no_torch_function_mode".into(),
            Check::GradEnabled(on) => if *on { "grad_enabled" } else { "grad_disabled" }.into(),
            Check::RequiresGrad(i, on) => format!("{}({})", if *on { "requires_grad" } else { "no_requires_grad" }, param(i)),
            Check::TensorMeta(i, t) => {
                format!(
                    "tensor({} {} {} {:?} stride {:?})",
                    param(i),
                    crate::bindings::ffi::dtype_name(t.dtype),
                    crate::bindings::ffi::device_name(t.device_type, t.device_index),
                    t.sizes,
                    t.strides
                )
            }
            Check::Value(i, l) => format!("value({} == {l})", param(i)),
            Check::NoForwardAD => "no_forward_ad".into(),
            Check::NoSavedTensorHooks => "no_saved_tensor_hooks".into(),
            Check::Pinned(i) => format!("pinned({})", self.pin(*i)),
            Check::Is(p, i) => format!("is({}, {})", param(p), self.pin(*i)),
            Check::Null(p) => format!("null({})", param(p)),
            Check::Ints(p, tuple, ns) => {
                let ns: Vec<String> = ns.iter().map(|n| n.to_string()).collect();
                let (l, r) = if *tuple { ("(", if ns.len() == 1 { ",)" } else { ")" }) } else { ("[", "]") };
                format!("value({} == {l}{}{r})", param(p), ns.join(", "))
            }
        }
    }

    fn unwind(&self, out: &mut String, blk: &Block, ver: &Version, indent: &str) {
        for (i, seg) in ver.unwind.iter().enumerate() {
            let to = ver.unwind.get(i + 1).map_or(ver.body.len(), |s| s.from);
            let dest = match &seg.handler {
                Handler::Block(k, args) => {
                    let args: Vec<String> = args.iter().map(|a| self.val(*a)).collect();
                    self.call(blk.handlers[*k], &args)
                }
                Handler::Outer(k, args) => format!("outer {k}({})", self.vals(args)),
                Handler::Propagate => "raise".into(),
            };
            let _ = writeln!(out, "{indent}unwind [{}..{}) -> {}", seg.from, to, dest);
        }
    }

    fn render(&self, out: &mut String, indent: &str) {
        let deeper = format!("{indent}    ");
        for b in self.cfg.block_ids() {
            let blk = &self.cfg[b];
            if blk.params.is_empty() {
                let _ = writeln!(out, "{indent}bb{}:", b.0);
            } else {
                let ps: Vec<String> = blk
                    .params
                    .iter()
                    .map(|p| format!("{} {}{}", self.val(*p), self.param_name(*p), self.ty(*p)))
                    .collect();
                let _ = writeln!(out, "{indent}bb{}({}):", b.0, ps.join(", "));
            }
            let inner = format!("{indent}  ");
            match blk.versions.as_slice() {
                // A single unguarded version prints inline, with its outputs
                // shown as the values they are.
                [ver] if ver.guard.checks.is_empty() && !ver.restart => {
                    self.body(out, ver, &inner);
                    let vals = |o: OutIdx| self.val(ver.outputs[o as usize]);
                    let _ = writeln!(out, "{inner}{}", self.term(blk, &vals));
                    self.notes(out, Site::Block(b), &inner);
                    self.unwind(out, blk, ver, &inner);
                }
                versions => {
                    for ver in versions {
                        let restart = if ver.restart { " (restarts)" } else { "" };
                        let _ = writeln!(out, "{inner}version{}{restart}:", self.guard(blk, &ver.guard));
                        self.body(out, ver, &deeper);
                        let _ = writeln!(out, "{deeper}outputs [{}]", self.vals(&ver.outputs));
                        self.unwind(out, blk, ver, &deeper);
                    }
                    let _ = writeln!(out, "{inner}{}", self.term(blk, &|o| format!("o{o}")));
                    self.notes(out, Site::Block(b), &inner);
                }
            }
            if let Some(origin) = &blk.origin {
                let _ = writeln!(out, "{inner}origin:");
                Printer::new(origin, self.raw, self.origins, self.ann).render(out, &deeper);
            }
        }
        for (i, t) in self.cfg.trainings.iter().enumerate() {
            let _ = writeln!(out, "{indent}t{i} forward:");
            Printer::new(&t.forward, self.raw, self.origins, &()).render(out, &deeper);
            let _ = writeln!(out, "{indent}t{i} backward:");
            Printer::new(&t.backward, self.raw, self.origins, &()).render(out, &deeper);
        }
    }
}
