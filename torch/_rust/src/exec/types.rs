// A type lattice over the values seen at each SSA value (instr results and
// block params) while executing a Cfg. Each observed object is abstracted to a
// Ty and joined into the value's Ty so far. Observing never runs Python code:
// only exact builtin types are looked into (a subclass may override anything),
// and tensors are read through tensor_meta.
//
// Observing costs a bounded amount per value: containers are recorded by kind
// and length only (small tuples per position, one level deep). What a
// container holds shows up where it is used, as the types of the values read
// out of it (subscripts, iteration, unpacking), the way Dynamo only learns
// about the parts of an input that the code touches.

use std::fmt;

use pyo3::ffi::{self, PyObject};

use crate::bindings::ffi::{device_name, dtype_name, tensor_by_type, tensor_meta, TensorMeta};

// More alternatives than this in a union widen it to Any.
const MAX_UNION: usize = 4;
// Strings longer than this are tracked as str without their value.
const MAX_STR: usize = 64;
// Longer tuples, and tuples inside a tuple, are recorded by length only.
const MAX_TUPLE: usize = 16;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Interval {
    pub lo: i64,
    pub hi: i64,
}

impl Interval {
    fn exact(n: i64) -> Self {
        Interval { lo: n, hi: n }
    }

    fn join(self, o: Interval) -> Self {
        Interval {
            lo: self.lo.min(o.lo),
            hi: self.hi.max(o.hi),
        }
    }
}

impl fmt::Display for Interval {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match (self.lo, self.hi) {
            (lo, hi) if lo == hi => write!(f, "{lo}"),
            (i64::MIN, i64::MAX) => write!(f, "?"),
            (lo, hi) => write!(f, "{lo}..{hi}"),
        }
    }
}

// A value that is either always the same or not.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum Konst<T> {
    Is(T),
    Varies,
}

impl<T: PartialEq> Konst<T> {
    fn join(self, o: Konst<T>) -> Self {
        match (self, o) {
            (Konst::Is(a), Konst::Is(b)) if a == b => Konst::Is(a),
            _ => Konst::Varies,
        }
    }
}

// An exact Python class, by identity. Not a reference: a class freed and
// another allocated at the same address would be conflated.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Class {
    ptr: usize,
    name: String,
}

impl Class {
    unsafe fn of(obj: *mut PyObject) -> Self {
        let tp = ffi::Py_TYPE(obj);
        let name = std::ffi::CStr::from_ptr((*tp).tp_name).to_string_lossy().into_owned();
        Class { ptr: tp as usize, name }
    }

    pub fn is(&self, tp: *const ffi::PyTypeObject) -> bool {
        self.ptr == tp as usize
    }
}

#[derive(Clone, PartialEq, Debug)]
pub enum Shape {
    Dims(Vec<Interval>),
    // Not known when observed (Python-dispatch or symbolic sizes) or rank varies.
    Unknown,
}

#[derive(Clone, PartialEq, Debug)]
pub struct TensorTy {
    class: Class,
    // c10::ScalarType; Varies also when the tensor was undefined.
    dtype: Konst<i8>,
    // (c10::DeviceType, index).
    device: Konst<(i8, i8)>,
    requires_grad: Konst<bool>,
    shape: Shape,
}

impl TensorTy {
    pub fn class(&self) -> &Class {
        &self.class
    }

    fn join(self, o: TensorTy) -> Self {
        let shape = match (self.shape, o.shape) {
            (Shape::Dims(a), Shape::Dims(b)) if a.len() == b.len() => {
                Shape::Dims(a.into_iter().zip(b).map(|(x, y)| x.join(y)).collect())
            }
            _ => Shape::Unknown,
        };
        TensorTy {
            class: self.class,
            dtype: self.dtype.join(o.dtype),
            device: self.device.join(o.device),
            requires_grad: self.requires_grad.join(o.requires_grad),
            shape,
        }
    }
}

#[derive(Clone, PartialEq, Debug)]
pub enum Kind {
    // CPython's NULL (e.g. the self slot of a non-method LoadMethod).
    Null,
    None,
    Bool(Konst<bool>),
    // Beyond i64 joins to the full range.
    Int(Interval),
    Float,
    Str(Konst<String>),
    Tensor(TensorTy),
    // Per position while every tuple seen had the same length.
    Tuple(Vec<Ty>),
    // Lengths only.
    VarTuple(Interval),
    List(Interval),
    Set(Interval),
    FrozenSet(Interval),
    Dict(Interval),
    // Anything else, by exact class.
    Object(Class),
}

impl Kind {
    // Whether `self` and `o` join into one alternative of a union.
    fn same(&self, o: &Kind) -> bool {
        match (self, o) {
            (Kind::Tensor(a), Kind::Tensor(b)) => a.class == b.class,
            (Kind::Object(a), Kind::Object(b)) => a == b,
            (Kind::Tuple(_) | Kind::VarTuple(_), Kind::Tuple(_) | Kind::VarTuple(_)) => true,
            _ => std::mem::discriminant(self) == std::mem::discriminant(o),
        }
    }

    // Only for kinds that are the same().
    fn join(self, o: Kind) -> Kind {
        match (self, o) {
            (Kind::Bool(a), Kind::Bool(b)) => Kind::Bool(a.join(b)),
            (Kind::Int(a), Kind::Int(b)) => Kind::Int(a.join(b)),
            (Kind::Str(a), Kind::Str(b)) => Kind::Str(a.join(b)),
            (Kind::Tensor(a), Kind::Tensor(b)) => Kind::Tensor(a.join(b)),
            (Kind::Tuple(a), Kind::Tuple(b)) if a.len() == b.len() => {
                Kind::Tuple(a.into_iter().zip(b).map(|(x, y)| x.join(y)).collect())
            }
            (a @ (Kind::Tuple(_) | Kind::VarTuple(_)), b) => Kind::VarTuple(a.tuple_len().join(b.tuple_len())),
            (Kind::List(a), Kind::List(b)) => Kind::List(a.join(b)),
            (Kind::Set(a), Kind::Set(b)) => Kind::Set(a.join(b)),
            (Kind::FrozenSet(a), Kind::FrozenSet(b)) => Kind::FrozenSet(a.join(b)),
            (Kind::Dict(a), Kind::Dict(b)) => Kind::Dict(a.join(b)),
            (a, _) => a,
        }
    }

    fn tuple_len(&self) -> Interval {
        match self {
            Kind::Tuple(elems) => Interval::exact(elems.len() as i64),
            Kind::VarTuple(len) => *len,
            _ => unreachable!("only tuples"),
        }
    }
}

#[derive(Clone, PartialEq, Debug)]
pub enum Ty {
    // One to MAX_UNION alternatives, no two the same().
    Union(Vec<Kind>),
    Any,
}

impl Ty {
    pub fn join(self, o: Ty) -> Ty {
        let (Ty::Union(mut alts), Ty::Union(more)) = (self, o) else {
            return Ty::Any;
        };
        for k in more {
            match alts.iter().position(|a| a.same(&k)) {
                Some(i) => {
                    let a = std::mem::replace(&mut alts[i], Kind::Null);
                    alts[i] = a.join(k);
                }
                None => alts.push(k),
            }
        }
        if alts.len() > MAX_UNION {
            Ty::Any
        } else {
            Ty::Union(alts)
        }
    }
}

// The Ty of one object (borrowed, possibly NULL). Holds the GIL.
pub unsafe fn observe(obj: *mut PyObject) -> Ty {
    Ty::Union(vec![kind(obj, false)])
}

unsafe fn kind(obj: *mut PyObject, in_tuple: bool) -> Kind {
    if obj.is_null() {
        return Kind::Null;
    }
    if obj == ffi::Py_None() {
        return Kind::None;
    }
    if ffi::PyBool_Check(obj) != 0 {
        return Kind::Bool(Konst::Is(obj == ffi::Py_True()));
    }
    if ffi::PyLong_CheckExact(obj) != 0 {
        let mut overflow = 0;
        let n = ffi::PyLong_AsLongLongAndOverflow(obj, &mut overflow);
        return Kind::Int(match overflow {
            0 => Interval::exact(n),
            _ => Interval {
                lo: i64::MIN,
                hi: i64::MAX,
            },
        });
    }
    if ffi::PyFloat_CheckExact(obj) != 0 {
        return Kind::Float;
    }
    if ffi::PyUnicode_CheckExact(obj) != 0 {
        return Kind::Str(str_value(obj));
    }
    if let Some(t) = tensor_by_type(obj as usize).as_ref() {
        return Kind::Tensor(tensor_ty(obj, t));
    }
    if ffi::PyTuple_CheckExact(obj) != 0 {
        let n = ffi::PyTuple_GET_SIZE(obj);
        if in_tuple || n as usize > MAX_TUPLE {
            return Kind::VarTuple(Interval::exact(n as i64));
        }
        return Kind::Tuple((0..n).map(|i| Ty::Union(vec![kind(ffi::PyTuple_GET_ITEM(obj, i), true)])).collect());
    }
    if ffi::PyList_CheckExact(obj) != 0 {
        return Kind::List(Interval::exact(ffi::PyList_GET_SIZE(obj) as i64));
    }
    if ffi::PySet_CheckExact(obj) != 0 {
        return Kind::Set(Interval::exact(ffi::PySet_Size(obj) as i64));
    }
    if ffi::PyFrozenSet_CheckExact(obj) != 0 {
        return Kind::FrozenSet(Interval::exact(ffi::PySet_Size(obj) as i64));
    }
    if ffi::PyDict_CheckExact(obj) != 0 {
        return Kind::Dict(Interval::exact(ffi::PyDict_Size(obj) as i64));
    }
    Kind::Object(Class::of(obj))
}

// Read by code point: PyUnicode_AsUTF8AndSize would cache a UTF-8 copy in a
// non-ASCII str, which sys.getsizeof can see.
unsafe fn str_value(obj: *mut PyObject) -> Konst<String> {
    let n = ffi::PyUnicode_GetLength(obj);
    if n as usize > MAX_STR {
        return Konst::Varies;
    }
    let mut out = String::new();
    for i in 0..n {
        // Lone surrogates aren't chars.
        match char::from_u32(ffi::PyUnicode_ReadChar(obj, i)) {
            Some(c) => out.push(c),
            None => return Konst::Varies,
        }
    }
    Konst::Is(out)
}

unsafe fn tensor_ty(obj: *mut PyObject, t: &crate::bindings::ffi::Tensor) -> TensorTy {
    let mut m = TensorMeta {
        defined: false,
        dtype: 0,
        requires_grad: false,
        has_device: false,
        device_type: 0,
        device_index: 0,
        has_shape: false,
        sizes: vec![],
    };
    tensor_meta(t, &mut m);
    let class = Class::of(obj);
    if !m.defined {
        return TensorTy {
            class,
            dtype: Konst::Varies,
            device: Konst::Varies,
            requires_grad: Konst::Varies,
            shape: Shape::Unknown,
        };
    }
    TensorTy {
        class,
        dtype: Konst::Is(m.dtype),
        device: if m.has_device {
            Konst::Is((m.device_type, m.device_index))
        } else {
            Konst::Varies
        },
        requires_grad: Konst::Is(m.requires_grad),
        shape: if m.has_shape {
            Shape::Dims(m.sizes.iter().map(|s| Interval::exact(*s)).collect())
        } else {
            Shape::Unknown
        },
    }
}

impl fmt::Display for Ty {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Ty::Any => write!(f, "Any"),
            Ty::Union(alts) => {
                for (i, k) in alts.iter().enumerate() {
                    if i > 0 {
                        write!(f, " | ")?;
                    }
                    write!(f, "{k}")?;
                }
                Ok(())
            }
        }
    }
}

impl fmt::Display for Kind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Kind::Null => write!(f, "null"),
            Kind::None => write!(f, "None"),
            Kind::Bool(Konst::Is(b)) => write!(f, "{}", if *b { "True" } else { "False" }),
            Kind::Bool(Konst::Varies) => write!(f, "bool"),
            Kind::Int(i) if i.lo == i.hi => write!(f, "int {i}"),
            Kind::Int(i) => write!(f, "int[{i}]"),
            Kind::Float => write!(f, "float"),
            Kind::Str(Konst::Is(s)) => write!(f, "str {s:?}"),
            Kind::Str(Konst::Varies) => write!(f, "str"),
            Kind::Tensor(t) => write!(f, "{t}"),
            Kind::Tuple(elems) => {
                let elems: Vec<String> = elems.iter().map(|e| e.to_string()).collect();
                write!(f, "tuple({})", elems.join(", "))
            }
            Kind::VarTuple(len) => write!(f, "tuple[{len}]"),
            Kind::List(len) => write!(f, "list[{len}]"),
            Kind::Set(len) => write!(f, "set[{len}]"),
            Kind::FrozenSet(len) => write!(f, "frozenset[{len}]"),
            Kind::Dict(len) => write!(f, "dict[{len}]"),
            Kind::Object(c) => write!(f, "{}", c.name),
        }
    }
}

impl fmt::Display for TensorTy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut parts = vec![];
        parts.push(match self.dtype {
            Konst::Is(d) => dtype_name(d).to_string(),
            Konst::Varies => "?".into(),
        });
        parts.push(match self.device {
            Konst::Is((t, i)) => device_name(t, i).to_string(),
            Konst::Varies => "?".into(),
        });
        parts.push(match &self.shape {
            Shape::Dims(d) => format!("[{}]", d.iter().map(|i| i.to_string()).collect::<Vec<_>>().join(", ")),
            Shape::Unknown => "[...]".into(),
        });
        match self.requires_grad {
            Konst::Is(true) => parts.push("grad".into()),
            Konst::Is(false) => {}
            Konst::Varies => parts.push("grad?".into()),
        }
        write!(f, "{}({})", self.class.name, parts.join(" "))
    }
}
