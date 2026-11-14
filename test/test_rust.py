# Owner(s): ["module: devx"]

import _thread
import contextlib
import dis
import importlib
import inspect
import io
import re
import signal
import sys
import unittest

import torch
from torch.testing._internal.common_utils import (
    HardwareClassification,
    run_tests,
    TestCase,
)


@unittest.skipIf(torch._rust is None, "torch._rust extension is not built")
class TestRustBindings(TestCase):
    hw_classification = HardwareClassification.GENERIC

    def test_function(self):
        self.assertEqual(torch._rust.test_function(), "hello from rust")

    def test_tensor_size(self):
        t = torch.zeros(2, 3, 4)
        self.assertEqual(torch._rust.tensor_size(t), [2, 3, 4])

    def test_tensor_size_scalar(self):
        t = torch.tensor(0.0)
        self.assertEqual(torch._rust.tensor_size(t), [])

    def test_tensor_size_1d(self):
        t = torch.zeros(5)
        self.assertEqual(torch._rust.tensor_size(t), [5])

    def test_collect_tensors_single(self):
        t = torch.zeros(3)
        result = torch._rust.collect_tensors(t)
        self.assertEqual(len(result), 1)
        self.assertIs(result[0], t)
        self.assertIsInstance(result[0], torch.Tensor)

    def test_collect_tensors_non_tensor(self):
        self.assertEqual(torch._rust.collect_tensors(42), [])
        self.assertEqual(torch._rust.collect_tensors("hello"), [])
        self.assertEqual(torch._rust.collect_tensors(None), [])
        self.assertEqual(torch._rust.collect_tensors(b"bytes"), [])

    def test_collect_tensors_list(self):
        a, b = torch.zeros(2), torch.ones(3)
        result = torch._rust.collect_tensors([a, 1, b, "x"])
        self.assertEqual([id(t) for t in result], [id(a), id(b)])

    def test_collect_tensors_tuple(self):
        a, b = torch.zeros(2), torch.ones(3)
        result = torch._rust.collect_tensors((a, b))
        self.assertEqual([id(t) for t in result], [id(a), id(b)])

    def test_collect_tensors_dict_collects_values(self):
        a, b = torch.zeros(2), torch.ones(3)
        result = torch._rust.collect_tensors({"a": a, "b": b})
        self.assertEqual([id(t) for t in result], [id(a), id(b)])

    def test_collect_tensors_dict_keys_ignored(self):
        a, b = torch.zeros(2), torch.ones(3)
        result = torch._rust.collect_tensors({a: b})
        self.assertEqual([id(t) for t in result], [id(b)])

    def test_collect_tensors_set(self):
        a, b = torch.zeros(2), torch.ones(3)
        result = torch._rust.collect_tensors({a, b})
        self.assertEqual({id(t) for t in result}, {id(a), id(b)})

    def test_collect_tensors_frozenset(self):
        a, b = torch.zeros(2), torch.ones(3)
        result = torch._rust.collect_tensors(frozenset([a, b]))
        self.assertEqual({id(t) for t in result}, {id(a), id(b)})

    def test_collect_tensors_nested(self):
        a, b, c = torch.zeros(1), torch.ones(2), torch.zeros(3)
        result = torch._rust.collect_tensors([a, (b,), {"k": [c]}])
        self.assertEqual([id(t) for t in result], [id(a), id(b), id(c)])

    def test_collect_tensors_empty_containers(self):
        self.assertEqual(torch._rust.collect_tensors([]), [])
        self.assertEqual(torch._rust.collect_tensors(()), [])
        self.assertEqual(torch._rust.collect_tensors({}), [])
        self.assertEqual(torch._rust.collect_tensors(set()), [])


def _leaf():
    return 1


def _middle():
    return sum(_leaf() for _ in range(3))


@unittest.skipIf(torch._rust is None, "torch._rust extension is not built")
@unittest.skipIf(sys.version_info < (3, 12), "requires Python 3.12+")
class TestCompile3(TestCase):
    hw_classification = HardwareClassification.GENERIC

    def _run(self, fn, *args):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            result = fn(*args)
        return result, out.getvalue()

    def test_prints_callees_in_order(self):
        @torch.compile3
        def top(x):
            return _middle() + x

        result, out = self._run(top, 10)
        self.assertEqual(result, 13)
        headers = [line for line in out.splitlines() if line.startswith("=== ")]
        names = [h.split()[1] for h in headers]
        self.assertEqual(
            names,
            [
                "TestCompile3.test_prints_callees_in_order.<locals>.top",
                "_middle",
                "_middle.<locals>.<genexpr>",
                "_leaf",
            ],
        )
        self.assertIn("RETURN_VALUE", out)
        self.assertIn("--- CFG ---", out)

    def test_each_code_printed_once_per_call(self):
        @torch.compile3
        def top():
            return [_leaf() for _ in range(5)]

        _, out = self._run(top)
        self.assertEqual(out.count("=== _leaf "), 1)
        _, out = self._run(top)
        self.assertEqual(out.count("=== _leaf "), 1)

    def test_disabled_after_return_and_raise(self):
        @torch.compile3
        def boom():
            raise ValueError("boom")

        with self.assertRaises(ValueError):
            self._run(boom)
        _, out = self._run(_middle)
        self.assertEqual(out, "")

    def test_preserves_metadata(self):
        def documented():
            """doc"""

        wrapped = torch.compile3(documented)
        self.assertEqual(wrapped.__name__, "documented")
        self.assertEqual(wrapped.__doc__, "doc")


# Subjects for TestBytecodeToCfg, kept at module level so their reprs are short.
def _cfg_straight(a, b):
    c = a + b
    return c * 2


def _cfg_if(c, a):
    if c:
        x = a
    else:
        x = 1
    return x


def _cfg_while(n):
    i = 0
    while i < n:
        i += 1
    return i


def _cfg_for(xs):
    t = 0
    for x in xs:
        t += x
    return t


def _cfg_try_except(a):
    try:
        y = a.f()
        a.g(y)
    except ValueError:
        return y
    return 0


def _cfg_try_finally(a):
    try:
        a.f()
    finally:
        a.g()


def _cfg_with(m):
    with m:
        m.work()


def _cfg_nested_with(a, b):
    with a:
        with b:
            a.work(b)


def _cfg_gen(xs):
    for x in xs:  # noqa: UP028 (tests FOR_ITER + YIELD_VALUE)
        yield x


def _cfg_yield_from(it):
    r = yield from it
    return r


def _cfg_comprehension(xs):
    return [x + 1 for x in xs]


def _cfg_genexpr(xs):
    return sum(x for x in xs)


def _cfg_closure(x):
    y = 1
    return lambda: x + y


def _cfg_captured_comprehension(xs):
    return [lambda: x for x in xs]


def _cfg_maybe_del(c):
    if c:
        x = 1
    del x


def _cfg_with_then_use(m):
    with m:
        y = m.g()
    return y


def _cfg_only_is_none(x):
    try:
        x is None  # noqa: B015 (a try body that can't raise)
    except ValueError:
        pass


def _cfg_class_body():
    x = 1

    class Inner:
        y = x

    return Inner


# After `except ...: pass` the handler jumps to the code after the try, which
# nothing else reaches: a chain of two blocks. Raises in it unwind to the
# second try's handler, outside the chain.
def _cfg_outline(x):
    try:
        return x.a
    except AttributeError:
        pass
    try:
        return x.b
    except KeyError:
        return "key"


_CFG_MIN_ARG_OPS = {
    "SWAP",
    "COPY",
    "LIST_APPEND",
    "SET_ADD",
    "MAP_ADD",
    "LIST_EXTEND",
    "SET_UPDATE",
    "DICT_MERGE",
    "DICT_UPDATE",
}


def _nested_code(fn):
    return next(c for c in fn.__code__.co_consts if hasattr(c, "co_code"))


def _all_code_objects(co):
    yield co
    for c in co.co_consts:
        if hasattr(c, "co_code"):
            yield from _all_code_objects(c)


@unittest.skipIf(torch._rust is None, "torch._rust extension is not built")
@unittest.skipIf(sys.version_info[:2] != (3, 12), "bytecode_to_cfg handles 3.12 only")
class TestBytecodeToCfg(TestCase):
    hw_classification = HardwareClassification.GENERIC

    def cfg(self, fn, simplify=True, origins=False):
        return torch._rust.bytecode_to_cfg(fn, simplify=simplify, origins=origins)

    def test_opcode_table(self):
        real = {
            name: op
            for name, op in dis.opmap.items()
            if op < 256 and not name.startswith("INSTRUMENTED_")
        }
        self.assertEqual(dict(torch._rust._cfg_opcodes()), real)
        for name, op in real.items():
            if name in ("CACHE", "RESERVED", "INTERPRETER_EXIT"):
                continue
            has_arg = op >= dis.HAVE_ARGUMENT
            for arg in (0, 1, 2, 3, 4, 7, 15, 0x102) if has_arg else (0,):
                for jump in (False, True):
                    try:
                        want = dis.stack_effect(op, arg if has_arg else None, jump=jump)
                    except ValueError:
                        continue
                    sig = torch._rust._cfg_signature(op, arg, jump)
                    # Opargs the compiler never emits (e.g. SWAP 1, LIST_APPEND 0).
                    if sig is None and arg < 2 and name in _CFG_MIN_ARG_OPS:
                        continue
                    self.assertIsNotNone(sig, f"{name} {arg}")
                    got = len(sig[1]) - len(sig[0])
                    if name == "FOR_ITER" and jump:
                        # dis models the jump to END_FOR; we jump past it.
                        want -= 2
                    if name == "RETURN_GENERATOR":
                        # We push the value that the following POP_TOP drops.
                        want += 1
                    self.assertEqual(got, want, f"{name} arg={arg} jump={jump}")

    def test_signatures(self):
        def sig(name, arg=0):
            return torch._rust._cfg_signature(dis.opmap[name], arg, False)

        self.assertEqual(sig("CHECK_EXC_MATCH"), ([True, True], [0, -1]))
        self.assertEqual(sig("CHECK_EG_MATCH"), ([True, True], [-1, -1]))
        self.assertEqual(sig("END_ASYNC_FOR"), ([True, True], []))
        self.assertEqual(sig("MATCH_CLASS", 1), ([True, True, True], [-1]))
        self.assertEqual(sig("MATCH_KEYS"), ([True, True], [0, 1, -1]))
        self.assertEqual(sig("GET_LEN"), ([True], [0, -1]))
        self.assertEqual(sig("GET_ANEXT"), ([True], [0, -1]))
        self.assertEqual(
            sig("WITH_EXCEPT_START"), ([True, False, False, True], [0, 1, 2, 3, -1])
        )
        self.assertEqual(sig("LIST_APPEND", 2), ([True, False, True], [0, 1]))
        self.assertEqual(sig("PUSH_EXC_INFO"), ([True], [-1, 0]))

    def test_straight(self):
        self.assertExpectedInline(
            self.cfg(_cfg_straight),
            """\
bb0:
  Resume
  %0 = LoadLocal a
  %1 = LoadLocal b
  %2 = BinaryOp + %0, %1
  StoreLocal c %2
  %3 = LoadLocal c
  %4 = Const 2
  %5 = BinaryOp * %3, %4
  return %5
""",
        )

    def test_if(self):
        self.assertExpectedInline(
            self.cfg(_cfg_if),
            """\
bb0:
  Resume
  %0 = LoadLocal c
  %1 = ToBool %0
  branch %1, bb1, bb2
bb1:
  %2 = LoadLocal a
  StoreLocal x %2
  %3 = LoadLocal x
  return %3
bb2:
  %4 = Const 1
  StoreLocal x %4
  %5 = LoadLocal x
  return %5
""",
        )

    def test_while(self):
        self.assertExpectedInline(
            self.cfg(_cfg_while),
            """\
bb0:
  Resume
  %0 = Const 0
  StoreLocal i %0
  %1 = LoadLocal i
  %2 = LoadLocal n
  %3 = CompareOp < %1, %2
  %4 = ToBool %3
  branch %4, bb1, bb3
bb1:
  %5 = LoadLocal i
  %6 = Const 1
  %7 = BinaryOp += %5, %6
  StoreLocal i %7
  %8 = LoadLocal i
  %9 = LoadLocal n
  %10 = CompareOp < %8, %9
  %11 = ToBool %10
  branch %11, bb2, bb3
bb2:
  EvalBreaker
  jump bb1
bb3:
  %12 = LoadLocal i
  return %12
""",
        )

    def test_for(self):
        self.assertExpectedInline(
            self.cfg(_cfg_for),
            """\
bb0:
  Resume
  %0 = Const 0
  StoreLocal t %0
  %1 = LoadLocal xs
  %2 = GetIter %1
  jump bb1(s0=%2)
bb1(%3 s0):
  %4 = IterNext %3
  %5 = Project 0 %4
  %6 = Project 1 %4
  branch %6, bb2(s0=%3, s1=%5), bb3
bb2(%7 s0, %8 s1):
  StoreLocal x %8
  %9 = LoadLocal t
  %10 = LoadLocal x
  %11 = BinaryOp += %9, %10
  StoreLocal t %11
  EvalBreaker
  jump bb1(s0=%7)
bb3:
  %12 = LoadLocal t
  return %12
""",
        )

    def test_for_unsimplified(self):
        self.assertExpectedInline(
            self.cfg(_cfg_for, simplify=False),
            """\
bb0:
  Resume
  %0 = Const 0
  StoreLocal t %0
  %1 = LoadLocal xs
  %2 = GetIter %1
  jump bb1(s0=%2)
bb1(%3 s0):
  %4 = IterNext %3
  %5 = Project 0 %4
  %6 = Project 1 %4
  branch %6, bb2(s0=%3, s1=%5), bb3
bb2(%7 s0, %8 s1):
  StoreLocal x %8
  %9 = LoadLocal t
  %10 = LoadLocal x
  %11 = BinaryOp += %9, %10
  StoreLocal t %11
  EvalBreaker
  jump bb1(s0=%7)
bb3:
  %12 = LoadLocal t
  return %12
""",
        )

    def test_try_except(self):
        # One block for the try body, with two unwind segments.
        self.assertExpectedInline(
            self.cfg(_cfg_try_except),
            """\
bb0:
  Resume
  %0 = LoadLocal a
  %1 = LoadMethod f %0
  %2 = Project 0 %1
  %3 = Project 1 %1
  %4 = Call 0 %2, %3
  StoreLocal y %4
  %5 = LoadLocal a
  %6 = LoadMethod g %5
  %7 = Project 0 %6
  %8 = Project 1 %6
  %9 = LoadLocal y
  %10 = Call 1 %7, %8, %9
  %11 = Const 0
  return %11
  unwind [2..14) -> bb1
bb1(%12 exc):
  %13 = PUSH_EXC_INFO 0 %12
  %14 = LoadGlobal ValueError
  %15 = CHECK_EXC_MATCH 0 %12, %14
  %16 = ToBool %15
  branch %16, bb2(s0=%13), bb3(s0=%13, s1=%12)
  unwind [1..4) -> bb4(s0=%13)
bb2(%17 s0):
  %18 = LoadLocalChecked y
  POP_EXCEPT 0 %17
  return %18
  unwind [0..2) -> bb4(s0=%17)
bb3(%19 s0, %20 s1):
  reraise %20 -> bb4(s0=%19)
bb4(%21 s0, %22 lasti, %23 exc):
  POP_EXCEPT 0 %21
  reraise %23, lasti=%22
""",
        )

    def test_try_finally(self):
        self.assertExpectedInline(
            self.cfg(_cfg_try_finally),
            """\
bb0:
  Resume
  %0 = LoadLocal a
  %1 = LoadMethod f %0
  %2 = Project 0 %1
  %3 = Project 1 %1
  %4 = Call 0 %2, %3
  %5 = LoadLocal a
  %6 = LoadMethod g %5
  %7 = Project 0 %6
  %8 = Project 1 %6
  %9 = Call 0 %7, %8
  %10 = Const None
  return %10
  unwind [2..7) -> bb1
  unwind [7..12) -> raise
bb1(%11 exc):
  %12 = PUSH_EXC_INFO 0 %11
  %13 = LoadLocal a
  %14 = LoadMethod g %13
  %15 = Project 0 %14
  %16 = Project 1 %14
  %17 = Call 0 %15, %16
  reraise %11 -> bb2(s0=%12)
  unwind [2..6) -> bb2(s0=%12)
bb2(%18 s0, %19 lasti, %20 exc):
  POP_EXCEPT 0 %18
  reraise %20, lasti=%19
""",
        )

    def test_with(self):
        self.assertExpectedInline(
            self.cfg(_cfg_with),
            """\
bb0:
  Resume
  %0 = LoadLocal m
  %1 = BEFORE_WITH 0 %0
  %2 = Project 0 %1
  %3 = Project 1 %1
  %4 = LoadLocal m
  %5 = LoadMethod work %4
  %6 = Project 0 %5
  %7 = Project 1 %5
  %8 = Call 0 %6, %7
  %9 = Const None
  %10 = Const None
  %11 = Const None
  %12 = Call 2 %2, %9, %10, %11
  %13 = Const None
  return %13
  unwind [6..13) -> bb1(s0=%2)
  unwind [13..15) -> raise
bb1(%14 s0, %15 lasti, %16 exc):
  %17 = PUSH_EXC_INFO 0 %16
  %18 = WITH_EXCEPT_START 0 %14, %16
  %19 = ToBool %18
  branch %19, bb2(s2=%17), bb3(s1=%15, s2=%17, s3=%16)
  unwind [1..3) -> bb4(s2=%17)
bb2(%20 s2):
  POP_EXCEPT 0 %20
  %21 = Const None
  return %21
bb3(%22 s1, %23 s2, %24 s3):
  reraise %24, lasti=%22 -> bb4(s2=%23)
bb4(%25 s2, %26 lasti, %27 exc):
  POP_EXCEPT 0 %25
  reraise %27, lasti=%26
""",
        )

    def test_nested_with(self):
        self.assertExpectedInline(
            self.cfg(_cfg_nested_with),
            """\
bb0:
  Resume
  %0 = LoadLocal a
  %1 = BEFORE_WITH 0 %0
  %2 = Project 0 %1
  %3 = Project 1 %1
  %4 = LoadLocal b
  %5 = BEFORE_WITH 0 %4
  %6 = Project 0 %5
  %7 = Project 1 %5
  %8 = LoadLocal a
  %9 = LoadMethod work %8
  %10 = Project 0 %9
  %11 = Project 1 %9
  %12 = LoadLocal b
  %13 = Call 1 %10, %11, %12
  %14 = Const None
  %15 = Const None
  %16 = Const None
  %17 = Call 2 %6, %14, %15, %16
  jump bb3(s0=%2)
  unwind [6..10) -> bb6(s0=%2)
  unwind [10..18) -> bb1(s0=%2, s1=%6)
  unwind [18..19) -> bb6(s0=%2)
bb1(%18 s0, %19 s1, %20 lasti, %21 exc):
  %22 = PUSH_EXC_INFO 0 %21
  %23 = WITH_EXCEPT_START 0 %19, %21
  %24 = ToBool %23
  branch %24, bb2(s0=%18, s3=%22), bb4(s0=%18, s2=%20, s3=%22, s4=%21)
  unwind [1..3) -> bb5(s0=%18, s3=%22)
bb2(%25 s0, %26 s3):
  POP_EXCEPT 0 %26
  EvalBreaker
  jump bb3(s0=%25)
  unwind [1..2) -> bb6(s0=%25)
bb3(%27 s0):
  %28 = Const None
  %29 = Const None
  %30 = Const None
  %31 = Call 2 %27, %28, %29, %30
  %32 = Const None
  return %32
bb4(%33 s0, %34 s2, %35 s3, %36 s4):
  reraise %36, lasti=%34 -> bb5(s0=%33, s3=%35)
bb5(%37 s0, %38 s3, %39 lasti, %40 exc):
  POP_EXCEPT 0 %38
  reraise %40, lasti=%39 -> bb6(s0=%37)
bb6(%41 s0, %42 lasti, %43 exc):
  %44 = PUSH_EXC_INFO 0 %43
  %45 = WITH_EXCEPT_START 0 %41, %43
  %46 = ToBool %45
  branch %46, bb7(s2=%44), bb8(s1=%42, s2=%44, s3=%43)
  unwind [1..3) -> bb9(s2=%44)
bb7(%47 s2):
  POP_EXCEPT 0 %47
  %48 = Const None
  return %48
bb8(%49 s1, %50 s2, %51 s3):
  reraise %51, lasti=%49 -> bb9(s2=%50)
bb9(%52 s2, %53 lasti, %54 exc):
  POP_EXCEPT 0 %52
  reraise %54, lasti=%53
""",
        )

    def test_generator(self):
        self.assertExpectedInline(
            self.cfg(_cfg_gen),
            """\
bb0:
  %0 = RETURN_GENERATOR 0
  Resume
  %1 = LoadLocal xs
  %2 = GetIter %1
  jump bb1(s0=%2)
  unwind [1..4) -> bb4
bb1(%3 s0):
  %4 = IterNext %3
  %5 = Project 0 %4
  %6 = Project 1 %4
  branch %6, bb2(s0=%3, s1=%5), bb3
  unwind [0..3) -> bb4
bb2(%7 s0, %8 s1):
  StoreLocal x %8
  %9 = LoadLocal x
  %10 = Yield %7, %9
  Resume
  EvalBreaker
  jump bb1(s0=%7)
  unwind [2..5) -> bb4
bb3:
  %11 = Const None
  return %11
bb4(%12 lasti, %13 exc):
  %14 = CALL_INTRINSIC_1 3 %13
  reraise %14, lasti=%12
""",
        )

    def test_yield_from(self):
        self.assertExpectedInline(
            self.cfg(_cfg_yield_from),
            """\
bb0:
  %0 = RETURN_GENERATOR 0
  Resume
  %1 = LoadLocal it
  %2 = GET_YIELD_FROM_ITER 0 %1
  %3 = Const None
  jump bb1(s0=%2, s1=%3)
  unwind [1..5) -> bb5
bb1(%4 s0, %5 s1):
  %6 = Send %4, %5
  %7 = Project 0 %6
  %8 = Project 1 %6
  branch %8, bb2(s0=%4, s1=%7), bb4(s1=%7)
  unwind [0..3) -> bb5
bb2(%9 s0, %10 s1):
  %11 = Yield %9, %10
  Resume
  jump bb1(s0=%9, s1=%11)
  unwind [0..1) -> bb3(s0=%9)
  unwind [1..2) -> bb5
bb3(%12 s0, %13 exc):
  %14 = CLEANUP_THROW 0 %12, unknown, %13
  %15 = Project 0 %14
  %16 = Project 1 %14
  EvalBreaker
  jump bb4(s1=%16)
  unwind [0..3) -> bb5
  unwind [3..4) -> raise
bb4(%17 s1):
  StoreLocal r %17
  %18 = LoadLocal r
  return %18
bb5(%19 lasti, %20 exc):
  %21 = CALL_INTRINSIC_1 3 %20
  reraise %21, lasti=%19
""",
        )

    def test_comprehension(self):
        self.assertExpectedInline(
            self.cfg(_cfg_comprehension),
            """\
bb0:
  Resume
  %0 = LoadLocal xs
  %1 = GetIter %0
  %2 = LoadLocal x
  StoreLocal x null
  %3 = BUILD_LIST 0
  jump bb1(s0=%2, s1=%3, s2=%1)
bb1(%4 s0, %5 s1, %6 s2):
  %7 = IterNext %6
  %8 = Project 0 %7
  %9 = Project 1 %7
  branch %9, bb2(s0=%4, s1=%5, s2=%6, s3=%8), bb3(s0=%4, s1=%5)
  unwind [0..3) -> bb4(s0=%4)
bb2(%10 s0, %11 s1, %12 s2, %13 s3):
  StoreLocal x %13
  %14 = LoadLocal x
  %15 = Const 1
  %16 = BinaryOp + %14, %15
  LIST_APPEND 2 %11, %16
  EvalBreaker
  jump bb1(s0=%10, s1=%11, s2=%12)
  unwind [3..6) -> bb4(s0=%10)
bb3(%17 s0, %18 s1):
  StoreLocal x %17
  return %18
bb4(%19 s0, %20 exc):
  StoreLocal x %19
  reraise %20
""",
        )

    def test_genexpr(self):
        self.assertExpectedInline(
            self.cfg(_nested_code(_cfg_genexpr)),
            """\
bb0:
  %0 = RETURN_GENERATOR 0
  Resume
  %1 = LoadLocal .0
  jump bb1(s0=%1)
  unwind [1..3) -> bb4
bb1(%2 s0):
  %3 = IterNext %2
  %4 = Project 0 %3
  %5 = Project 1 %3
  branch %5, bb2(s0=%2, s1=%4), bb3
  unwind [0..3) -> bb4
bb2(%6 s0, %7 s1):
  StoreLocal x %7
  %8 = LoadLocal x
  %9 = Yield %6, %8
  Resume
  EvalBreaker
  jump bb1(s0=%6)
  unwind [2..5) -> bb4
bb3:
  %10 = Const None
  return %10
bb4(%11 lasti, %12 exc):
  %13 = CALL_INTRINSIC_1 3 %12
  reraise %13, lasti=%11
""",
        )

    def test_closure(self):
        self.assertExpectedInline(
            self.cfg(_cfg_closure),
            """\
bb0:
  MakeCell x
  MakeCell y
  Resume
  %0 = Const 1
  StoreDeref y %0
  %1 = LoadLocal x
  %2 = LoadLocal y
  %3 = BuildTuple 2 %1, %2
  %4 = Const <code _cfg_closure.<locals>.<lambda>>
  %5 = MakeFunction 0x8 %3, %4
  return %5
""",
        )
        self.assertExpectedInline(
            self.cfg(_nested_code(_cfg_closure)),
            """\
bb0:
  CopyFreeVars 2
  Resume
  %0 = LoadDeref x
  %1 = LoadDeref y
  %2 = BinaryOp + %0, %1
  return %2
""",
        )

    def test_captured_comprehension(self):
        self.assertExpectedInline(
            self.cfg(_cfg_captured_comprehension),
            """\
bb0:
  MakeCell x
  Resume
  %0 = LoadLocal xs
  %1 = GetIter %0
  %2 = LoadLocal x
  StoreLocal x null
  MakeCell x
  %3 = BUILD_LIST 0
  jump bb1(s0=%2, s1=%3, s2=%1)
bb1(%4 s0, %5 s1, %6 s2):
  %7 = IterNext %6
  %8 = Project 0 %7
  %9 = Project 1 %7
  branch %9, bb2(s0=%4, s1=%5, s2=%6, s3=%8), bb3(s0=%4, s1=%5)
  unwind [0..3) -> bb4(s0=%4)
bb2(%10 s0, %11 s1, %12 s2, %13 s3):
  StoreDeref x %13
  %14 = LoadLocal x
  %15 = BuildTuple 1 %14
  %16 = Const <code _cfg_captured_comprehension.<locals>.<lambda>>
  %17 = MakeFunction 0x8 %15, %16
  LIST_APPEND 2 %11, %17
  EvalBreaker
  jump bb1(s0=%10, s1=%11, s2=%12)
  unwind [6..7) -> bb4(s0=%10)
bb3(%18 s0, %19 s1):
  StoreLocal x %18
  return %19
bb4(%20 s0, %21 exc):
  StoreLocal x %20
  reraise %21
""",
        )

    def test_del_local(self):
        self.assertExpectedInline(
            self.cfg(_cfg_maybe_del),
            """\
bb0:
  Resume
  %0 = LoadLocal c
  %1 = ToBool %0
  branch %1, bb1, bb2
bb1:
  %2 = Const 1
  StoreLocal x %2
  jump bb2
bb2:
  DeleteLocal x
  %3 = Const None
  return %3
""",
        )

    def test_with_then_use(self):
        self.assertExpectedInline(
            self.cfg(_cfg_with_then_use),
            """\
bb0:
  Resume
  %0 = LoadLocal m
  %1 = BEFORE_WITH 0 %0
  %2 = Project 0 %1
  %3 = Project 1 %1
  %4 = LoadLocal m
  %5 = LoadMethod g %4
  %6 = Project 0 %5
  %7 = Project 1 %5
  %8 = Call 0 %6, %7
  StoreLocal y %8
  %9 = Const None
  %10 = Const None
  %11 = Const None
  %12 = Call 2 %2, %9, %10, %11
  %13 = LoadLocal y
  return %13
  unwind [6..14) -> bb1(s0=%2)
  unwind [14..16) -> raise
bb1(%14 s0, %15 lasti, %16 exc):
  %17 = PUSH_EXC_INFO 0 %16
  %18 = WITH_EXCEPT_START 0 %14, %16
  %19 = ToBool %18
  branch %19, bb2(s2=%17), bb3(s1=%15, s2=%17, s3=%16)
  unwind [1..3) -> bb4(s2=%17)
bb2(%20 s2):
  POP_EXCEPT 0 %20
  %21 = LoadLocalChecked y
  return %21
bb3(%22 s1, %23 s2, %24 s3):
  reraise %24, lasti=%22 -> bb4(s2=%23)
bb4(%25 s2, %26 lasti, %27 exc):
  POP_EXCEPT 0 %25
  reraise %27, lasti=%26
""",
        )

    def test_unreached_handler_dropped(self):
        self.assertExpectedInline(
            self.cfg(_cfg_only_is_none),
            """\
bb0:
  Resume
  %0 = LoadLocal x
  %1 = Const None
  %2 = IS_OP 0 %0, %1
  %3 = Const None
  return %3
""",
        )

    def test_origins(self):
        # Raising ops print the byte offset of the bytecode they came from.
        raising = {
            i.offset
            for i in dis.get_instructions(_cfg_try_except)
            if i.opname in ("LOAD_ATTR", "CALL", "LOAD_GLOBAL", "LOAD_FAST_CHECK")
        }
        out = self.cfg(_cfg_try_except, origins=True)
        printed = {int(m) for m in re.findall(r" @(\d+)", out)}
        self.assertTrue(raising <= printed, (raising, printed))
        self.assertExpectedInline(out.splitlines()[2], """  %0 = LoadLocal a""")

    def test_class_body(self):
        self.assertExpectedInline(
            self.cfg(_nested_code(_cfg_class_body)),
            """\
bb0:
  CopyFreeVars 1
  Resume
  %0 = LOAD_NAME 0
  STORE_NAME 1 %0
  %1 = Const '_cfg_class_body.<locals>.Inner'
  STORE_NAME 2 %1
  %2 = LOAD_LOCALS 0
  %3 = LoadFromDictOrDeref x %2
  STORE_NAME 3 %3
  %4 = Const None
  return %4
""",
        )

    def test_extended_arg(self):
        body = "".join(f"        x = x + {i}\n" for i in range(150))
        ns = {}
        exec(f"def big(x):\n    if x:\n{body}    return x\n", ns)
        big = ns["big"]
        self.assertTrue(
            any(i.opname == "EXTENDED_ARG" for i in dis.get_instructions(big))
        )
        out = self.cfg(big)
        self.assertIn("branch %1, bb1, bb2\n", out)
        self.assertTrue(
            out.endswith("  jump bb2\nbb2:\n  %452 = LoadLocal x\n  return %452\n")
        )

    def test_outline(self):
        self.assertExpectedInline(
            torch._rust.bytecode_to_cfg(_cfg_outline, outline="merge"),
            """\
bb0:
  Resume
  %0 = LoadLocal x
  %1 = LoadAttr a %0
  return %1
  unwind [2..3) -> bb1
bb1(%2 exc):
  %3 = PUSH_EXC_INFO 0 %2
  %4 = LoadGlobal AttributeError
  %5 = CHECK_EXC_MATCH 0 %2, %4
  %6 = ToBool %5
  branch %6, bb2(s0=%3), bb7(s0=%3, s1=%2)
  unwind [1..4) -> bb8(s0=%3)
bb2(%7 s0):
  POP_EXCEPT 0 %7
  %8 = LoadLocal x
  %9 = LoadAttr b %8
  return %9
  unwind [2..3) -> bb3
  origin:
    bb0(%0 s0):
      POP_EXCEPT 0 %0
      jump bb1
    bb1:
      %1 = LoadLocal x
      %2 = LoadAttr b %1
      leave(%2)
      unwind [1..2) -> outer 0()
bb3(%10 exc):
  %11 = PUSH_EXC_INFO 0 %10
  %12 = LoadGlobal KeyError
  %13 = CHECK_EXC_MATCH 0 %10, %12
  %14 = ToBool %13
  branch %14, bb4(s0=%11), bb5(s0=%11, s1=%10)
  unwind [1..4) -> bb6(s0=%11)
bb4(%15 s0):
  POP_EXCEPT 0 %15
  %16 = Const 'key'
  return %16
bb5(%17 s0, %18 s1):
  reraise %18 -> bb6(s0=%17)
bb6(%19 s0, %20 lasti, %21 exc):
  POP_EXCEPT 0 %19
  reraise %21, lasti=%20
bb7(%22 s0, %23 s1):
  reraise %23 -> bb8(s0=%22)
bb8(%24 s0, %25 lasti, %26 exc):
  POP_EXCEPT 0 %24
  reraise %26, lasti=%25
""",
        )

    def test_robustness_sweep(self):
        modules = [
            "json",
            "inspect",
            "dis",
            "ast",
            "asyncio.base_events",
            "contextlib",
            "typing",
            "torch.nn.modules.module",
        ]
        for name in modules:
            src = inspect.getsource(importlib.import_module(name))
            for co in _all_code_objects(compile(src, name, "exec")):
                # Raises NotImplementedError/RuntimeError on failure.
                torch._rust.bytecode_to_cfg(co, simplify=False)
                for outline in (None, "merge", "origin"):
                    torch._rust.bytecode_to_cfg(co, outline=outline)


def _exec_tb(e):
    out = []
    tb = e.__traceback__
    while tb is not None:
        out.append(
            (
                tb.tb_frame.f_code.co_name,
                tb.tb_lineno - tb.tb_frame.f_code.co_firstlineno,
            )
        )
        tb = tb.tb_next
    return out


# Here the chains' raises unwind to handlers outside them with stack values
# as args.
def _cfg_outline_context():
    context = Exception("context")
    try:
        raise context
    except Exception:
        try:
            raise Exception("caught")
        except Exception:
            pass
        try:
            raise Exception("new")
        except Exception as exc:
            raised = exc
    return raised.__context__ is context


def _cfg_outline_tb(fn, x):
    try:
        fn(x)
    except TypeError as e:
        return _exec_tb(e)


class _Del:
    log = []

    def __init__(self, n):
        self.n = n

    def __del__(self):
        _Del.log.append(self.n)


@unittest.skipIf(torch._rust is None, "torch._rust extension is not built")
@unittest.skipIf(sys.version_info[:2] != (3, 12), "the CFG executor handles 3.12 only")
class TestCfgExecution(TestCase):
    hw_classification = HardwareClassification.GENERIC

    # Runs fn both normally and under the CFG executor; results must match.
    def same(self, fn, *args):
        expected = fn(*args)
        self.assertEqual(torch._rust.call_with_cfg_execution(fn, *args), expected)
        return expected

    def test_control_flow(self):
        def f(n):
            t = 0
            for i in range(n):
                if i % 3 == 0:
                    continue
                t += i
            while t > 10:
                t -= 7
            a, *b, c = [1, 2, 3, 4]
            return t, a, b, c, {"x": 1, "y": [i for i in range(3)]}, f"{t!r:>4}"

        self.same(f, 20)

    def test_exceptions_and_tracebacks(self):
        def deep():
            raise ValueError("boom")

        def f():
            try:
                deep()
            except ValueError as e:
                tb = _exec_tb(e)
            try:
                try:
                    1 / 0
                except ZeroDivisionError as e:
                    raise KeyError("k") from e
            except KeyError as e:
                chained = repr(e.__cause__)
            finally:
                done = sys.exception()
            return tb, chained, done

        self.assertEqual(self.same(f)[0], [("f", 2), ("deep", 1)])

    def test_unbound_local_and_name_error(self):
        def f():
            if False:
                y = 1
            try:
                return y
            except UnboundLocalError as e:
                msg = str(e)
            try:
                return undefined_name
            except NameError as e:
                return msg, str(e), e.name

        self.same(f)

    def test_generators_and_coroutines(self):
        def f():
            def inner():
                r = yield 1
                return r

            def outer():
                v = yield from inner()
                yield v

            g = outer()
            out = [next(g), g.send("sent")]
            gen = (x * x for x in range(4))

            async def coro(x):
                await asyncio.sleep(0)
                return x * 2

            async def agen():
                yield 1
                yield 2

            async def main():
                return await coro(21), [x async for x in agen()]

            return out, list(gen), asyncio.run(main())

        import asyncio

        self.same(f)

    def test_throw_into_generator_created_outside(self):
        # The executor first sees the generator's code when the throw resumes
        # it (as when GC closes a generator), so it builds it then.
        src = "def gen(log):\n    try:\n        yield 1\n    finally:\n        log.append('finally')\n"

        def started(log):
            ns = {}
            exec(src, ns)
            g = ns["gen"](log)
            next(g)
            return g

        def close(g):
            g.close()

        def throw(g):
            try:
                g.throw(ValueError("thrown"))
            except ValueError as e:
                return str(e)

        log = []
        run = torch._rust.call_with_cfg_execution
        out = [run(close, started(log)), run(throw, started(log))]
        self.assertEqual((out, log), ([None, "thrown"], ["finally", "finally"]))

    def test_lifetimes(self):
        def f():
            _Del.log.clear()
            _Del(1)
            _Del.log.append("temp dropped")
            x = _Del(2)
            _Del.log.append("local alive")
            x = None  # noqa: F841
            _Del.log.append("local rebound")
            y = _Del(3)  # noqa: F841
            return list(_Del.log)

        self.same(f)

    def test_recursion_limit(self):
        def rec(n):
            return rec(n + 1)

        def f():
            try:
                rec(0)
            except RecursionError as e:
                return str(e)

        self.same(f)

    def test_pep695_intrinsics(self):
        ns = {}
        src = "def f[T, *Ts, **P](x: T) -> T:\n    return x\ntype Alias[K] = list[K]\n"
        exec(src, ns)

        def g():
            p = ns["f"].__type_params__
            return [(type(t).__name__, t.__name__) for t in p], ns["Alias"].__value__

        self.same(g)

    def test_eval_breaker_on_generator_resume(self):
        class Boom(Exception):
            pass

        def handler(*args):
            raise Boom

        def f():
            def g():
                yield 0
                try:
                    yield 1
                except Boom:
                    yield "caught"

            it = g()
            next(it)
            next(it)
            # zip calls interrupt_main (which only trips the signal) and then
            # resumes g from C, so g's RESUME is the first eval-breaker check.
            return list(zip(map(_thread.interrupt_main, [signal.SIGUSR1]), it))

        old = signal.signal(signal.SIGUSR1, handler)
        try:
            self.assertEqual(self.same(f), [(None, "caught")])
        finally:
            signal.signal(signal.SIGUSR1, old)

    def test_outlined_execution(self):
        class A:
            a = 1

        class B:
            b = 2

        class K:
            @property
            def b(self):
                raise KeyError

        class T:
            @property
            def b(self):
                raise TypeError("t")

        # Fresh code objects: the executor caches each one's CFG.
        def fresh(f):
            return type(f)(f.__code__.replace(co_name=f.__name__), f.__globals__)

        for mode in ("merge", "origin"):
            fn = fresh(_cfg_outline)
            torch._rust._set_cfg_outline(mode)
            try:
                for x in (A(), B(), K()):
                    self.same(fn, x)
                self.same(_cfg_outline_tb, fn, T())
                self.assertTrue(self.same(fresh(_cfg_outline_context)))
            finally:
                torch._rust._set_cfg_outline(None)

    def test_tracing_is_refused(self):
        def f():
            sys.settrace(lambda *a: None)
            try:
                return (lambda: 1)()
            finally:
                sys.settrace(None)

        with self.assertRaisesRegex(RuntimeError, "settrace"):
            torch._rust.call_with_cfg_execution(f)
        sys.settrace(None)


if __name__ == "__main__":
    run_tests()
