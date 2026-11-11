# Owner(s): ["module: devx"]

import contextlib
import io
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


if __name__ == "__main__":
    run_tests()
