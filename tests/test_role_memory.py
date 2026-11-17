"""Dependency-free unit checks for the two-role memory substrate.

Run: python -m unittest tests.test_role_memory -v
These are NUMERICAL PLUMBING TESTS, not tests on a trained Qwen.
"""
import unittest

from qimag.role_memory import RoleBinder, topk, cosine


class TestBinding(unittest.TestCase):
    def setUp(self):
        self.width = 513  # tests padding to the next power of two
        self.binder = RoleBinder(self.width, seed=20261118)
        self.a = [0.0]*self.width
        self.b = [0.0]*self.width
        self.a[3], self.a[7], self.a[18] = 1.0, 0.5, 0.3
        self.b[202], self.b[301], self.b[500] = 1.2, 0.6, 0.2

    def test_single_role_inverse_is_exact(self):
        for role, vec in (("first", self.a), ("second", self.b)):
            recovered = self.binder.unbind(role, self.binder.bind(role, vec))
            self.assertLess(max(abs(a-b) for a,b in zip(vec, recovered)), 1e-10)

    def test_two_independent_sparse_bags_survive_packing(self):
        packed = self.binder.pack(self.a, self.b)
        self.assertEqual(len(packed.values), 1024)
        self.assertEqual(topk(self.binder.read(packed, "first"), 1)[0][0], 3)
        self.assertEqual(topk(self.binder.read(packed, "second"), 1)[0][0], 202)
        self.assertGreater(cosine(self.a, self.binder.read(packed, "first")), .7)
        self.assertGreater(cosine(self.b, self.binder.read(packed, "second")), .7)

    def test_swapping_roles_returns_the_other_thought(self):
        packed = self.binder.pack(self.a, self.b)
        self.assertEqual(topk(self.binder.read(packed, "first"), 1)[0][0], 3)
        self.assertEqual(topk(self.binder.swap_labels(packed)["first"], 1)[0][0], 202)

    def test_counterfactual_fork_does_not_edit_root(self):
        root = self.binder.pack(self.a, self.b)
        change = [0.0]*self.width
        change[3],change[88] = -1,2
        branch = self.binder.fork(root,"first",change)
        self.assertEqual(root.tags,("inferred","inferred"))
        self.assertEqual(branch.tags,("imagined","inferred"))
        self.assertEqual(topk(self.binder.read(root,"first"),1)[0][0],3)
        self.assertEqual(topk(self.binder.read(branch,"first"),1)[0][0],88)
        self.assertEqual(topk(self.binder.read(branch,"second"),1)[0][0],202)

    def test_reproducible_keys(self):
        twin = RoleBinder(self.width,seed=20261118)
        self.assertEqual(twin.pack(self.a,self.b),self.binder.pack(self.a,self.b))


if __name__ == "__main__":
    unittest.main()
