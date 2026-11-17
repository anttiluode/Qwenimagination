"""Two-role holographic memory for J-lens concept bags.

One packed vector holds two sparse concept readings. Different deterministic
orthogonal transforms bind their ROLE ("first"/"second"); inverse transforms
approximately recover each bag despite crosstalk. There is no text store.

The role transforms and provenance tags are ENGINEERED, not learned.
An improved Qwen recall result would test an interface, not prove a neural
model spontaneously learned relations or experienced imagination.

Pure Python, no torch/numpy. Code width is the next power of two.
"""
from dataclasses import dataclass
import random
from math import sqrt

ROLES = ("first", "second")


def topk(values, k=8):
    indices = sorted(range(len(values)), key=lambda i: (-values[i], i))[:k]
    return [(i, max(0.0, values[i])) for i in indices if values[i] > 0]


def sparse(values, k=8):
    out = [0.0] * len(values)
    for i, value in topk(values, k):
        out[i] = value
    return out


@dataclass(frozen=True)
class PackedState:
    values: tuple
    tags: tuple = ("inferred", "inferred")


def _hadamard(values):
    """Normalized Walsh-Hadamard transform, its own inverse."""
    z = list(values)
    n = len(z)
    h = 1
    while h < n:
        for i in range(0, n, 2*h):
            for j in range(i, i+h):
                a, b = z[j], z[j+h]
                z[j], z[j+h] = a+b, a-b
        h *= 2
    scale = 1 / sqrt(n)
    return [v * scale for v in z]


class RoleBinder:
    def __init__(self, width, seed=20261118):
        if width < 16:
            raise ValueError("at least 16 concept dimensions required")
        self.width = int(width)
        self.code_width = 1 << (self.width-1).bit_length()
        rng = random.Random(seed)
        self.keys = {}
        for role in ROLES:
            perm = list(range(self.code_width))
            rng.shuffle(perm)
            signs_in = [1 if rng.getrandbits(1) else -1 for _ in range(self.code_width)]
            signs_out = [1 if rng.getrandbits(1) else -1 for _ in range(self.code_width)]
            self.keys[role] = (perm, signs_in, signs_out)

    def bind(self, role, values):
        if len(values) != self.width:
            raise ValueError("wrong concept width")
        perm, si, so = self.keys[role]
        padded = list(map(float, values)) + [0.0]*(self.code_width-self.width)
        z = _hadamard([a*b for a, b in zip(si, padded)])
        return [so[j]*z[perm[j]] for j in range(self.code_width)]

    def unbind(self, role, values):
        if len(values) != self.code_width:
            raise ValueError("wrong encoded width")
        perm, si, so = self.keys[role]
        z = [0.0]*self.code_width
        for j in range(self.code_width):
            z[perm[j]] = so[j]*float(values[j])
        x = _hadamard(z)
        return [si[j]*x[j] for j in range(self.width)]

    def pack(self, first, second, tags=("inferred", "inferred")):
        a, b = self.bind("first", first), self.bind("second", second)
        if len(tags) != 2:
            raise ValueError("two source tags required")
        return PackedState(tuple(x+y for x, y in zip(a, b)), tuple(tags))

    def read(self, state, role):
        return self.unbind(role, state.values)

    def fork(self, state, role, delta, source="imagined"):
        """Fork and add an externally SUPPLIED hypothetical change to a role.

        This does not generate, learn or evaluate that change. Only isolation
        of branches is tested; genuine imagination needs learned dynamics.
        """
        change = self.bind(role, delta)
        tags = list(state.tags)
        tags[ROLES.index(role)] = source
        return PackedState(
            tuple(x+y for x, y in zip(state.values, change)), tuple(tags))

    def swap_labels(self, state):
        return {"first": self.read(state, "second"),
                "second": self.read(state, "first")}


def cosine(x, y):
    dot = sum(a*b for a, b in zip(x, y))
    aa = sqrt(sum(a*a for a in x))
    bb = sqrt(sum(b*b for b in y))
    return dot/(aa*bb) if aa > 0 and bb > 0 else 0.0
