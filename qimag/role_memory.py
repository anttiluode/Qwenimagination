"""Small, transparent ROLE binding layer for neural concept readings.

The original Qwenimagination memory was an unordered bag of J-lens concepts.
Here two such bags are stored in ONE vector, with reversible signed-permutation
role keys. The keys are engineered, not learned; this is an information-
preserving interface experiment, not evidence that Qwen learned bindings.

With two simultaneously bound items there is crosstalk. Its magnitude is
measured and compared to controls, not assumed to vanish.

No torch/numpy dependency: unit-testable without model weights.
"""
from dataclasses import dataclass
import random


ROLES = ("first", "second")


def topk(values, k=8):
    """Rank indices by positive magnitude; keep original positive amplitudes."""
    indices = sorted(range(len(values)), key=lambda i: (-values[i], i))[:k]
    return [(i, max(0.0, values[i])) for i in indices if values[i] > 0]


def sparse(values, k=8):
    out = [0.0] * len(values)
    for i, value in topk(values, k):
        out[i] = value
    return out


@dataclass(frozen=True)
class PackedState:
    """No stored source bags: just one numeric vector and small source labels."""
    values: tuple
    tags: tuple = ("inferred", "inferred")


class RoleBinder:
    def __init__(self, width, seed=20261118):
        if width < 16:
            raise ValueError("role binding requires at least 16 dimensions")
        self.width = int(width)
        rng = random.Random(seed)
        self.keys = {}
        for role in ROLES:
            perm = list(range(width))
            rng.shuffle(perm)
            signs = [1 if rng.getrandbits(1) else -1 for _ in range(width)]
            self.keys[role] = (perm, signs)

    def _check(self, values):
        if len(values) != self.width:
            raise ValueError(f"expected width {self.width}, got {len(values)}")

    def bind(self, role, values):
        self._check(values)
        perm, signs = self.keys[role]
        return [signs[i] * float(values[perm[i]]) for i in range(self.width)]

    def unbind(self, role, values):
        self._check(values)
        perm, signs = self.keys[role]
        result = [0.0] * self.width
        for i in range(self.width):
            result[perm[i]] = signs[i] * float(values[i])
        return result

    def pack(self, first, second, tags=("inferred", "inferred")):
        a, b = self.bind("first", first), self.bind("second", second)
        if len(tags) != 2:
            raise ValueError("two source tags required")
        return PackedState(tuple(x + y for x, y in zip(a, b)), tuple(tags))

    def read(self, state, role):
        return self.unbind(role, state.values)

    def fork(self, state, role, delta, source="imagined"):
        """Add a *supplied* hypothetical change to one role in a fork.

        This intentionally doesn't learn or generate a transition. It only
        provides counterfactual isolation for a future learned dynamics model.
        """
        change = self.bind(role, delta)
        tags = list(state.tags)
        tags[ROLES.index(role)] = source
        return PackedState(
            tuple(x + y for x, y in zip(state.values, change)),
            tuple(tags),
        )

    def swap_labels(self, state):
        """Adversarial intervention: reverse only the label interpreted at read."""
        return {role: self.read(state, other) for role, other in
                (("first", "second"), ("second", "first"))}


def cosine(x, y):
    from math import sqrt
    dot = sum(a * b for a, b in zip(x, y))
    aa = sqrt(sum(a * a for a in x))
    bb = sqrt(sum(b * b for b in y))
    return dot / (aa * bb) if aa > 0 and bb > 0 else 0.0
