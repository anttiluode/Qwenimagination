"""Exercise the real cache and hook path on a tiny random Qwen, offline."""
import unittest
from types import SimpleNamespace
import torch
from qimag.model import load_stub
from qimag.lens import Injector
from recall_slots import LABELS, prompt_parts, prefix_cache, query, run_pair, summary


class DelayedRecall(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.lm = load_stub("cpu", layers=24, d=16)
        # Markers and question words must be distinct in the stand-in tokenizer.
        old = self.lm.tok
        labels = ["Alpha", " Beta", "Beta", " Alpha", ":"]
        for token in labels:
            if token not in old.idx:
                old.idx[token] = len(old.vocab)
                old.vocab.append(token)
        self.lm.model.resize_token_embeddings(len(old.vocab))

    def test_cached_questions_match_full_forward_with_two_marker_writes(self):
        lm = self.lm
        base, marks, tails = prompt_parts(lm, LABELS, "Some unrelated text.")
        writes = [torch.randn(lm.d), torch.randn(lm.d)]
        cache = prefix_cache(lm, base, marks, writes)
        plain = prefix_cache(lm, base)
        original_len = cache.get_seq_length()
        for label in LABELS[::-1]:
            ids = base + tails[label]
            hook = Injector(lm, 22)
            try:
                hook.delta = torch.zeros(len(ids), lm.d)
                hook.delta[marks] = torch.stack(writes)
                with torch.no_grad():
                    expected = lm.logits(torch.tensor([ids]))[0, -1]
            finally:
                hook.remove()
            actual = query(lm, cache, tails[label])
            torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
            self.assertGreater((actual-query(lm, plain, tails[label])).abs().max().item(), 1e-6)
            self.assertEqual(cache.get_seq_length(), original_len)

    def test_fully_crossed_design_and_no_write_null(self):
        lm = self.lm
        targets = ["sheep", "hammer"]
        setup = SimpleNamespace(choice_ids=[lm.single_token(w) for w in targets], choice_words=targets)
        vectors = {word: torch.randn(lm.d) for word in targets}
        pair = [("", word, "") for word in targets]
        rows = run_pair(lm, setup, 0, pair, vectors, vectors)
        self.assertEqual(len(rows), 32)
        result = summary(rows, True)
        self.assertAlmostEqual(result["arms"]["none"]["mean_margin"], 0)
        self.assertFalse(result["role_signal"])


if __name__ == "__main__":
    unittest.main()
