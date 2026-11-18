"""Exercise Qwen's punctuation/newline BPE merges without downloading weights."""
import unittest

from tokenizers import Tokenizer, decoders, models, pre_tokenizers
from transformers import Qwen2TokenizerFast

from qimag.items import FILLERS
from qimag.model import Lm
from recall_slots import LABELS, prompt_parts


class MergedBoundaries(unittest.TestCase):
    def test_cache_and_markers_match_unsplit_chat(self):
        # A small merge table covering the exact punctuation around our cuts.
        chars = sorted(pre_tokenizers.ByteLevel.alphabet())
        merges = [("Ċ", "Ċ"), (".", "ĊĊ"), (":", "ĊĊ"), (":", "Ċ")]
        vocab = {s: i for i, s in enumerate(chars + [a+b for a, b in merges])}
        backend = Tokenizer(models.BPE(vocab, merges))
        backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
        backend.decoder = decoders.ByteLevel()
        tok = Qwen2TokenizerFast(tokenizer_object=backend)
        tok.chat_template = "{% for m in messages %}{{ m.role }}\n{{ m.content }}{% endfor %}\nassistant\n"
        lm = object.__new__(Lm)
        lm.tok = tok
        for filler in FILLERS:
            for order in (LABELS, LABELS[::-1]):
                for words in (None, {"Alpha": "hammer", "Beta": "sheep"}):
                    base, marks, tails = prompt_parts(lm, order, filler, words)
                    prefix = "Remember the word stored under each label."
                    for label in order:
                        prefix += "\n" + label + ":" + (" " + words[label] if words else "")
                    prefix += "\n\n" + filler
                    self.assertTrue(lm.decode(base).endswith(filler + "\n\n"))
                    if not words:
                        self.assertEqual([lm.decode([base[i]]) for i in marks], [":\n", ":\n\n"])
                    for label, tail in tails.items():
                        prompt = prefix + "\n\nWhich word was stored under " + label + "? Answer with one word."
                        self.assertEqual(base + tail, lm.chat([("user", prompt)], "The answer was"))
                        self.assertTrue(lm.decode(tail).startswith("Which word"))
