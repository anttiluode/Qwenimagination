"""Model access: a real Qwen through Hugging Face, or a random Qwen-shaped
model with a stub tokenizer for offline plumbing tests.

Everything else in the package talks to the model only through `Lm`:
  lm.chat(messages, prefix) -> token ids of a chat ending in an assistant prefix
  lm.single_token(word)     -> id of " word" if it is one token, else None
  lm.layers                 -> the decoder layers (for hooks)
  lm.final_hidden(ids)      -> last, normed hidden states
  lm.unembed                -> [vocab, d] unembedding matrix
"""
import re
import torch


class Lm:
    def __init__(self, model, tok, device, name):
        self.model = model.eval()
        self.tok = tok
        self.device = device
        self.name = name
        self.layers = model.model.layers
        self.n_layers = len(self.layers)
        self.d = model.config.hidden_size
        self.unembed = model.get_output_embeddings().weight  # [V, d]

    # ---- text ---------------------------------------------------------
    def chat(self, messages, prefix=""):
        """messages: list of (role, text). Ends with an open assistant turn
        that already contains `prefix`."""
        return self.tok(self.chat_text(messages, prefix), add_special_tokens=False)["input_ids"]

    def chat_text(self, messages, prefix=""):
        msgs = [{"role": r, "content": t} for r, t in messages]
        try:
            text = self.tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True,
                enable_thinking=False)
        except TypeError:
            text = self.tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True)
        return text + prefix

    def encode(self, text):
        return self.tok(text, add_special_tokens=False)["input_ids"]

    def single_token(self, word):
        ids = self.encode(" " + word)
        return ids[0] if len(ids) == 1 else None

    def decode(self, ids):
        return self.tok.decode(ids)

    # ---- forward ------------------------------------------------------
    def final_hidden(self, ids, **kw):
        """Last-layer hidden states after the final norm: [B, T, d]."""
        return self.model.model(input_ids=ids, **kw).last_hidden_state

    def logits(self, ids, **kw):
        return self.model(input_ids=ids, **kw).logits


def load_hf(name, device, dtype=torch.float32):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    try:
        model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype)
    except TypeError:                       # transformers < 4.56
        model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype)
    model = model.to(device)
    return Lm(model, tok, device, name)


# ---------------------------------------------------------------------------
# Offline stand-in: random weights, word-level tokenizer. Tests plumbing only.
# ---------------------------------------------------------------------------
class StubTokenizer:
    SPECIAL = ["<|im_start|>", "<|im_end|>", "<pad>", "<unk>"]
    PAT = re.compile(r"<\|im_start\|>|<\|im_end\|>| ?[A-Za-z']+| ?\d+| ?[^\sA-Za-z\d]|\n")

    def __init__(self, texts):
        vocab = list(self.SPECIAL)
        seen = set(vocab)
        for t in texts:
            for p in self.PAT.findall(t):
                for q in (p, " " + p.strip()):
                    if q not in seen:
                        seen.add(q)
                        vocab.append(q)
        self.vocab = vocab
        self.idx = {w: i for i, w in enumerate(vocab)}

    def __call__(self, text, add_special_tokens=False):
        ids = [self.idx.get(p, self.idx["<unk>"]) for p in self.PAT.findall(text)]
        return {"input_ids": ids}

    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True, **kw):
        s = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in msgs)
        return s + ("<|im_start|>assistant\n" if add_generation_prompt else "")

    def decode(self, ids):
        return "".join(self.vocab[i] for i in ids)


def load_stub(device, seed=0, layers=8, d=64):
    from transformers import Qwen2Config, Qwen2ForCausalLM
    from . import items as I
    from .items_v2 import ITEMS_V2
    from .items_v3 import ITEMS_V3
    words = [w for it in ITEMS_V2 + ITEMS_V3 for w in it] + ['Read this phrase: "', '". Okay, I have read it.'] + [d_ for d_, _, _ in I.ITEMS] + [t for _, t, _ in I.ITEMS] + \
        [c for _, _, c in I.ITEMS] + I.DECOYS + I.FILLERS + I.CALIBRATION + \
        [I.HOLD_PROMPT, I.HOLD_REPLY, I.NULL_HOLD_PROMPT, I.ASK, I.ANSWER_PREFIX,
         "user assistant system"]
    tok = StubTokenizer(words)
    torch.manual_seed(seed)
    cfg = Qwen2Config(vocab_size=len(tok.vocab), hidden_size=d, intermediate_size=2 * d,
                      num_hidden_layers=layers, num_attention_heads=4,
                      num_key_value_heads=2, max_position_embeddings=1024,
                      tie_word_embeddings=True)
    model = Qwen2ForCausalLM(cfg).to(device)
    return Lm(model, tok, device, f"stub-random-L{layers}-d{d}")


def load(name, device, dtype=torch.float32):
    if name == "stub":
        return load_stub(device)
    return load_hf(name, device, dtype)
