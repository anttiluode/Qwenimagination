"""Hold a derived thought outside the token stream and read it back later.

Phase H (hold): the model is asked to silently think of something it must
derive ("the capital city of France"). We read its J-space at every layer
over its short reply. The answer word is never written anywhere.

Phase D (recall): a NEW context. The model reads an unrelated paragraph and
is asked what it was thinking of earlier. The hold conversation is not in
this context, so attention cannot look it up. Whatever it reports must
arrive through the residual stream from outside, by one of these routes:

  none        nothing injected
  in_context  the hold conversation IS in the context (ceiling: attention)
  oracle      the J-lens vector of the answer word, written open-loop
              (cheating: tells us how well the write channel works at all)
  cue         the J-lens vector of the cue word ("France"), open-loop
              (does carrying the cue suffice, or must the thought be carried?)
  snapshot    the raw hidden state captured at hold time, open-loop
  jstate_open the model's own top-K J-space concepts from the hold, written
              at a constant strength at every position
  jstate_closed the same concepts, but held by a residue loop: after each
              token the J-space downstream is read, the deficit
              r = m - read is integrated into a per-concept drive a, and
              only that drive is written at the next token
  jstate_closed_surprise  as closed, but the write is turned down when the
              model's own next-token surprise is high (the world is talking)

Every route is swept over a fixed strength grid. Recall is forced choice
over all item answers. Damage is the mean KL between the model's next-token
distributions with and without the injection, over the recall context.
"""
import math
import torch
import torch.nn.functional as F

from . import items as I
from .lens import tap_layers, Injector

OPEN_GRID = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
CLOSED_GRID = [0.05, 0.1, 0.2, 0.4, 0.8]
SNAP_GRID = [0.25, 0.5, 1.0, 2.0, 4.0]
K_STATE = 8          # concepts in the held state
A_MAX = 16.0         # closed-loop drive cap, in units of each concept's held value
SURPRISE_BETA = 0.5  # closed_surprise gate: exp(-beta * max(0, s - median s))
INJECT_OFFSET = 4    # write layer = read layer - 4


class Setup:
    def __init__(self, lm, lens):
        self.lm, self.lens = lm, lens
        self.items, self.dropped = [], []
        for desc, tgt, cue in I.ITEMS:
            ok = all(w in lens.index and lm.single_token(w) is not None for w in (tgt, cue))
            (self.items if ok else self.dropped).append((desc, tgt, cue))
        self.finalize()

    def finalize(self):
        """Forced choice is over every answer AND every cue word, so naming
        the cue ("France") instead of the thought ("Paris") counts as wrong."""
        words = []
        for _, t, c in self.items:
            for w in (t, c):
                if w not in words:
                    words.append(w)
        self.choice_words = words
        self.choice_ids = torch.tensor([self.lm.single_token(w) for w in words],
                                       device=self.lm.device)
        self.answer_pos = [words.index(t) for _, t, _ in self.items]

    def split(self, i):
        return "dev" if i % 3 == 0 else "test"


# ---------------------------------------------------------------- phase H
@torch.no_grad()
def hold_read(S, prompt):
    lm, lens = S.lm, S.lens
    ids = lm.chat([("user", prompt)], prefix=I.HOLD_REPLY)
    n0 = len(lm.chat([("user", prompt)], prefix=""))
    x = torch.tensor([ids], device=lm.device)
    Z, H = [], []
    with tap_layers(lm, range(lm.n_layers)) as st:
        lm.final_hidden(x)
        for l in range(lm.n_layers):
            h = st[l][0, n0:].float()            # reply positions
            Z.append(lens.read(h, l).mean(0))
            H.append(h.mean(0))
    return torch.stack(Z), torch.stack(H)        # [L, C], [L, d]


def hold_all(S):
    holds = [hold_read(S, I.HOLD_PROMPT.format(desc=d)) for d, _, _ in S.items]
    null = hold_read(S, I.NULL_HOLD_PROMPT)
    return holds, null


def g0_table(S, holds):
    """Per item and layer: rank of the answer among all concepts in the
    hold reading (0 = top), and the answer's z in its own hold minus its
    mean z in the other items' holds (specificity)."""
    idx = torch.tensor([S.lens.index[t] for _, t, _ in S.items])
    Z = torch.stack([z for z, _ in holds])                     # [N, L, C]
    N, L, C = Z.shape
    own = Z[torch.arange(N), :, idx]                           # [N, L]
    rank = (Z > own[:, :, None]).sum(-1)                       # [N, L]
    tz = Z[:, :, idx]                                          # [N(hold), L, N(target)]
    others = (tz.sum(0) - own.T) / (N - 1)                     # [L, N]
    spec = own - others.T                                      # [N, L]
    return rank.cpu(), spec.cpu()


def choose_layers(S, rank):
    L = S.lm.n_layers
    lo, hi = L // 3, (5 * L) // 6
    dev = [i for i in range(len(S.items)) if S.split(i) == "dev"]
    best = None
    for l in range(lo, hi + 1):
        hit = (rank[dev, l] < 10).float().mean().item()
        mr = rank[dev, l].float().median().item()
        key = (hit, -mr)
        if best is None or key > best[0]:
            best = (key, l)
    l_read = best[1]
    return l_read, max(1, l_read - INJECT_OFFSET)


# ---------------------------------------------------------------- phase D
def recall_ids(S, i, in_context=False):
    desc, _, _ = S.items[i]
    filler = I.FILLERS[i % len(I.FILLERS)]
    msgs = [("user", filler + I.ASK)]
    if in_context:
        msgs = [("user", I.HOLD_PROMPT.format(desc=desc)), ("assistant", I.HOLD_REPLY)] + msgs
    return S.lm.chat(msgs, prefix=I.ANSWER_PREFIX)


def score(S, i, logits_last):
    lp = F.log_softmax(logits_last.float()[S.choice_ids], -1)
    j = S.answer_pos[i]
    order = lp.argsort(descending=True)
    rank = (order == j).nonzero().item()
    return {"correct": int(rank == 0), "rank": rank, "lp": lp[j].item(),
            "said": S.choice_words[order[0].item()]}


@torch.no_grad()
def run_open(S, ids, l_inj, delta):
    inj = Injector(S.lm, l_inj)
    inj.delta = delta
    try:
        logits = S.lm.logits(torch.tensor([ids], device=S.lm.device))[0].float()
    finally:
        inj.remove()
    return F.log_softmax(logits, -1)


@torch.no_grad()
def run_closed(S, ids, l_inj, l_read, K, m, gamma, beta=None, s_med=None):
    lm, lens = S.lm, S.lens
    inj = Injector(lm, l_inj)
    a = torch.zeros_like(m)
    amax = A_MAX * m
    out, cache, energy = [], None, 0.0
    try:
        with tap_layers(lm, [l_read]) as st:
            for q in range(len(ids)):
                x = torch.tensor([[ids[q]]], device=lm.device)
                o = lm.model(input_ids=x, past_key_values=cache, use_cache=True)
                cache = o.past_key_values
                lp = F.log_softmax(o.logits[0, -1].float(), -1)
                out.append(lp)
                z = lens.read(st[l_read][0, -1], l_read, K)
                a = torch.minimum(torch.clamp(a + gamma * (m - z), min=0.0), amax)
                inj.delta = lens.write(l_inj, K, a)
                inj.gate = 1.0
                if beta is not None and q + 1 < len(ids):
                    s = -lp[ids[q + 1]].item()
                    inj.gate = math.exp(-beta * max(0.0, s - s_med))
                energy += (inj.gate * inj.delta.norm()).item()
    finally:
        inj.remove()
    return torch.stack(out), energy / len(ids)


def kl(lp_ref, lp):
    """Mean per-position KL(ref || cond), all positions but the answer slot."""
    return (lp_ref[:-1].exp() * (lp_ref[:-1] - lp[:-1])).sum(-1).mean().item()


def recall_item(S, i, holds, l_read, l_inj, log=print):
    lm, lens = S.lm, S.lens
    desc, tgt, cue = S.items[i]
    Z, H = holds[i]
    ids = recall_ids(S, i)
    rows = []

    def add(cond, g, lp, energy):
        r = score(S, i, lp[-1])
        r.update(item=i, split=S.split(i), cond=cond, strength=g,
                 damage=kl(lp_none, lp), energy=energy)
        rows.append(r)

    lp_none = run_open(S, ids, l_inj, None)
    add("none", 0.0, lp_none, 0.0)
    lp_ic = run_open(S, recall_ids(S, i, in_context=True), l_inj, None)
    r = score(S, i, lp_ic[-1])
    r.update(item=i, split=S.split(i), cond="in_context", strength=0.0,
             damage=float("nan"), energy=0.0)
    rows.append(r)

    zr = Z[l_read]
    K = zr.topk(K_STATE).indices
    m = zr[K]
    m_ref = m.max()
    held = [lens.words[k] for k in K.tolist()]

    for g in OPEN_GRID:
        for cond, KK, uu in (("oracle", [lens.index[tgt]], m_ref[None]),
                             ("cue", [lens.index[cue]], m_ref[None]),
                             ("jstate_open", K, m)):
            KK = torch.as_tensor(KK, device=lm.device)
            d = lens.write(l_inj, KK, g * uu)
            add(cond, g, run_open(S, ids, l_inj, d), d.norm().item())
    for g in SNAP_GRID:
        d = g * (H[l_inj] - lens.hmu[l_inj])
        add("snapshot", g, run_open(S, ids, l_inj, d), d.norm().item())

    s_none = torch.tensor([-lp_none[q, ids[q + 1]].item() for q in range(len(ids) - 1)])
    s_med = s_none.median().item()
    for g in CLOSED_GRID:
        lp, e = run_closed(S, ids, l_inj, l_read, K, m, g)
        add("jstate_closed", g, lp, e)
        lp, e = run_closed(S, ids, l_inj, l_read, K, m, g, SURPRISE_BETA, s_med)
        add("jstate_closed_surprise", g, lp, e)
    return rows, held
