"""Jacobian lens over a concept vocabulary, plus read and write in its coordinates.

For concept token t and layer l, the lens vector is the average gradient of
t's logit, summed over the current and all later positions, with respect to
the residual stream after layer l:

    j[l, t] = E_q [ d( sum_{p >= q} z_t(p) ) / d h_l(q) ]

averaged over every position q of a calibration corpus. This is the
"now or in the future" averaged-Jacobian construction of Gurnee et al.
(2026), using raw logits. Causality makes the p >= q restriction automatic.

Reading:  z_l(h)[t] = (h . j[l,t] - mu[l,t]) / sd[l,t]   (z-scored on the corpus)
Writing:  the minimum-norm change to h that raises the chosen concepts'
          z-reads at layer l by u:   dh = J_K^T (J_K J_K^T)^-1 (sd_K * u)
"""
import contextlib
import torch


def _hs(out):
    return out[0] if isinstance(out, tuple) else out


def _replace(out, new):
    if isinstance(out, tuple):
        return (new,) + tuple(out[1:])
    return new


@contextlib.contextmanager
def tap_layers(lm, layers, retain_grad=False):
    """Record the residual stream after each layer in `layers`."""
    store, hooks = {}, []
    for l in layers:
        def hook(mod, inp, out, l=l):
            h = _hs(out)
            if retain_grad:
                h.retain_grad()
            store[l] = h
        hooks.append(lm.layers[l].register_forward_hook(hook))
    try:
        yield store
    finally:
        for h in hooks:
            h.remove()


class Injector:
    """Adds `self.delta` (a [d] vector, or [T, d]) to the residual stream
    after one layer. Set delta=None for no injection."""

    def __init__(self, lm, layer):
        self.delta = None
        self.gate = 1.0
        self.h = lm.layers[layer].register_forward_hook(self._hook)

    def _hook(self, mod, inp, out):
        if self.delta is None:
            return out
        h = _hs(out)
        return _replace(out, h + self.gate * self.delta.to(h.dtype))

    def remove(self):
        self.h.remove()


def compute_jlens(lm, concept_ids, texts, chunk=128, batch=8, log=print):
    L, d, C = lm.n_layers, lm.d, len(concept_ids)
    dev = lm.device
    for p in lm.model.parameters():
        p.requires_grad_(False)
    J = torch.zeros(L, C, d, dtype=torch.float32, device=dev)
    emb_layer = lm.model.get_input_embeddings()
    W = lm.unembed.detach().float()
    cid = torch.tensor(concept_ids, device=dev)
    n_pos = 0
    chunks = []
    for t in texts:
        ids = lm.encode(t)
        for i in range(0, len(ids), chunk):
            if len(ids[i:i + chunk]) >= 8:
                chunks.append(ids[i:i + chunk])
    for ci, ids in enumerate(chunks):
        x = torch.tensor([ids], device=dev)
        T = x.shape[1]
        for b0 in range(0, C, batch):
            tids = cid[b0:b0 + batch]
            B = len(tids)
            emb = emb_layer(x.repeat(B, 1)).detach().requires_grad_(True)
            with tap_layers(lm, range(L), retain_grad=True) as st:
                h = lm.model.model(inputs_embeds=emb).last_hidden_state.float()
                z = (h * W[tids][:, None, :]).sum(-1)       # [B, T]
                z.sum().backward()
                for l in range(L):
                    J[l, b0:b0 + B] += st[l].grad.float().sum(1)
        n_pos += T
        log(f"  lens chunk {ci + 1}/{len(chunks)} ({T} tokens)")
    J /= n_pos
    return J


@torch.no_grad()
def reading_stats(lm, J, texts):
    """Mean/sd of every lens read, and mean hidden state, per layer."""
    L = lm.n_layers
    reads = [[] for _ in range(L)]
    hs = [[] for _ in range(L)]
    for t in texts:
        x = torch.tensor([lm.encode(t)], device=lm.device)
        with tap_layers(lm, range(L)) as st:
            lm.final_hidden(x)
            for l in range(L):
                h = st[l][0].float()
                hs[l].append(h)
                reads[l].append(h @ J[l].T)
    mu = torch.stack([torch.cat(r).mean(0) for r in reads])
    sd = torch.stack([torch.cat(r).std(0) for r in reads]).clamp_min(1e-6)
    hmu = torch.stack([torch.cat(h).mean(0) for h in hs])
    return mu, sd, hmu


class Lens:
    def __init__(self, J, mu, sd, hmu, words):
        self.J, self.mu, self.sd, self.hmu, self.words = J, mu, sd, hmu, words
        self.index = {w: i for i, w in enumerate(words)}

    def read(self, h, l, K=None):
        """z-scored reads of hidden state(s) h at layer l, for concepts K (all if None)."""
        J, mu, sd = self.J[l], self.mu[l], self.sd[l]
        if K is not None:
            J, mu, sd = J[K], mu[K], sd[K]
        return (h.float() @ J.T - mu) / sd

    def write(self, l, K, u, ridge=1e-4):
        """Minimum-norm residual change raising z-reads of concepts K by u."""
        JK = self.J[l][K]                                    # [k, d]
        G = JK @ JK.T
        G = G + ridge * G.diagonal().mean() * torch.eye(len(K), device=G.device)
        c = torch.linalg.solve(G, self.sd[l][K] * u)
        return JK.T @ c                                      # [d]

    def save(self, path):
        torch.save({"J": self.J.cpu(), "mu": self.mu.cpu(), "sd": self.sd.cpu(),
                    "hmu": self.hmu.cpu(), "words": self.words}, path)

    @classmethod
    def load(cls, path, device):
        s = torch.load(path, map_location=device)
        return cls(s["J"].to(device), s["mu"].to(device), s["sd"].to(device),
                   s["hmu"].to(device), s["words"])
