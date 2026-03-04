"""
transformer_model.py
====================
Transformer architecture for the Axiomatic Deep Voting framework.
s
Drop into the repo root next to models.py. Provides:
  - VotingTransformer        (the nn.Module)
  - Transformer2logits       (List[Profile] → logits tensor, for axiom losses)
  - Transformer2rule         (plain decoding wrapper)
  - Transformer2rule_n       (neutrality-averaged decoding wrapper)

Architecture
------------
  ranking → one-hot position matrix (m_max × m_max) → flatten to m_max²
  → Linear → d_model token per voter
  → pad voters with learned PAD embedding + mask
  → nn.TransformerEncoder (NO positional encoding → anonymous by construction)
  → mean-pool over voters (excluding pad)
  → Linear → ReLU → Linear → m_max logits

References
----------
  Hornischer & Terzopoulou 2025 "Learning How to Vote with Principles" 
  Anil & Bao 2021 "Learning to Elect" (NeurIPS)
  Vaswani et al. 2017 "Attention Is All You Need"
  Lee et al. 2018 "Set Transformer"
"""

import itertools
from random import sample

import torch
from torch import nn

from utils import profile_to_onehot
from pref_voting.profiles import Profile


# ── Model ───────────────────────────────────────────────────────
class VotingTransformer(nn.Module):

    def __init__(
        self,
        max_num_voters,
        max_num_alternatives,
        d_model=128,
        nhead=4,
        num_layers=2,
        dim_feedforward=256,
        dropout=0.1,
    ):
        super().__init__()
        self.max_num_voters = max_num_voters
        self.max_num_alternatives = max_num_alternatives
        self.d_model = d_model

        input_dim = max_num_alternatives * max_num_alternatives
        self.input_projection = nn.Linear(input_dim, d_model)
        self.pad_embedding = nn.Parameter(torch.randn(d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers,
        )
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Linear(128, max_num_alternatives),
        )

    def forward(self, x, padding_mask=None):
        """
        x:  (batch, n_max, m_max²)   flattened one-hot position matrices
        padding_mask: (batch, n_max)  True = pad position
        returns: (batch, m_max) logits
        """
        B, N, _ = x.shape
        if padding_mask is None:
            padding_mask = (x.abs().sum(dim=-1) == 0)

        h = self.input_projection(x)
        pad = self.pad_embedding.unsqueeze(0).unsqueeze(0).expand(B, N, -1)
        m = padding_mask.unsqueeze(-1).float()
        h = h * (1.0 - m) + pad * m

        h = self.transformer_encoder(h, src_key_padding_mask=padding_mask)

        real = (~padding_mask).float().unsqueeze(-1)
        count = real.sum(dim=1).clamp(min=1)
        pooled = (h * real).sum(dim=1) / count

        return self.classifier(pooled)


# ── Encoding ────────────────────────────────────────────────────
def _profile_to_tensor(profile, max_num_voters, max_num_alternatives):
    """Single Profile → (n_max, m_max²) float tensor."""
    onehot = profile_to_onehot(profile, max_num_voters, max_num_alternatives)
    rows = []
    for ranking_oh in onehot:
        flat = []
        for vec in ranking_oh:
            flat.extend(vec)
        rows.append(flat)
    return torch.tensor(rows, dtype=torch.float)


# ── Transformer2logits  (used by axioms_continuous) ─────────────
def Transformer2logits(model, X):
    """List[Profile] → (len(X), m_max) logits.  Same interface as WEC2logits."""
    batch = torch.stack([
        _profile_to_tensor(p, model.max_num_voters, model.max_num_alternatives)
        for p in X
    ], dim=0)
    return model(batch)


# ── Plain decoding ──────────────────────────────────────────────
def Transformer2rule_prediction(model, profile, full=False):
    """Same interface as WEC2rule_prediction in models.py."""
    model.eval()
    with torch.no_grad():
        x = _profile_to_tensor(
            profile, model.max_num_voters, model.max_num_alternatives
        ).unsqueeze(0)
        logits = model(x)
        binary = torch.round(torch.sigmoid(logits)).squeeze()
        if full:
            return [i for i in range(len(binary)) if int(binary[i]) == 1]
        return [i for i in range(len(binary))
                if int(binary[i]) == 1 and i in profile.candidates]


# ── Neutrality-averaged decoding ────────────────────────────────
def Transformer2rule_prediction_n(
    model, profile, num_samples, full=False, print_sigmoids=False
):
    """Same interface as WEC2rule_prediction_n in models.py."""
    model.eval()
    with torch.no_grad():
        m = profile.num_cands
        m_max = model.max_num_alternatives

        profiles, perms = [], []
        if num_samples is None:                         # all m! permutations
            for p in itertools.permutations(range(m)):
                perms.append(tuple(list(p) + list(range(m, m_max))))
                profiles.append(
                    Profile([[p[a] for a in r] for r in profile.rankings])
                )
        else:                                           # sampled permutations
            for _ in range(num_samples):
                p = sample(list(range(m)), m)
                if p in perms:
                    break
                perms.append(tuple(p + list(range(m, m_max))))
                profiles.append(
                    Profile([[p[a] for a in r] for r in profile.rankings])
                )

        batch = torch.stack([
            _profile_to_tensor(pr, model.max_num_voters, m_max)
            for pr in profiles
        ], dim=0)
        logits = model(batch)

        # de-permute and average
        re = torch.zeros_like(logits)
        for j in range(len(logits)):
            re[j] = logits[j][perms[j],]
        prediction = re.mean(dim=0)

        sigs = torch.sigmoid(prediction)
        binary = torch.round(sigs).squeeze()

        if full:
            return [i for i in range(len(binary)) if int(binary[i]) == 1]
        if print_sigmoids:
            return {
                "winning_set": [i for i in range(len(binary))
                                if int(binary[i]) == 1 and i in profile.candidates],
                "sigmoids": sigs.squeeze().tolist(),
            }
        return [i for i in range(len(binary))
                if int(binary[i]) == 1 and i in profile.candidates]


# ── Lambda wrappers (same pattern as models.py) ────────────────
def Transformer2rule(model, full=False):
    """Returns a voting rule function: Profile → list of winner indices."""
    return lambda profile: Transformer2rule_prediction(model, profile, full)


def Transformer2rule_n(model, num_samples, full=False, print_sigmoids=False):
    """Returns a neut-averaged voting rule function."""
    return lambda profile: Transformer2rule_prediction_n(
        model, profile, num_samples, full, print_sigmoids
    )
