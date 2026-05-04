"""
Set Transformer architecture for voting rule synthesis — Majority Tournament Encoding.

Based on:
- Lee et al. (2019) "Set Transformer: A Framework for Attention-Based 
  Permutation-Invariant Neural Networks"
- Anil & Bao (2022) "Learning to Elect"
- Xiong et al. (2020) "On Layer Normalization in the Transformer Architecture"
  (pre-layer-norm configuration for stability)
- Hornischer & Terzopoulou (2025) "Learning How to Vote with Principles"
  (JAIR 2025) — weighted tournament encoding for pairwise majority margins

KEY CHANGE (v2 — Majority Tournament Encoding):
    Instead of feeding raw voter rankings as a (n_voters, m²) one-hot matrix,
    we first aggregate the profile into a Pairwise Majority Margin matrix M
    of shape (m, m), where M[i,j] = n_{i>j} - n_{j>i}.

    Benefits:
    1. Anonymity by construction — the margin matrix is invariant to voter order.
    2. Fixed input size — depends only on m, not n.
    3. Information-theoretically sufficient for all tournament solutions
       (Copeland, Stable Voting, Top Cycle, etc.).
    4. Dramatically smaller input: O(m²) vs O(n·m²).
"""

import math
import torch
from torch import nn
import torch.nn.functional as F


class MultiheadAttention(nn.Module):
    """Standard multi-head attention with pre-layer-norm."""

    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)

        q = self.W_q(Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k = self.W_k(K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v = self.W_v(V).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        context = torch.matmul(attn, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.W_o(context)


class MAB(nn.Module):
    """
    Multihead Attention Block (MAB) with pre-layer-norm.

    MAB(X, Y) = LayerNorm(H + rFF(H))
    where H = LayerNorm(X + Multihead(X, Y, Y))

    Using pre-LN per Xiong et al. (2020) for training stability.
    """

    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn = MultiheadAttention(d_model, n_heads, dropout)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, X, Y):
        X_norm = self.norm1(X)
        Y_norm = self.norm1(Y) if Y is not X else X_norm
        H = X + self.attn(X_norm, Y_norm, Y_norm)
        out = H + self.ff(self.norm2(H))
        return out


class SAB(nn.Module):
    """
    Set Attention Block: SAB(X) = MAB(X, X)
    Self-attention over set elements (alternatives in tournament encoding).
    Permutation-equivariant.
    """

    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.mab = MAB(d_model, n_heads, d_ff, dropout)

    def forward(self, X):
        return self.mab(X, X)


class PMA(nn.Module):
    """
    Pooling by Multihead Attention: PMA_k(Z) = MAB(S, rFF(Z))
    where S is a learned seed matrix of k seed vectors.
    Permutation-invariant.
    """

    def __init__(self, d_model, n_heads, d_ff, n_seeds, dropout=0.1):
        super().__init__()
        self.seeds = nn.Parameter(torch.randn(1, n_seeds, d_model))
        nn.init.xavier_uniform_(self.seeds)
        self.ff_pre = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
        )
        self.mab = MAB(d_model, n_heads, d_ff, dropout)

    def forward(self, Z):
        batch_size = Z.size(0)
        seeds = self.seeds.expand(batch_size, -1, -1)
        return self.mab(seeds, self.ff_pre(Z))


class SetTransformer(nn.Module):
    """
    Set Transformer for voting rule synthesis — Majority Tournament Encoding.

    Architecture (v2 — Tournament):
        Tournament Margin Matrix (m, m)
        -> Row projection to d_model: each row = one alternative's margin profile
        -> Encoder (2 SAB layers — fewer needed since input is pre-aggregated)
        -> Decoder (PMA with 1 seed + SAB)
        -> Output MLP -> logits over m alternatives

    Input: (batch, m, m) pairwise majority margin matrix.
    Output: (batch, m) logits.

    Anonymity is guaranteed by construction: the margin matrix M[i,j] = n_{i>j} - n_{j>i}
    is invariant to any permutation of voters.

    Args:
        max_num_voters: maximum number of voters (kept for interface compatibility)
        max_num_alternatives: number of alternatives (m)
        d_model: hidden dimension (default: 128)
        n_heads: number of attention heads (default: 8)
        d_ff: feedforward dimension (default: 256)
        n_enc_layers: number of SAB layers in encoder (default: 2)
        dropout: dropout rate (default: 0.1)
    """

    def __init__(
        self,
        max_num_voters,
        max_num_alternatives,
        d_model=128,
        n_heads=8,
        d_ff=256,
        n_enc_layers=2,     # Reduced from 4: input is pre-aggregated
        dropout=0.1,
    ):
        super().__init__()
        self.max_num_voters = max_num_voters
        self.max_num_alternatives = max_num_alternatives
        self.d_model = d_model

        # Input: each alternative's row in the margin matrix has m entries
        # Row i of M = [M[i,0], M[i,1], ..., M[i,m-1]]
        input_dim = max_num_alternatives
        self.input_proj = nn.Linear(input_dim, d_model)

        # Encoder: chain of SAB layers (self-attention over alternatives)
        self.encoder = nn.ModuleList([
            SAB(d_model, n_heads, d_ff, dropout) for _ in range(n_enc_layers)
        ])

        # Decoder: PMA (1 seed) + SAB
        self.pma = PMA(d_model, n_heads, d_ff, n_seeds=1, dropout=dropout)
        self.decoder_sab = SAB(d_model, n_heads, d_ff, dropout)

        # Output head
        self.output_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, max_num_alternatives),
        )

    def forward(self, x):
        """
        Args:
            x: tensor of shape (batch_size, m, m)
               Pairwise majority margin matrix.
               x[b, i, j] = (# voters preferring i over j) - (# preferring j over i)
        Returns:
            logits: tensor of shape (batch_size, max_num_alternatives)
        """
        # Each row = one alternative's margin profile against all others
        # Shape: (batch, m, m) -> project each row -> (batch, m, d_model)
        h = self.input_proj(x)

        # Encoder: self-attention over alternatives
        for sab in self.encoder:
            h = sab(h)

        # Decoder
        h = self.pma(h)            # (batch, 1, d_model)
        h = self.decoder_sab(h)    # (batch, 1, d_model)

        # Output
        h = h.squeeze(1)           # (batch, d_model)
        logits = self.output_head(h)  # (batch, m)
        return logits


# ============================================================
# MAJORITY TOURNAMENT ENCODING — Core Conversion Functions
# ============================================================

def profile_to_majority_margin(profile, max_num_alternatives):
    """
    Convert a pref_voting Profile to a Pairwise Majority Margin matrix.

    For each pair (i, j), compute:
        M[i, j] = n_{i>j} - n_{j>i}

    where n_{i>j} = number of voters who rank alternative i above j.

    This is the weighted tournament representation used by tournament
    solutions (Copeland, Stable Voting, Split Cycle, etc.).

    Returns: tensor of shape (m, m), where m = max_num_alternatives.
             For alternatives not in the profile, rows/cols are zero.
    """
    m = max_num_alternatives
    margin = [[0.0] * m for _ in range(m)]

    rankings = profile.rankings
    candidates = list(profile.candidates)

    for ranking in rankings:
        # For each pair in this voter's ranking, the higher-ranked
        # alternative gets +1 margin, the lower gets -1
        for pos_i in range(len(ranking)):
            for pos_j in range(pos_i + 1, len(ranking)):
                winner = ranking[pos_i]   # ranked higher
                loser = ranking[pos_j]    # ranked lower
                if winner < m and loser < m:
                    margin[winner][loser] += 1.0
                    margin[loser][winner] -= 1.0

    return torch.tensor(margin, dtype=torch.float32)


def profile_to_set_input(profile, max_num_voters, max_num_alternatives):
    """
    Convert a pref_voting Profile to a tournament margin tensor.

    This replaces the old one-hot (n_voters, m²) representation with
    the aggregated (1, m, m) majority margin matrix.

    Args:
        profile: pref_voting Profile
        max_num_voters: ignored (kept for interface compatibility)
        max_num_alternatives: m

    Returns: tensor of shape (1, m, m)
    """
    margin = profile_to_majority_margin(profile, max_num_alternatives)
    return margin.unsqueeze(0)  # (1, m, m)


def SetTransformer2logits(model, X):
    """
    Compute logits prediction of the Set Transformer on a list of profiles.

    Input: The model and a list of profiles X.
    Output: A tensor of logits predictions for each profile in X.

    Uses Majority Tournament Encoding: each profile is converted to its
    (m, m) pairwise majority margin matrix before being fed to the model.
    """
    m = model.max_num_alternatives
    batch_tensors = []

    for profile in X:
        margin = profile_to_majority_margin(profile, m)
        batch_tensors.append(margin)

    # Stack into (batch_size, m, m)
    batch = torch.stack(batch_tensors, dim=0)
    logits = model(batch)
    return logits


def SetTransformer2rule_prediction(model, profile, full=False):
    """
    Takes a SetTransformer model and a profile and outputs the winners.
    """
    model.eval()
    with torch.no_grad():
        tensor = profile_to_set_input(
            profile, model.max_num_voters, model.max_num_alternatives
        )
        logits = model(tensor)
        binary = torch.round(torch.sigmoid(logits)).squeeze()
        if not full:
            return [i for i in range(len(binary))
                    if int(binary[i]) == 1 and i in profile.candidates]
        else:
            return [i for i in range(len(binary)) if int(binary[i]) == 1]


def SetTransformer2rule(model, full=False):
    """Returns a voting rule function from the model."""
    return lambda profile: SetTransformer2rule_prediction(model, profile, full)


def SetTransformer2rule_prediction_n(model, profile, num_samples, full=False):
    """
    Neutrality-averaged prediction.

    Generates all (or sampled) permutations of alternatives, computes
    the model output on each permuted profile, de-permutes, and averages.

    With tournament encoding, each permuted profile produces a permuted
    margin matrix. The model output is de-permuted and averaged.
    """
    import itertools
    from random import sample
    from pref_voting.profiles import Profile

    model.eval()
    with torch.no_grad():
        num_alts = profile.num_cands
        m = model.max_num_alternatives

        # Generate permutations
        profiles_perm = []
        permutations = []

        if num_samples is None:
            perm_list = list(itertools.permutations(range(num_alts)))
        else:
            perm_list = []
            seen = set()
            for _ in range(num_samples):
                p = tuple(sample(list(range(num_alts)), num_alts))
                if p not in seen:
                    seen.add(p)
                    perm_list.append(p)

        for p in perm_list:
            p_max = list(p) + list(range(len(p), m))
            permutations.append(tuple(p_max))
            permuted_rankings = [
                [p[alt] for alt in ranking] for ranking in profile.rankings
            ]
            profiles_perm.append(Profile(permuted_rankings))

        # Batch compute logits using tournament encoding
        batch_tensors = []
        for perm_profile in profiles_perm:
            margin = profile_to_majority_margin(perm_profile, m)
            batch_tensors.append(margin)

        batch = torch.stack(batch_tensors, dim=0)  # (num_perms, m, m)
        logits = model(batch)

        # De-permute and average
        re_permuted = torch.zeros_like(logits)
        for j in range(len(logits)):
            re_permuted[j] = logits[j][permutations[j],]
        prediction = re_permuted.mean(dim=0)

        # Binary decision
        binary = torch.round(torch.sigmoid(prediction)).squeeze()
        if not full:
            return [i for i in range(len(binary))
                    if int(binary[i]) == 1 and i in profile.candidates]
        else:
            return [i for i in range(len(binary)) if int(binary[i]) == 1]


def SetTransformer2rule_n(model, num_samples, full=False):
    """Returns a neutrality-averaged voting rule function."""
    return lambda profile: SetTransformer2rule_prediction_n(
        model, profile, num_samples, full
    )