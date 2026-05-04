"""
Set Transformer v3 for voting rule synthesis.

Key upgrades over v2:
1. Pairwise margin injection — explicit aggregate pairwise majority structure
   broadcast-added to every voter representation before encoder processing.
   This provides an inductive bias toward the Independence axiom.
2. Deeper encoder: 6 layers (SAB + ISAB×4 + SAB) vs v2's 4 layers
3. Increased inducing points: 32 vs v2's 16 — captures 10 pairwise margins
   with ~3 inducing points per pair + interaction effects
4. Expanded feedforward dimension: d_ff=512 vs v2's 256
5. All v2 features retained: dual-pathway input, GELU, multi-seed PMA,
   residual output head, pre-LayerNorm

Based on:
- Lee et al. (2019) "Set Transformer" — ISAB, PMA
- Anil & Bao (2022) "Learning to Elect" — voting rule synthesis
- Hornischer & Terzopoulou (2025) "Learning How to Vote with Principles"
- Xiong et al. (2020) "On Layer Normalization in the Transformer Architecture"
- Hendrycks & Gimpel (2016) "Gaussian Error Linear Units" — GELU
- Vaswani et al. (2017) "Attention Is All You Need" — multi-head attention

Optimized for Apple M2 (8-core, 16GB RAM).
"""

import math
import torch
from torch import nn
import torch.nn.functional as F


# ============================================================
# Core Attention Components
# ============================================================

class MultiheadAttention(nn.Module):
    """Standard multi-head attention."""

    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
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
    Multihead Attention Block with pre-layer-norm and GELU.

    MAB(X, Y) = LayerNorm(H + FF(H))
    where H = LayerNorm(X + MHA(X, Y, Y))
    """

    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn = MultiheadAttention(d_model, n_heads, dropout)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
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
    """Set Attention Block: SAB(X) = MAB(X, X). Permutation-equivariant."""

    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.mab = MAB(d_model, n_heads, d_ff, dropout)

    def forward(self, X):
        return self.mab(X, X)


class ISAB(nn.Module):
    """
    Induced Set Attention Block from Lee et al. (2019).

    Uses m_ind inducing points to reduce complexity from O(n^2) to O(n*m_ind).
    Critical for scaling to 55 voters.

    ISAB_m(X) = MAB(X, H) where H = MAB(I, X)
    I = learned inducing points of shape (m_ind, d_model)
    """

    def __init__(self, d_model, n_heads, d_ff, n_inducing, dropout=0.1):
        super().__init__()
        self.inducing = nn.Parameter(torch.randn(1, n_inducing, d_model))
        nn.init.xavier_uniform_(self.inducing)
        self.mab1 = MAB(d_model, n_heads, d_ff, dropout)
        self.mab2 = MAB(d_model, n_heads, d_ff, dropout)

    def forward(self, X):
        batch_size = X.size(0)
        I = self.inducing.expand(batch_size, -1, -1)
        H = self.mab1(I, X)       # (batch, m_ind, d_model)
        return self.mab2(X, H)    # (batch, n_voters, d_model)


class PMA(nn.Module):
    """
    Pooling by Multihead Attention with pre-FF.
    PMA_k(Z) = MAB(S, FF(Z)), S = learned seeds.
    """

    def __init__(self, d_model, n_heads, d_ff, n_seeds, dropout=0.1):
        super().__init__()
        self.seeds = nn.Parameter(torch.randn(1, n_seeds, d_model))
        nn.init.xavier_uniform_(self.seeds)
        self.ff_pre = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )
        self.mab = MAB(d_model, n_heads, d_ff, dropout)

    def forward(self, Z):
        batch_size = Z.size(0)
        seeds = self.seeds.expand(batch_size, -1, -1)
        return self.mab(seeds, self.ff_pre(Z))


# ============================================================
# Set Transformer v3
# ============================================================

class SetTransformerV3(nn.Module):
    """
    Set Transformer v3 with pairwise margin injection.

    Key differences from v2:
    1. Pairwise margin injection: computes aggregate M[a,b] = mean_i C_i(a,b)
       and projects + broadcasts to all voter representations
    2. Deeper encoder: 6 layers (SAB + ISAB×4 + SAB)
    3. More inducing points: 32 (vs 16)
    4. Wider feedforward: d_ff=512 (vs 256)

    Architecture:
        [one-hot → proj_A] + [pairwise → proj_B] → Gated Fusion
        + Margin Injection (aggregate pairwise majority → broadcast)
        → SAB → ISAB(32) × 4 → SAB  [6 encoder layers]
        → PMA(m seeds) → SAB  [decoder]
        → Residual Output Head → logits ŷ ∈ ℝᵐ
    """

    def __init__(
        self,
        max_num_voters,
        max_num_alternatives,
        d_model=128,
        n_heads=8,
        d_ff=512,
        n_enc_layers=6,
        n_inducing=32,
        dropout=0.1,
    ):
        super().__init__()
        self.max_num_voters = max_num_voters
        self.max_num_alternatives = max_num_alternatives
        self.d_model = d_model
        m = max_num_alternatives

        # === Dual-Pathway Input Encoding ===
        # Path A: one-hot ranking features (m*m dims)
        input_dim_onehot = m * m
        self.proj_onehot = nn.Sequential(
            nn.Linear(input_dim_onehot, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Path B: pairwise comparison features (m*m dims)
        input_dim_pairwise = m * m
        self.proj_pairwise = nn.Sequential(
            nn.Linear(input_dim_pairwise, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Fusion gate: learned weighting of the two pathways
        self.fusion_gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid(),
        )
        self.fusion_proj = nn.Linear(d_model * 2, d_model)

        # === NEW in v3: Pairwise Margin Injection ===
        # Projects the aggregate m×m margin matrix into d_model,
        # broadcast-added to all voter representations.
        # This gives the encoder explicit access to pairwise majority
        # structure from the very first layer.
        self.margin_proj = nn.Sequential(
            nn.Linear(m * m, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        # === Deeper Encoder: SAB + ISAB×(n_enc_layers-2) + SAB ===
        # First layer: full self-attention for initial voter-voter interactions
        # Middle layers: ISAB for O(n * m_ind) efficiency
        # Last layer: full self-attention for final refinement
        encoder_layers = []
        for i in range(n_enc_layers):
            if i == 0 or i == n_enc_layers - 1:
                encoder_layers.append(SAB(d_model, n_heads, d_ff, dropout))
            else:
                encoder_layers.append(ISAB(d_model, n_heads, d_ff, n_inducing, dropout))
        self.encoder = nn.ModuleList(encoder_layers)

        # === Multi-Seed PMA Decoder ===
        # m seeds: one per alternative → richer per-alternative representations
        self.pma = PMA(d_model, n_heads, d_ff, n_seeds=m, dropout=dropout)
        self.decoder_sab = SAB(d_model, n_heads, d_ff, dropout)

        # === Residual Output Head ===
        self.output_norm = nn.LayerNorm(d_model)
        self.output_ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.output_norm2 = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, 1)  # per-alternative logit

    def forward(self, x, x_pairwise=None):
        """
        Args:
            x: tensor (batch, max_num_voters, m*m) — one-hot rankings
            x_pairwise: tensor (batch, max_num_voters, m*m) — pairwise comparisons
                         If None, computed from x on-the-fly.
        Returns:
            logits: tensor (batch, max_num_alternatives)
        """
        m = self.max_num_alternatives

        # Compute pairwise features if not provided
        if x_pairwise is None:
            x_pairwise = self._onehot_to_pairwise(x, m)

        # === Dual-Pathway Encoding ===
        h_onehot = self.proj_onehot(x)              # (batch, n_voters, d_model)
        h_pairwise = self.proj_pairwise(x_pairwise)  # (batch, n_voters, d_model)

        # Gated fusion
        combined = torch.cat([h_onehot, h_pairwise], dim=-1)  # (batch, n, 2*d_model)
        gate = self.fusion_gate(combined)                       # (batch, n, d_model)
        h_proj = self.fusion_proj(combined)                     # (batch, n, d_model)
        h = gate * h_onehot + (1 - gate) * h_proj

        # === Pairwise Margin Injection (NEW in v3) ===
        # Compute aggregate pairwise margins: M[a,b] = mean over active voters
        # x_pairwise: (batch, n_voters, m*m)
        voter_norms = x_pairwise.norm(dim=-1, keepdim=True)  # (batch, n, 1)
        active_mask = (voter_norms > 1e-6).float()             # (batch, n, 1)
        n_active = active_mask.sum(dim=1, keepdim=True).clamp(min=1)  # (batch, 1, 1)

        # Average pairwise comparisons across active (non-padding) voters
        margin = (x_pairwise * active_mask).sum(dim=1) / n_active.squeeze(-1)  # (batch, m*m)

        # Project margin matrix and broadcast-add to all voter representations
        margin_emb = self.margin_proj(margin)  # (batch, d_model)
        h = h + margin_emb.unsqueeze(1)        # broadcast to (batch, n_voters, d_model)

        # === Encoder ===
        for layer in self.encoder:
            h = layer(h)

        # === Decoder ===
        h = self.pma(h)             # (batch, m, d_model) — m seeds
        h = self.decoder_sab(h)     # (batch, m, d_model)

        # === Residual Output Head ===
        h_norm = self.output_norm(h)
        h_res = h + self.output_ff(h_norm)      # residual connection
        h_res = self.output_norm2(h_res)
        logits = self.output_proj(h_res).squeeze(-1)  # (batch, m)

        return logits

    @staticmethod
    def _onehot_to_pairwise(x_onehot, m):
        """
        Convert one-hot ranking tensor to pairwise comparison matrix.

        From one-hot: position matrix P where P[pos, alt] = 1
        To pairwise: C[i, j] = 1 if alt i is ranked above alt j

        This is computed differentiably for gradient flow.
        """
        batch_size, n_voters, _ = x_onehot.shape

        # Reshape to (batch, n_voters, m, m) — position × alternative
        P = x_onehot.view(batch_size, n_voters, m, m)

        # For each voter, compute the rank of each alternative
        # rank[alt] = position where alt appears (lower = better)
        positions = torch.arange(m, device=x_onehot.device).float()
        # ranks shape: (batch, n_voters, m)
        ranks = torch.einsum('bnpa,p->bna', P, positions)

        # Pairwise comparison: C[i,j] = 1 if rank[i] < rank[j]
        # Use sigmoid approximation for differentiability
        rank_diff = ranks.unsqueeze(-1) - ranks.unsqueeze(-2)  # (batch, n, m, m)
        # C[i,j] ≈ σ(-α * (rank_i - rank_j)), α controls sharpness
        C = torch.sigmoid(-5.0 * rank_diff)

        # Flatten to (batch, n_voters, m*m)
        C_flat = C.view(batch_size, n_voters, m * m)
        return C_flat


# ============================================================
# Helper functions — interface with existing codebase
# ============================================================

def profile_to_set_input_v3(profile, max_num_voters, max_num_alternatives):
    """
    Convert profile to both one-hot and pairwise tensors.
    Returns: (x_onehot, x_pairwise) each of shape (1, max_num_voters, m*m)
    """
    m = max_num_alternatives
    voter_onehot = []
    voter_pairwise = []

    for ranking in profile.rankings:
        # One-hot encoding
        onehot = [0.0] * (m * m)
        for pos, alt in enumerate(ranking):
            if pos < m and alt < m:
                onehot[pos * m + alt] = 1.0
        voter_onehot.append(onehot)

        # Pairwise comparison matrix
        pairwise = [0.0] * (m * m)
        for i in range(min(len(ranking), m)):
            for j in range(i + 1, min(len(ranking), m)):
                alt_i = ranking[i]
                alt_j = ranking[j]
                if alt_i < m and alt_j < m:
                    pairwise[alt_i * m + alt_j] = 1.0  # alt_i preferred over alt_j
        voter_pairwise.append(pairwise)

    # Pad to max_num_voters
    for _ in range(len(voter_onehot), max_num_voters):
        voter_onehot.append([0.0] * (m * m))
        voter_pairwise.append([0.0] * (m * m))

    t_onehot = torch.tensor([voter_onehot], dtype=torch.float32)
    t_pairwise = torch.tensor([voter_pairwise], dtype=torch.float32)
    return t_onehot, t_pairwise


def SetTransformerV3_2logits(model, X):
    """
    Compute logits for a list of profiles. Compatible with axioms_continuous.py.
    """
    m = model.max_num_alternatives
    n = model.max_num_voters
    batch_onehot = []
    batch_pairwise = []

    for profile in X:
        voter_oh = []
        voter_pw = []
        for ranking in profile.rankings:
            onehot = [0.0] * (m * m)
            pairwise = [0.0] * (m * m)
            for pos, alt in enumerate(ranking):
                if pos < m and alt < m:
                    onehot[pos * m + alt] = 1.0
            for i in range(min(len(ranking), m)):
                for j in range(i + 1, min(len(ranking), m)):
                    alt_i = ranking[i]
                    alt_j = ranking[j]
                    if alt_i < m and alt_j < m:
                        pairwise[alt_i * m + alt_j] = 1.0
            voter_oh.append(onehot)
            voter_pw.append(pairwise)

        for _ in range(len(voter_oh), n):
            voter_oh.append([0.0] * (m * m))
            voter_pw.append([0.0] * (m * m))

        batch_onehot.append(voter_oh)
        batch_pairwise.append(voter_pw)

    t_oh = torch.tensor(batch_onehot, dtype=torch.float32)
    t_pw = torch.tensor(batch_pairwise, dtype=torch.float32)
    logits = model(t_oh, t_pw)
    return logits


def SetTransformerV3_2rule_prediction(model, profile, full=False):
    """Predict winners from a single profile."""
    model.eval()
    with torch.no_grad():
        t_oh, t_pw = profile_to_set_input_v3(
            profile, model.max_num_voters, model.max_num_alternatives
        )
        logits = model(t_oh, t_pw)
        binary = torch.round(torch.sigmoid(logits)).squeeze()
        if not full:
            return [i for i in range(len(binary))
                    if int(binary[i]) == 1 and i in profile.candidates]
        else:
            return [i for i in range(len(binary)) if int(binary[i]) == 1]


def SetTransformerV3_2rule(model, full=False):
    """Returns a voting rule function from the model."""
    return lambda profile: SetTransformerV3_2rule_prediction(model, profile, full)


def SetTransformerV3_2rule_prediction_n(model, profile, num_samples, full=False):
    """
    Neutrality-averaged prediction.
    Generates all (or sampled) permutations, de-permutes, and averages.
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

        # Batch compute logits
        batch_oh = []
        batch_pw = []
        for perm_profile in profiles_perm:
            voter_oh = []
            voter_pw = []
            for ranking in perm_profile.rankings:
                onehot = [0.0] * (m * m)
                pairwise = [0.0] * (m * m)
                for pos, alt in enumerate(ranking):
                    if pos < m and alt < m:
                        onehot[pos * m + alt] = 1.0
                for i in range(min(len(ranking), m)):
                    for j in range(i + 1, min(len(ranking), m)):
                        alt_i = ranking[i]
                        alt_j = ranking[j]
                        if alt_i < m and alt_j < m:
                            pairwise[alt_i * m + alt_j] = 1.0
                voter_oh.append(onehot)
                voter_pw.append(pairwise)

            for _ in range(len(voter_oh), model.max_num_voters):
                voter_oh.append([0.0] * (m * m))
                voter_pw.append([0.0] * (m * m))

            batch_oh.append(voter_oh)
            batch_pw.append(voter_pw)

        t_oh = torch.tensor(batch_oh, dtype=torch.float32)
        t_pw = torch.tensor(batch_pw, dtype=torch.float32)
        logits = model(t_oh, t_pw)

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


def SetTransformerV3_2rule_n(model, num_samples, full=False):
    """Returns a neutrality-averaged voting rule function."""
    return lambda profile: SetTransformerV3_2rule_prediction_n(
        model, profile, num_samples, full
    )