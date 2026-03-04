"""
set_transformer.py — Set-Transformer architecture for Axiomatic Deep Voting

Implements the Set-Transformer from Anil & Bao (2021) / Lee et al. (2019)
with Pooling by Multihead Attention (PMA), adapted to the Hornischer &
Terzopoulou codebase.

Architecture (following Anil & Bao, Appendix A.2.1):
  Encoder: 4 × SAB  (Self-Attention Blocks, pre-LayerNorm variant)
  Decoder: PMA_k → SAB → rFF

The model is anonymous by design (permutation-invariant over voters)
because the encoder is permutation-equivariant and PMA is
permutation-invariant.  Neutrality is obtained via neutrality-averaged
decoding, exactly like the WEC.
"""

import math
import itertools
from random import sample as random_sample

import torch
from torch import nn
from pref_voting.profiles import Profile
from utils import profile_to_onehot, flatten_onehot_profile


# =====================================================================
#  Building Blocks
# =====================================================================

class MultiheadAttentionBlock(nn.Module):
    """MAB(X, Y) = LayerNorm(H + rFF(H))  where H = LayerNorm(X + MHA(X, Y, Y))
    
    Uses the pre-LayerNorm variant (Xiong et al., 2020) as recommended
    by Anil & Bao for training stability.
    """

    def __init__(self, d_model, n_heads, d_head, ff_dim=None):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_head

        # Pre-LN on queries/keys
        self.ln1 = nn.LayerNorm(d_model)
        self.ln_kv = nn.LayerNorm(d_model)  # normalise key-value input too

        # Multihead attention projections
        self.W_q = nn.Linear(d_model, n_heads * d_head, bias=False)
        self.W_k = nn.Linear(d_model, n_heads * d_head, bias=False)
        self.W_v = nn.Linear(d_model, n_heads * d_head, bias=False)
        self.W_o = nn.Linear(n_heads * d_head, d_model, bias=False)

        # Feed-forward
        if ff_dim is None:
            ff_dim = d_model
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, d_model),
        )

    def _multihead_attn(self, Q, K, V):
        B, n_q, _ = Q.shape
        _, n_kv, _ = K.shape
        h, d = self.n_heads, self.d_head

        Q = self.W_q(Q).view(B, n_q, h, d).transpose(1, 2)   # (B, h, n_q, d)
        K = self.W_k(K).view(B, n_kv, h, d).transpose(1, 2)  # (B, h, n_kv, d)
        V = self.W_v(V).view(B, n_kv, h, d).transpose(1, 2)  # (B, h, n_kv, d)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d)
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, V)                           # (B, h, n_q, d)
        out = out.transpose(1, 2).contiguous().view(B, n_q, h * d)
        return self.W_o(out)

    def forward(self, X, Y):
        # Pre-LN variant: normalise before attention
        X_norm = self.ln1(X)
        Y_norm = self.ln_kv(Y)
        H = X + self._multihead_attn(X_norm, Y_norm, Y_norm)
        out = H + self.ff(self.ln2(H))
        return out


class SAB(nn.Module):
    """Self-Attention Block:  SAB(X) = MAB(X, X)"""

    def __init__(self, d_model, n_heads, d_head, ff_dim=None):
        super().__init__()
        self.mab = MultiheadAttentionBlock(d_model, n_heads, d_head, ff_dim)

    def forward(self, X):
        return self.mab(X, X)


class PMA(nn.Module):
    """Pooling by Multihead Attention:  PMA_k(Z) = MAB(S, rFF(Z))
    
    S ∈ R^{k × d} is a learnable seed matrix.
    k = number of output "slots" (we use k=1 for voting → single pooled vector).
    """

    def __init__(self, d_model, n_heads, d_head, k=1, ff_dim=None):
        super().__init__()
        self.seed = nn.Parameter(torch.randn(1, k, d_model))
        nn.init.xavier_uniform_(self.seed)
        self.ff_pre = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
        )
        self.mab = MultiheadAttentionBlock(d_model, n_heads, d_head, ff_dim)

    def forward(self, Z):
        B = Z.shape[0]
        S = self.seed.expand(B, -1, -1)       # (B, k, d)
        return self.mab(S, self.ff_pre(Z))     # (B, k, d)


# =====================================================================
#  Set-Transformer for Voting
# =====================================================================

class SetTransformer(nn.Module):
    """
    Set-Transformer adapted for the axiomatic deep voting framework.

    Input encoding: one-hot representation (following Anil & Bao, Appendix C).
    Each voter's ranking is encoded as a vote vector of dimension
    m_max * m_max (one-hot encoding of each position), then projected to d_model.
    The set of voters is the "set" processed by the transformer.

    Architecture (Anil & Bao, Appendix A.2.1):
      Encoder: input_proj → 4 × SAB
      Decoder: PMA_1 → SAB → output_proj
    """

    def __init__(
        self,
        max_num_voters,
        max_num_alternatives,
        d_model=128,
        n_heads=4,
        d_head=32,
        n_encoder_layers=4,
        pma_outputs=1,
    ):
        super().__init__()
        self.max_num_voters = max_num_voters
        self.max_num_alternatives = max_num_alternatives

        # Input dimensionality: each voter's ranking is a one-hot profile
        # row of shape (m_max × m_max) flattened
        input_dim = max_num_alternatives * max_num_alternatives

        # Project each voter's vote-vector into d_model dimensions
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.ReLU(),
            nn.LayerNorm(d_model),
        )

        # Encoder: stack of SABs
        self.encoder = nn.ModuleList([
            SAB(d_model, n_heads, d_head, ff_dim=d_model)
            for _ in range(n_encoder_layers)
        ])

        # Decoder: PMA → SAB → output projection
        self.pma = PMA(d_model, n_heads, d_head, k=pma_outputs, ff_dim=d_model)
        self.decoder_sab = SAB(d_model, n_heads, d_head, ff_dim=d_model)
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, max_num_alternatives),
        )

    def forward(self, x):
        """
        x: (B, n_voters, m_max * m_max)  — one-hot encoded vote vectors
        Returns: (B, m_max) logits
        """
        # Project each voter into d_model
        h = self.input_proj(x)         # (B, n_voters, d_model)

        # Encode with self-attention
        for sab in self.encoder:
            h = sab(h)                 # (B, n_voters, d_model)

        # Decode: pool voters into k output slots, then refine
        h = self.pma(h)                # (B, k, d_model)
        h = self.decoder_sab(h)        # (B, k, d_model)
        h = h.squeeze(1)               # (B, d_model)  if k=1
        logits = self.output_proj(h)   # (B, m_max)
        return logits


# =====================================================================
#  Encoding: Profile → Tensor for the Set-Transformer
# =====================================================================

def profile_to_set_transformer_input(profile, max_num_voters, max_num_alternatives):
    """
    Encode a single profile as a tensor of shape (n_max, m_max * m_max).

    Each voter's ranking is one-hot encoded per position (same scheme as
    the rest of the codebase), then flattened into a single vote vector.
    Padded voters get all-zero vectors.
    """
    onehot = profile_to_onehot(profile, max_num_voters, max_num_alternatives)
    # onehot is a list of n_max rankings, each a list of m_max one-hot vectors
    # of length m_max → flatten each ranking to m_max * m_max
    flat = [
        [val for position_vec in ranking for val in position_vec]
        for ranking in onehot
    ]
    return torch.tensor(flat, dtype=torch.float)  # (n_max, m_max^2)


# =====================================================================
#  Bridge functions: logits, rule, rule_n  (matching WEC/CNN patterns)
# =====================================================================

def SetTransformer2logits(model, X):
    """
    Compute logits for a list of profiles X.
    Used during training (no eval mode, no no_grad).
    """
    tensorized = []
    for profile in X:
        t = profile_to_set_transformer_input(
            profile, model.max_num_voters, model.max_num_alternatives
        )
        tensorized.append(t)
    batch = torch.stack(tensorized, dim=0)  # (B, n_max, m_max^2)
    return model(batch)


def SetTransformer2rule_prediction(model, profile, full=False):
    """Single profile → winning set (plain, no averaging)."""
    model.eval()
    with torch.no_grad():
        x = profile_to_set_transformer_input(
            profile, model.max_num_voters, model.max_num_alternatives
        )
        logits = model(x.unsqueeze(0))  # (1, m_max)
        binary = torch.round(torch.sigmoid(logits)).squeeze()
        if not full:
            return [i for i in range(len(binary))
                    if int(binary[i]) == 1 and i in profile.candidates]
        else:
            return [i for i in range(len(binary)) if int(binary[i]) == 1]


def SetTransformer2rule_prediction_n(model, profile, num_samples, full=False):
    """Neutrality-averaged prediction (same logic as WEC2rule_prediction_n)."""
    model.eval()
    with torch.no_grad():
        num_alternatives = profile.num_cands
        profiles_list = []
        permutations = []

        if num_samples is None:
            for p in list(itertools.permutations(range(num_alternatives))):
                p_max = list(p) + list(range(len(p), model.max_num_alternatives))
                permutations.append(tuple(p_max))
                perm_rankings = [[p[alt] for alt in r] for r in profile.rankings]
                profiles_list.append(Profile(perm_rankings))
        else:
            for _ in range(num_samples):
                p = random_sample(list(range(num_alternatives)), num_alternatives)
                if p in permutations:
                    break
                p_max = p + list(range(len(p), model.max_num_alternatives))
                permutations.append(tuple(p_max))
                perm_rankings = [[p[alt] for alt in r] for r in profile.rankings]
                profiles_list.append(Profile(perm_rankings))

        # Batch all permuted profiles
        batch = torch.stack([
            profile_to_set_transformer_input(
                pf, model.max_num_voters, model.max_num_alternatives
            )
            for pf in profiles_list
        ], dim=0)

        logits = model(batch)

        # De-permute and average
        re_permuted = torch.zeros_like(logits)
        for j in range(len(logits)):
            re_permuted[j] = logits[j][permutations[j],]
        prediction = re_permuted.mean(dim=0)

        binary = torch.round(torch.sigmoid(prediction)).squeeze()
        if not full:
            return [i for i in range(len(binary))
                    if int(binary[i]) == 1 and i in profile.candidates]
        else:
            return [i for i in range(len(binary)) if int(binary[i]) == 1]


def SetTransformer2rule(model, full=False):
    return lambda profile: SetTransformer2rule_prediction(model, profile, full)


def SetTransformer2rule_n(model, sample, full=False):
    return lambda profile: SetTransformer2rule_prediction_n(
        model, profile, sample, full
    )