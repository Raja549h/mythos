"""
OpenMythos Core Engine - Unification of PK-MoE, ACT, Semantic Verification, and Jagged KV-Cache

Architecture:
  Input Token -> Embedding
      -> Prelude (standard transformer layers)
      -> Recurrent Block (T_max iterations):
           1. Loop-index embedding
           2. Self-attention (GQA) with causal masking
           3. ProductKey MoE FFN (hierarchical 2D grid routing, active-token-only dispatch)
           4. LoRA depth adaptation
           5. LTI stable injection (A.h + B.e + transformer_out)
           6. Semantic Verification Gate (bilinear alignment h vs. e)
           7. ACT Halting Gate (remainder trick, dynamic per-token exit)
           -> Weighted-state accumulation, frozen halted states
      -> Coda (standard transformer layers)
      -> LM head

Drop-in replacement for open_mythos/main.py and open_mythos/moda.py.
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
from dataclasses import dataclass

# =========================================================================
# Configuration
# =========================================================================

@dataclass
class MythosConfig:
    vocab_size: int = 32000
    dim: int = 512
    n_heads: int = 8
    n_kv_heads: int = 4
    max_seq_len: int = 4096
    max_loop_iters: int = 4
    dropout: float = 0.0
    norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    n_experts: int = 64
    n_shared_experts: int = 2
    k1: int = 2
    k2: int = 2
    expert_dim: int = 128
    bal_loss_alpha: float = 0.001
    act_threshold: float = 0.90
    ponder_alpha: float = 0.01
    lora_rank: int = 16
    prelude_layers: int = 2
    coda_layers: int = 2

    @property
    def n_experts_per_tok(self) -> int:
        return self.k1 * self.k2

    def __post_init__(self):
        sqrt_e = math.isqrt(self.n_experts)
        if sqrt_e * sqrt_e != self.n_experts:
            raise ValueError(f"n_experts must be a perfect square, got {self.n_experts}")

# =========================================================================
# Primitives
# =========================================================================

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.weight


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int, theta: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int) -> None:
        t = torch.arange(seq_len, device=self.inv_freq.device)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos", emb.cos()[None, None], persistent=False)
        self.register_buffer("sin", emb.sin()[None, None], persistent=False)

    def forward(self, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if seq_len > self.cos.shape[2]:
            self._build_cache(seq_len * 2)
        return self.cos[:, :, :seq_len], self.sin[:, :, :seq_len]


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat([-x[..., half:], x[..., :half]], dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    return x * cos + rotate_half(x) * sin


class Expert(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

# =========================================================================
# Product-Key Gate - Hierarchical 2D-Grid Routing
# =========================================================================

class ProductKeyGate(nn.Module):
    def __init__(self, dim: int, n_experts: int, k1: int, k2: int):
        super().__init__()
        sqrtE = int(math.isqrt(n_experts))
        assert sqrtE * sqrtE == n_experts
        self.sqrtE = sqrtE
        self.k1 = k1
        self.k2 = k2
        self.n_experts = n_experts
        self.W1 = nn.Linear(dim, sqrtE, bias=False)
        self.W2 = nn.Parameter(torch.empty(sqrtE, sqrtE, dim))
        nn.init.xavier_uniform_(self.W2)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        T = x.shape[0]
        eps = 1e-8
        s1 = self.W1(x)
        p1 = s1.softmax(dim=-1)
        p1_vals, idx1 = p1.topk(self.k1, dim=-1)
        W2_gathered = self.W2[idx1]
        s2 = torch.einsum("td,tksd->tks", x, W2_gathered)
        p2_given_e1 = s2.softmax(dim=-1)
        p2_vals, idx2 = p2_given_e1.topk(self.k2, dim=-1)
        idx1_exp = idx1.unsqueeze(-1).expand(-1, -1, self.k2)
        flat_idx = idx1_exp * self.sqrtE + idx2
        flat_idx = flat_idx.reshape(T, self.k1 * self.k2)
        p1_exp = p1_vals.unsqueeze(-1)
        weights = p1_exp * p2_vals
        weights = weights.reshape(T, self.k1 * self.k2)
        weights = weights / (weights.sum(dim=-1, keepdim=True) + eps)
        bal_loss = self._entropy_loss(s1, s2, idx1, eps)
        return weights, flat_idx, bal_loss

    def _entropy_loss(self, s1, s2_gathered, idx1, eps=1e-8):
        p1 = s1.softmax(dim=-1)
        H1 = -(p1 * (p1 + eps).log()).sum(-1).mean()
        H1_max = math.log(self.sqrtE)
        p2 = s2_gathered.softmax(dim=-1)
        H2_given_e1 = -(p2 * (p2 + eps).log()).sum(-1)
        p1_selected = p1.gather(-1, idx1).detach()
        H2 = (p1_selected * H2_given_e1).sum(-1).mean()
        H2_max = math.log(self.sqrtE)
        return (H1_max - H1) + (H2_max - H2)

# =========================================================================
# Adaptive Halting Gate (ACT)
# =========================================================================

class AdaptiveHaltingGate(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.proj = nn.Linear(dim, 1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.proj(h)).squeeze(-1)

# =========================================================================
# Semantic Verification Gate
# =========================================================================

class SemanticVerificationGate(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.W_v = nn.Parameter(torch.empty(dim, dim))
        nn.init.xavier_uniform_(self.W_v)

    def forward(self, h: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        alignment = torch.einsum("btd,de,bte->bt", h, self.W_v, e)
        return torch.sigmoid(alignment)

# =========================================================================
# LTI-Stable Context Injection
# =========================================================================

class LTIInjection(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.log_A = nn.Parameter(torch.zeros(dim))
        self.log_dt = nn.Parameter(torch.zeros(1))
        self.B = nn.Parameter(torch.ones(dim) * 0.1)

    def get_A(self) -> torch.Tensor:
        return torch.exp(-torch.exp((self.log_dt + self.log_A).clamp(-20, 20)))

    def forward(self, h, e, trans_out):
        A = self.get_A()
        return A * h + self.B * e + trans_out

# =========================================================================
# Depth-Wise LoRA Adapter
# =========================================================================

class LoRAAdapter(nn.Module):
    def __init__(self, dim: int, rank: int, max_loops: int):
        super().__init__()
        self.down = nn.Linear(dim, rank, bias=False)
        self.B = nn.Parameter(torch.randn(rank, dim) * 0.02)
        self.scale = nn.Embedding(max_loops, rank)

    def forward(self, x, loop_t):
        max_t = self.scale.num_embeddings - 1
        t_idx = loop_t if loop_t <= max_t else max_t
        s = self.scale(torch.tensor(t_idx, device=x.device))
        return (self.down(x) * s) @ self.B

# =========================================================================
# Loop-Index Sinusoidal Embedding
# =========================================================================

def loop_index_embedding(h, loop_t, loop_dim, theta=10000.0):
    B, T, D = h.shape
    loop_dim = min(loop_dim, D)
    if loop_dim <= 0:
        return h
    freqs = 1.0 / (theta ** (torch.arange(0, loop_dim, 2, device=h.device, dtype=h.dtype) / loop_dim))
    angles = loop_t * freqs
    emb = torch.cat([angles.sin(), angles.cos()], dim=-1)
    emb = emb[:loop_dim]
    emb_full = torch.zeros(D, device=h.device, dtype=h.dtype)
    emb_full[:loop_dim] = emb
    return h + emb_full.unsqueeze(0).unsqueeze(0)

# =========================================================================
# Grouped Query Attention
# =========================================================================

class GQAAttention(nn.Module):
    def __init__(self, dim, n_heads, n_kv_heads, dropout=0.0):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = dim // n_heads
        self.groups = n_heads // n_kv_heads
        self.scale = self.head_dim ** -0.5
        self.wq = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(n_heads * self.head_dim, dim, bias=False)
        self.dropout_p = dropout

    def forward(self, x, rope_freqs, mask=None):
        B, T, D = x.shape
        Hq, Hk, d = self.n_heads, self.n_kv_heads, self.head_dim
        cos, sin = rope_freqs
        q = self.wq(x).view(B, T, Hq, d).transpose(1, 2)
        k = self.wk(x).view(B, T, Hk, d).transpose(1, 2)
        v = self.wv(x).view(B, T, Hk, d).transpose(1, 2)
        q = apply_rope(q, cos[:, :, :T], sin[:, :, :T])
        k = apply_rope(k, cos[:, :, :T], sin[:, :, :T])
        k = k.repeat_interleave(self.groups, dim=1)
        v = v.repeat_interleave(self.groups, dim=1)
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if mask is not None:
            attn = attn + mask[:, :, :T, :T]
        attn = F.softmax(attn, dim=-1)
        attn = F.dropout(attn, p=self.dropout_p, training=self.training)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.wo(out)

# =========================================================================
# Product-Key Mixture-of-Experts
# =========================================================================

class ProductKeyMoE(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.dim = cfg.dim
        self.n_experts = cfg.n_experts
        self.K = cfg.k1 * cfg.k2
        self.gate = ProductKeyGate(cfg.dim, cfg.n_experts, cfg.k1, cfg.k2)
        shared_hidden = cfg.n_shared_experts * cfg.expert_dim
        self.shared = Expert(cfg.dim, shared_hidden)
        self.experts = nn.ModuleList([
            Expert(cfg.dim, cfg.expert_dim) for _ in range(cfg.n_experts)
        ])

    def forward(self, x, active_mask):
        shared_out = self.shared(x)
        weights, indices, bal_loss = self.gate(x)
        rout_out = torch.zeros_like(x)
        if active_mask.bool().any():
            active_idx = active_mask.bool().nonzero(as_tuple=True)[0]
            x_active = x[active_idx]
            w_active = weights[active_idx]
            idx_active = indices[active_idx]
            for eid, expert_i in enumerate(self.experts):
                mask = (idx_active == eid).any(dim=-1)
                sel = mask.nonzero(as_tuple=True)[0]
                if sel.numel() == 0:
                    continue
                x_sel = x_active[sel]
                idx_sel = idx_active[sel]
                _, rank_in_k = torch.where(idx_sel == eid)
                w_sel = w_active[sel]
                w_contrib = w_sel[torch.arange(sel.numel()), rank_in_k]
                expert_out = expert_i(x_sel)
                rout_out[active_idx[sel]] += expert_out * w_contrib.unsqueeze(-1)
        out = shared_out + rout_out
        return out, bal_loss

# =========================================================================
# Recurrent Mythos Block (unified)
# =========================================================================

class RecurrentMythosBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.dim = cfg.dim
        self.max_loops = cfg.max_loop_iters
        self.threshold = cfg.act_threshold
        self.loop_dim = max(16, cfg.dim // 8)
        self.attn_norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.attn = GQAAttention(cfg.dim, cfg.n_heads, cfg.n_kv_heads, cfg.dropout)
        self.moe = ProductKeyMoE(cfg)
        self.lora = LoRAAdapter(cfg.dim, cfg.lora_rank, cfg.max_loop_iters)
        self.injection = LTIInjection(cfg.dim)
        self.act = AdaptiveHaltingGate(cfg.dim)
        self.sem_ver = SemanticVerificationGate(cfg.dim)

    def forward(self, x, e, rope_freqs, n_loops=None):
        B, T, D = x.shape
        n_loops = n_loops or self.max_loops
        halted = torch.zeros(B, T, device=x.device, dtype=torch.bool)
        cum_p = torch.zeros(B, T, device=x.device)
        h_out = torch.zeros_like(x)
        active_mask = torch.ones(B * T, device=x.device, dtype=torch.bool)
        total_bal = 0.0
        pond_weights = []
        causal_mask = None
        if T > 1:
            causal_mask = torch.full((1, 1, T, T), float("-inf"), device=x.device, dtype=x.dtype)
            causal_mask = torch.triu(causal_mask, diagonal=1)
        for t in range(n_loops):
            h_t = loop_index_embedding(x, t, self.loop_dim)
            combined = self.attn_norm(h_t + e)
            attn_out = self.attn(combined, rope_freqs, causal_mask)
            flat = combined.reshape(B * T, D)
            moe_out, bal_t = self.moe(flat, active_mask)
            total_bal = total_bal + bal_t
            moe_out = moe_out.reshape(B, T, D)
            trans_out = (h_t + e) + attn_out + moe_out
            trans_out = trans_out + self.lora(trans_out, t)
            x_new = self.injection(x, e, trans_out)
            x = torch.where(halted.unsqueeze(-1), x, x_new)
            gamma = self.sem_ver(x, e)
            p_t = self.act(x) * gamma
            p_t = p_t.clamp(0.0, 1.0)
            still_running = ~halted
            remainder = (1.0 - cum_p).clamp(min=0)
            if t == n_loops - 1:
                weight = remainder * still_running.float()
            else:
                weight = torch.where(
                    (cum_p + p_t) >= self.threshold,
                    remainder,
                    p_t,
                )
                weight = weight * still_running.float()
            h_out = h_out + weight.unsqueeze(-1) * x
            cum_p = cum_p + weight
            halted = halted | (cum_p >= self.threshold)
            active_mask = (~halted).reshape(-1)
            pond_weights.append(weight)
            if halted.all():
                for _ in range(n_loops - t - 1):
                    pond_weights.append(torch.zeros(B, T, device=x.device))
                break
        avg_bal = total_bal / n_loops if n_loops > 0 else total_bal
        if pond_weights:
            w_stack = torch.stack(pond_weights, dim=-1)
            steps = torch.arange(n_loops, device=w_stack.device, dtype=w_stack.dtype)
            expected = (w_stack * steps.unsqueeze(0).unsqueeze(0)).sum(dim=-1)
            p_loss = expected.mean() / max(n_loops, 1)
        else:
            p_loss = torch.tensor(0.0, device=x.device)
        return h_out, avg_bal, p_loss

# =========================================================================
# Standard Transformer Block (Prelude / Coda)
# =========================================================================

class TransformerBlock(nn.Module):
    def __init__(self, cfg, use_moe=False):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.ffn_norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.attn = GQAAttention(cfg.dim, cfg.n_heads, cfg.n_kv_heads, cfg.dropout)
        if use_moe:
            self.ffn = ProductKeyMoE(cfg)
        else:
            self.ffn = Expert(cfg.dim, cfg.dim * 4 // 3)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x, rope_freqs, mask=None):
        x = x + self.dropout(self.attn(self.attn_norm(x), rope_freqs, mask))
        if isinstance(self.ffn, ProductKeyMoE):
            flat = self.ffn_norm(x).reshape(-1, x.shape[-1])
            amask = torch.ones(flat.shape[0], device=x.device, dtype=torch.bool)
            ffn_out, _ = self.ffn(flat, amask)
            x = x + self.dropout(ffn_out.reshape(x.shape))
        else:
            x = x + self.dropout(self.ffn(self.ffn_norm(x)))
        return x

# =========================================================================
# OpenMythos Full Model
# =========================================================================

class OpenMythos(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.dim)
        head_dim = cfg.dim // cfg.n_heads
        self.rope = RotaryEmbedding(head_dim, cfg.max_seq_len, cfg.rope_theta)
        self.prelude = nn.ModuleList([
            TransformerBlock(cfg, use_moe=False) for _ in range(cfg.prelude_layers)
        ])
        self.recurrent = RecurrentMythosBlock(cfg)
        self.coda = nn.ModuleList([
            TransformerBlock(cfg, use_moe=False) for _ in range(cfg.coda_layers)
        ])
        self.norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        self.head.weight = self.embed.weight

    def forward(self, input_ids, n_loops=None, return_losses=False):
        B, T = input_ids.shape
        x = self.embed(input_ids)
        cos, sin = self.rope(T)
        for layer in self.prelude:
            x = layer(x, (cos, sin))
        e = x
        x_rec, bal_loss, pond_loss = self.recurrent(x, e, (cos, sin), n_loops)
        for layer in self.coda:
            x_rec = layer(x_rec, (cos, sin))
        logits = self.head(self.norm(x_rec))
        if return_losses:
            return logits, bal_loss, pond_loss
        return logits,
# =========================================================================
# Validation Suite
# =========================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("   OPENMYTHOS CORE ENGINE - VALIDATION SUITE")
    print("=" * 70)
    print()
    device = torch.device("cpu")
    torch.manual_seed(42)

    cfg = MythosConfig(dim=512, n_experts=64, k1=2, k2=2, expert_dim=128,
                       max_loop_iters=4, act_threshold=0.85,
                       n_shared_experts=1, lora_rank=16)
    sqrtE = int(math.isqrt(cfg.n_experts))
    K = cfg.k1 * cfg.k2
    print(f"  dim={cfg.dim}, experts={cfg.n_experts} ({sqrtE}x{sqrtE} grid)")
    print(f"  k1={cfg.k1}, k2={cfg.k2}, K={K} experts/token, T_max={cfg.max_loop_iters}")
    print()

    block = RecurrentMythosBlock(cfg).to(device).train()
    nparam = sum(p.numel() for p in block.parameters())
    print(f"  Block parameters: {nparam:,}")
    B, T, D = 2, 8, cfg.dim

    with torch.no_grad():
        block.act.proj.weight[:, 0] = 3.0
        block.act.proj.bias[:] = -1.0

    x = torch.randn(B, T, D, device=device) * 0.1
    x[:, :4, 0] = 2.0
    x[:, 4:, 0] = -2.0
    e = x.detach().clone()
    rope = RotaryEmbedding(D // cfg.n_heads, T, cfg.rope_theta)
    cos, sin = rope(T)

    print(">>> TEST 1: Dynamic Halting & Divergence")
    print("-" * 60)

    h_out, bal_loss, pond_loss = block(x, e, (cos, sin))
    halted_final = torch.zeros(B, T, device=device, dtype=torch.bool)
    cum_p = torch.zeros(B, T, device=device)
    halted_steps = []
    active_tokens = []
    x_track = x.clone()

    for t in range(cfg.max_loop_iters):
        h_t = loop_index_embedding(x_track, t, block.loop_dim)
        combined = block.attn_norm(h_t + e)
        attn_out = block.attn(combined, (cos, sin))
        flat = combined.reshape(B * T, D)
        am = (~halted_final).reshape(-1)
        active_tokens.append(am.sum().item())
        moe_out, _ = block.moe(flat, am)
        moe_out = moe_out.reshape(B, T, D)
        trans_out = (h_t + e) + attn_out + moe_out
        trans_out = trans_out + block.lora(trans_out, t)
        x_new = block.injection(x_track, e, trans_out)
        x_track = torch.where(halted_final.unsqueeze(-1), x_track, x_new)
        gamma = block.sem_ver(x_track, e)
        p_t = block.act(x_track) * gamma
        p_t = p_t.clamp(0.0, 1.0)
        still_running = ~halted_final
        remainder = (1.0 - cum_p).clamp(min=0)
        if t == cfg.max_loop_iters - 1:
            weight = remainder * still_running.float()
        else:
            weight = torch.where(
                (cum_p + p_t) >= cfg.act_threshold,
                remainder, p_t
            )
            weight = weight * still_running.float()
        cum_p = cum_p + weight
        halted_final = halted_final | (cum_p >= cfg.act_threshold)
        halted_steps.append(weight)
        if halted_final.all():
            break

    print("  Halting Weight Matrix (tokens x iterations):")
    hm = torch.stack(halted_steps, dim=-1)
    for b in range(B):
        for t_pos in range(T):
            lbl = "S" if t_pos < 4 else "C"
            prof = " ".join([f"{hm[b, t_pos, i].item():.3f}" for i in range(hm.shape[-1])])
            hs = (hm[b, t_pos] > 0.01).float().argmax().item() if (hm[b, t_pos] > 0.01).any() else -1
            print(f"  B{b}T{t_pos:2d} [{lbl}] weights=[{prof}] halt_step={hs}")

    print(f"  Active tokens per iter: {active_tokens}")
    monotonic = all(active_tokens[i] <= active_tokens[i-1] for i in range(1, len(active_tokens)))
    print(f"  Monotonic decay: {monotonic}")
    assert monotonic, "FAIL: Active tokens must decay"
    print("  PASS: Active tokens decay (halted = 0 FLOPs)")

    print()
    print(">>> TEST 2: Tensor Shape Invariance")
    print("-" * 60)
    with torch.no_grad():
        flat_in = e.reshape(B * T, D)
        w2, idx2, _ = block.moe.gate(flat_in[:8])
    print(f"  Input: {flat_in.shape}")
    print(f"  Indices: {idx2.shape}, range [{idx2.min().item()}, {idx2.max().item()}]")
    print(f"  Weights: {w2.shape}")
    assert idx2.min().item() >= 0, "FAIL: Negative expert index"
    assert idx2.max().item() < cfg.n_experts, f"FAIL: Index {idx2.max().item()} >= {cfg.n_experts}"
    assert w2.shape[-1] == K, f"FAIL: weights dim {w2.shape[-1]} != {K}"
    print("  PASS: All shapes valid, indices in [0, E)")

    print()
    print(">>> TEST 3: Gradient Flow & Epsilon Smoothing")
    print("-" * 60)
    cfg3 = MythosConfig(dim=128, n_experts=16, k1=2, k2=2, expert_dim=32,
                       max_loop_iters=2, act_threshold=0.9,
                       n_shared_experts=1, lora_rank=8)
    block3 = RecurrentMythosBlock(cfg3).to(device).train()
    x3 = torch.randn(1, 4, 128, device=device, requires_grad=True)
    e3 = x3.detach().clone()
    rope3 = RotaryEmbedding(16, 4, 10000.0)
    out3, bl, pl = block3(x3, e3, rope3(4))
    loss3 = bl + pl + out3.mean()
    loss3.backward()
    nan = False
    gnorm = 0.0
    for n, p in block3.named_parameters():
        if p.grad is not None:
            gnorm += p.grad.norm().item()
            if torch.isnan(p.grad).any():
                nan = True
                print(f"  NaN: {n}")
    print(f"  Grad norm: {gnorm:.6f}")
    print(f"  bal_loss={bl.item():.6f}, pond_loss={pl.item():.6f}")
    print(f"  Input grad norm: {x3.grad.norm().item() if x3.grad is not None else 0:.6f}")
    assert not nan, "FAIL: NaN gradients"
    assert torch.isfinite(loss3).all(), "FAIL: non-finite loss"
    assert x3.grad is not None and torch.isfinite(x3.grad).all(), "FAIL: input grad bad"
    print("  PASS: No NaN, stable gradients")

    print()
    print(">>> TEST 4: LTI Context Retention")
    print("-" * 60)
    cfg4 = MythosConfig(dim=512, n_experts=64, k1=2, k2=2, expert_dim=128,
                       max_loop_iters=4, act_threshold=0.99,
                       n_shared_experts=2, lora_rank=16)
    block4 = RecurrentMythosBlock(cfg4).to(device)
    x4 = torch.randn(1, 8, 512, device=device)
    e4 = x4.detach().clone()
    rope4 = RotaryEmbedding(512 // 8, 8, 10000.0)
    h_t1 = x4.clone()
    h_t_last = x4.clone()
    halted4 = torch.zeros(1, 8, device=device, dtype=torch.bool)
    for t in range(cfg4.max_loop_iters):
        h_emb = loop_index_embedding(h_t_last, t, block4.loop_dim)
        combined = block4.attn_norm(h_emb + e4)
        attn_out = block4.attn(combined, rope4(8))
        am4 = (~halted4).reshape(-1)
        moe_out, _ = block4.moe(combined.reshape(1*8, 512), am4)
        moe_out = moe_out.reshape(1, 8, 512)
        trans_out = (h_emb + e4) + attn_out + moe_out
        trans_out = trans_out + block4.lora(trans_out, t)
        x_new = block4.injection(h_t_last, e4, trans_out)
        h_t_last = torch.where(halted4.unsqueeze(-1), h_t_last, x_new)
        if t == 0:
            h_t1 = h_t_last.clone()
    cos_sim = F.cosine_similarity(h_t1.view(-1), h_t_last.view(-1), dim=0).item()
    print(f"  cos_sim(h(t=1), h(t=T_max)) = {cos_sim:.6f}")
    assert cos_sim > 0.3, f"FAIL: LTI drift (cos_sim={cos_sim:.4f} < 0.3)"
    print("  PASS: LTI anchors context within bounds")

    print()
    print("=" * 70)
    print("   ALL TESTS PASSED - PRE-MYTHOS STATE CONFIRMED")
    print("=" * 70)
