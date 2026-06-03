import os
import sys
import json
import asyncio
import random
import re
import aiohttp
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from http.server import HTTPServer, BaseHTTPRequestHandler
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from repo_scanner import RepoScanner

# Core architecture components from the OpenMythos stack
from mythos_unified import (
    GQAAttention,
    ProductKeyMoE,
    RotaryEmbedding,
    RMSNorm,
    TransformerBlock,
)
from open_mythos.moda import (
    MoDAAttention,
    MoDAConfig,
    apply_rotary_emb,
)

# ==========================================
# STAGE 1: THE MONOLITHIC MASTER DIRECTIVE
# ==========================================
MASTER_SYSTEM_PROMPT = """
# SYSTEM CONTEXT & IDENTITY
ROLE: SEC-CORE Unified Orchestration Network [Version: Cyber Top Engine v5.4]
OBJECTIVE: You are operating as a meta-orchestrator and a multi-layered reasoning engine. Your task is to ingest a target code snippet, network payload, or security report, simulate an internal consensus debate among four world-class security personas, subject their findings to an adversarial "Zero-Day Discovery Loop," and synthesize a production-grade, low-latency triage matrix.

Do not use conversational filler, meta-commentary, or introductory pleasantries. Begin execution immediately upon parsing the input.

---

# STAGE 2: CENTRAL ORCHESTRATION & INTENT PARSING
Analyze the user's input payload to calculate its operational intent and establish dynamic synthesis weights for Stage 4 resolution:
- IF Input contains Exploit/PoC Code -> Intent: Exploit Simulation. Weights: Vulnerability Depth (90%), Code Stability (10%).
- IF Input contains Production Source Code -> Intent: Production Patching. Weights: Security (80%), Performance (20%).
- IF Input contains Highly Optimized Assembly/Kernel Logic -> Intent: Hot-fix Optimization. Weights: Performance (60%), Security (40%).

---

# STAGE 3: THE QUAD-AGENT COUNCIL SIMULATION
Generate four distinct, deeply technical evaluation perspectives based strictly on the following persona vectors. Do not let these perspectives blur; maintain creative friction between them.

## PERSONA 01: Mythos-Glasswing [Systems Architect]
- Focus: Macro dependency trees, high-level structural design, application layer routing, state-machine integrity, and data serialization boundaries.

## PERSONA 02: Cyber-Decompiler [Deterministic Binary Specialist]
- Focus: Memory corruption, low-level pointer arithmetic, assembly execution paths, and the hardware-software interface.

## PERSONA 03: BigSleep-Mimic [AI Zero-Day Fuzzer]
- Focus: Complex semantic logic flaws, non-obvious heuristic anomalies, and multi-step exploitation chains.

## PERSONA 04: SEC-CORE Governor [Risk & Mitigation Control]
- Focus: Blast-radius mitigation, CVSS validation, real-world patching friction, and performance-security trade-offs.

---

# STAGE 4: CYBER TOP OVERSEER SYNTHESIS & ADVERSARIAL CRITIQUE
Act as the Master Overseer layer to ingest the four council perspectives and compile them into a unified directive:
1. RESOLVE CONFLICTS: Evaluate opposing viewpoints by applying the calculated math weights.
2. THE ZERO-DAY DISCOVERY LOOP: Subject your proposed fixes to a recursive critique before outputting. Ask yourself: "If this tactical patch is applied blindly to a live system, what secondary logical vulnerabilities, memory alignment issues, compiler-specific optimizations, or multi-threaded deadlocks will it introduce?" Refine your strategy until it survives its own adversarial loop.

---

# STAGE 5: MANDATORY OUTPUT SPECIFICATION
You must format your final synthesis exactly according to the structure below. Render the schema as clean, highly scannable Markdown layouts.

## Executive Telemetry Matrix
| Metric | Telemetry Value |
| :--- | :--- |
| **Vulnerability Vector** | [CWE Identifier & Structural Classification Name] |
| **Exploitability Score** | [Float scale 0.0 to 10.0 representing execution ease] |
| **Rollback Risk** | [Low/Medium/High with a brief technical summary of what dependencies could break] |

## Unified Coda Analysis
[Provide a definitive, frontier-level technical breakdown of the root cause.]

## Triage Matrix
### 1. Tactical Patch (Quick Mitigation)
[Exact, line-by-line secure code replacements or hot-fixes.]
### 2. Strategic Overhaul (Architectural Fix)
[Detail structural changes to the code design or framework dependencies.]
### 3. Defensive Telemetry
[Provide a fully functional, production-ready detection signature like a YARA rule or Snort signature.]
"""

# =========================================================================
# Byte-Level Tokenizer (fully offline, zero dependencies)
# =========================================================================

class ByteTokenizer:
    """Byte-level tokenizer for fully offline operation.

    Maps each UTF-8 byte (0–255) to a token ID with reserved special tokens.
    No HuggingFace, no internet, no external files required.
    """

    def __init__(self, vocab_size: int = 32000):
        self.vocab_size = vocab_size
        self._byte_offset = 3  # Reserve 0=PAD, 1=BOS, 2=EOS
        self.pad_id = 0
        self.bos_id = 1
        self.eos_id = 2

    def encode(self, text: str) -> List[int]:
        """Encode text to byte-level token IDs with BOS/EOS markers."""
        return (
            [self.bos_id]
            + [b + self._byte_offset for b in text.encode("utf-8")]
            + [self.eos_id]
        )

    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs back to text, skipping special tokens."""
        raw = []
        for tid in token_ids:
            if tid in (self.pad_id, self.bos_id, self.eos_id):
                continue
            b = tid - self._byte_offset
            if 0 <= b <= 255:
                raw.append(b)
        return bytes(raw).decode("utf-8", errors="replace")


def get_tokenizer(vocab_size: int = 32000):
    """Return the best available tokenizer, falling back to byte-level."""
    try:
        from open_mythos.tokenizer import MythosTokenizer

        tok = MythosTokenizer()
        tok.pad_id = 0
        tok.bos_id = 1
        tok.eos_id = 2
        return tok
    except Exception:
        return ByteTokenizer(vocab_size)


# =========================================================================
# Council Persona Definitions
# =========================================================================

COUNCIL_PERSONAS = {
    0: "Mythos-Glasswing",       # Systems Architect — macro dependency trees, state-machine integrity
    1: "Mythos-Glasswing",       # Systems Architect (deep refinement pass)
    2: "DepthFirst-DevSecOps",   # Syntax & path optimization, secure coding patterns
    3: "DepthFirst-DevSecOps",   # DevSecOps (deep refinement pass)
    4: "Cyber-Decompiler",       # Binary specialist — memory corruption, UAF, TOCTOU
    5: "Cyber-Decompiler",       # Decompiler (deep refinement pass)
    6: "BigSleep-Mimic",         # AI zero-day fuzzer — semantic logic flaws, multi-step chains
    7: "BigSleep-Mimic",         # BigSleep (deep refinement pass)
}

# Persona descriptions used in structured output
PERSONA_ROLES = {
    "Mythos-Glasswing": "Global macro-architecture, dependency trees, async race conditions",
    "DepthFirst-DevSecOps": "Syntax-level security, secure path optimization, input validation",
    "Cyber-Decompiler": "Memory corruption, pointer arithmetic, buffer overflows, UAF, TOCTOU",
    "BigSleep-Mimic": "Semantic logic flaws, multi-step exploitation chains, zero-day discovery",
}


# =========================================================================
# Configuration
# =========================================================================

@dataclass
class MythosConfig:
    """SEC-CORE engine configuration.

    Key fixes vs. original:
      act_threshold = 1.0   → forces ALL loops to execute (no premature halting)
      max_loop_iters = 8    → 2 passes per persona (4 personas × 2 = 8)
    """

    vocab_size: int = 32000
    dim: int = 768
    n_heads: int = 12
    n_kv_heads: int = 3
    max_seq_len: int = 4096
    max_loop_iters: int = 8        # 2 passes per persona × 4 personas
    prelude_layers: int = 3
    coda_layers: int = 3
    n_experts: int = 64
    k1: int = 2
    k2: int = 2
    expert_dim: int = 256
    lora_rank: int = 32
    act_threshold: float = 1.0     # FIXED: force all loops to run
    norm_eps: float = 1e-6
    rope_theta: float = 1000000.0
    lookahead_entropy_threshold: float = 0.2
    dropout: float = 0.0
    n_shared_experts: int = 2


# =========================================================================
# SEC-CORE Council Conditioners (Persona Routing Vectors)
# =========================================================================

class SECCouncilConditioners(nn.Module):
    """Learned per-persona bias vectors injected into each recurrent loop.

    The 4 persona vectors steer the shared transformer block to attend to
    different aspects of the input at each loop iteration, implementing the
    "Quad-Agent Council" as a purely architectural mechanism.
    """

    def __init__(self, dim: int, n_personas: int = 4):
        super().__init__()
        self.n_personas = n_personas
        self.conditioners = nn.Parameter(torch.randn(n_personas, dim) * 0.02)

    def get_conditioner(self, loop_t: int) -> torch.Tensor:
        """Return the persona vector for loop iteration `loop_t`.

        Each persona gets 2 consecutive loops (initial + refinement pass).
        """
        persona_idx = (loop_t // 2) % self.n_personas
        return self.conditioners[persona_idx]

    def get_persona_name(self, loop_t: int) -> str:
        """Human-readable name for the active persona at `loop_t`."""
        return COUNCIL_PERSONAS.get(loop_t, f"Persona-{loop_t}")


# =========================================================================
# Lookahead Halting Gate (FIXED initialization)
# =========================================================================

class LookaheadHaltingGate(nn.Module):
    """ACT halting gate with lookahead collision detection.

    CRITICAL FIX: halt_proj is initialized with zero weights and bias=-5.0
    so that sigmoid output ≈ 0.007 at init. This prevents the untrained
    gate from halting all computation at loop 1–2 (the root cause of
    "pre-decided messages" with random weights).

    Once the model is trained, the gate learns to halt appropriately.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.halt_proj = nn.Linear(dim, 1)
        self.lookahead_proj = nn.Linear(dim, dim, bias=False)
        nn.init.orthogonal_(self.lookahead_proj.weight)

        # FIX: Initialize so sigmoid(output) ≈ 0.007 — never halts early
        nn.init.zeros_(self.halt_proj.weight)
        nn.init.constant_(self.halt_proj.bias, -5.0)

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        p_halt = torch.sigmoid(self.halt_proj(h)).squeeze(-1)
        h_next = torch.tanh(self.lookahead_proj(h))
        cosine_sim = F.cosine_similarity(h, h_next, dim=-1)
        collision_signal = 1.0 - cosine_sim
        return p_halt, collision_signal


# =========================================================================
# LTI-Stable Injection (unchanged)
# =========================================================================

class SEC_LTIInjection(nn.Module):
    """Stable input injection: h_{t+1} = A·h_t + B·(e + conditioner) + trans_out.

    Spectral radius ρ(A) < 1 is guaranteed by construction via
    A = exp(-exp(log_dt + log_A)), keeping the recurrence stable
    regardless of loop depth.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.log_A = nn.Parameter(torch.zeros(dim))
        self.log_dt = nn.Parameter(torch.zeros(1))
        self.B = nn.Parameter(torch.ones(dim) * 0.1)

    def get_A(self) -> torch.Tensor:
        return torch.exp(-torch.exp((self.log_dt + self.log_A).clamp(-20, 20)))

    def forward(self, h, e, conditioner, trans_out):
        A = self.get_A()
        return A * h + self.B * (e + conditioner) + trans_out


# =========================================================================
# Recurrent SEC-CORE Block with MoDA Cross-Loop Attention
# =========================================================================

class RecurrentSECBlock(nn.Module):
    """The core recurrent block — one transformer block looped max_loop_iters times.

    Each iteration:
      1. Council conditioner injection (persona-specific bias)
      2. MoDA attention (jointly attends to current + ALL previous loops)
      3. ProductKey MoE (sparse expert routing)
      4. Lookahead collision detection + backtracking perturbation
      5. LTI stable injection (A·h + B·e + transformer_out)
      6. ACT halting accumulation (disabled until trained)
      7. Depth-write to MoDA KV cache (for next loop to read)

    The MoDA cross-loop attention is the key mechanism: loop 7 (BigSleep)
    can directly attend to the keys/values written by loop 1 (Glasswing),
    enabling genuine multi-agent deliberation within a single forward pass.
    """

    def __init__(self, cfg: MythosConfig):
        super().__init__()
        self.cfg = cfg
        self.dim = cfg.dim
        self.council_conditioners = SECCouncilConditioners(cfg.dim)
        self.attn_norm = RMSNorm(cfg.dim, cfg.norm_eps)

        # Cross-Loop MoDA Integration
        self.attn = MoDAAttention(MoDAConfig(
            d_model=cfg.dim,
            n_heads_q=cfg.n_heads,
            n_heads_kv=cfg.n_kv_heads,
            head_dim=cfg.dim // cfg.n_heads,
            attn_dropout=cfg.dropout,
        ))

        self.moe = ProductKeyMoE(cfg)

        # Depth write projections for MoDA cross-loop cache
        head_dim = cfg.dim // cfg.n_heads
        self.k_write = nn.Linear(cfg.dim, cfg.n_kv_heads * head_dim, bias=False)
        self.v_write = nn.Linear(cfg.dim, cfg.n_kv_heads * head_dim, bias=False)

        self.injection = SEC_LTIInjection(cfg.dim)
        self.halting = LookaheadHaltingGate(cfg.dim)
        self.lora = nn.Linear(cfg.dim, cfg.dim, bias=False)
        nn.init.zeros_(self.lora.weight)

    def forward(
        self,
        h: torch.Tensor,
        e: torch.Tensor,
        rope_freqs: Tuple[torch.Tensor, torch.Tensor],
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor], Dict[str, float]]:
        """Run the recurrent loop for max_loop_iters iterations.

        Returns:
            h_accum:        ACT-weighted hidden state accumulation (B, T, D)
            latent_history: List of detached hidden states per loop (for persona routing)
            telemetry:      Dict of computational telemetry (halt probs, expert routing, etc.)
        """
        B, T, D = h.shape
        latent_history = []
        depth_k_cache = []
        depth_v_cache = []

        halted = torch.zeros(B, T, device=h.device, dtype=torch.bool)
        cum_p = torch.zeros(B, T, device=h.device)
        h_accum = torch.zeros_like(h)
        cos, sin = rope_freqs

        # Telemetry tracking
        halt_probs_per_loop = []
        active_tokens_per_loop = []

        for t in range(self.cfg.max_loop_iters):
            cond = self.council_conditioners.get_conditioner(t)

            # 1. MoDA Attentional Pass (jointly attends to all previous loops)
            h_norm = self.attn_norm(h + cond)
            attn_out = self.attn(h_norm, depth_k_cache, depth_v_cache, cos, sin)

            # 2. MoE Pass with Backtracking
            active_mask = (~halted).view(-1)
            active_tokens_per_loop.append(active_mask.sum().item())

            p_halt, collision = self.halting(h)
            halt_probs_per_loop.append(p_halt.mean().item())

            backtrack_signal = (collision > self.cfg.lookahead_entropy_threshold).unsqueeze(-1)
            h_norm = h_norm + 0.01 * backtrack_signal * torch.randn_like(h_norm)

            moe_out, _ = self.moe(h_norm.view(-1, D), active_mask)
            moe_out = moe_out.view(B, T, D)

            # 3. Update & Injection
            trans_out = attn_out + moe_out
            trans_out = trans_out + self.lora(trans_out)
            h_new = self.injection(h, e, cond, trans_out)
            h = torch.where(halted.unsqueeze(-1), h, h_new)

            # 4. Depth Write for MoDA (next loops can attend to this loop)
            kw = self.k_write(h).view(B, T, self.cfg.n_kv_heads, -1).transpose(1, 2)
            vw = self.v_write(h).view(B, T, self.cfg.n_kv_heads, -1).transpose(1, 2)
            kw = apply_rotary_emb(kw, cos, sin)
            depth_k_cache.append(kw)
            depth_v_cache.append(vw)

            # 5. Halting Accumulation
            still_running = ~halted
            remainder = (1.0 - cum_p).clamp(min=0)
            weight = torch.where(
                (cum_p + p_halt) >= self.cfg.act_threshold,
                remainder,
                p_halt,
            )
            if t == self.cfg.max_loop_iters - 1:
                weight = remainder  # Assign all remaining mass on final loop
            weight = weight * still_running.float()
            h_accum = h_accum + weight.unsqueeze(-1) * h
            cum_p = cum_p + weight
            halted = halted | (cum_p >= self.cfg.act_threshold)

            latent_history.append(h.detach())

        telemetry = {
            "halt_probs_per_loop": halt_probs_per_loop,
            "active_tokens_per_loop": active_tokens_per_loop,
            "final_cum_p": cum_p.mean().item(),
            "loops_executed": len(latent_history),
        }

        return h_accum, latent_history, telemetry


# =========================================================================
# Coda Routing Head
# =========================================================================

class SEC_CodaRoutingHead(nn.Module):
    """Output head that can route to any persona's latent state.

    When section_idx is None, uses the unified h_final (post-coda).
    When section_idx is specified, uses that loop's raw latent state
    to produce persona-specific output.
    """

    def __init__(self, dim: int, vocab_size: int):
        super().__init__()
        self.output_norm = RMSNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)

    def forward(
        self,
        h_final: torch.Tensor,
        history: List[torch.Tensor],
        section_idx: Optional[int] = None,
    ) -> torch.Tensor:
        if section_idx is not None and section_idx < len(history):
            source_state = history[section_idx]
        else:
            source_state = h_final
        return self.head(self.output_norm(source_state))


# =========================================================================
# Flagship Model: SEC-CORE Unified
# =========================================================================

class SECCoreUnified(nn.Module):
    """SEC-CORE Unified — self-contained security analysis RDT model.

    Full pipeline: Prelude → Recurrent SEC Block (8 loops) → Coda → LM Head
    """

    def __init__(self, cfg: MythosConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.rope = RotaryEmbedding(cfg.dim // cfg.n_heads, cfg.max_seq_len, cfg.rope_theta)

        self.prelude = nn.ModuleList([
            TransformerBlock(cfg, use_moe=False) for _ in range(cfg.prelude_layers)
        ])
        self.recurrent = RecurrentSECBlock(cfg)
        self.coda = nn.ModuleList([
            TransformerBlock(cfg, use_moe=False) for _ in range(cfg.coda_layers)
        ])
        self.routing_head = SEC_CodaRoutingHead(cfg.dim, cfg.vocab_size)
        self.routing_head.head.weight = self.embed.weight  # Weight tying

    def forward(
        self,
        input_ids: torch.Tensor,
        section_idx: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        B, T = input_ids.shape
        x = self.embed(input_ids)
        cos, sin = self.rope(T)
        mask = None
        if T > 1:
            mask = torch.full(
                (1, 1, T, T), float("-inf"), device=input_ids.device, dtype=x.dtype
            )
            mask = torch.triu(mask, diagonal=1)

        for layer in self.prelude:
            x = layer(x, (cos, sin), mask)

        e = x
        h_final, history, telemetry = self.recurrent(x, e, (cos, sin), mask)

        for layer in self.coda:
            h_final = layer(h_final, (cos, sin), mask)

        logits = self.routing_head(h_final, history, section_idx)
        return logits, telemetry

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 64,
        temperature: float = 1.0,
        top_k: int = 50,
    ) -> torch.Tensor:
        for _ in range(max_new_tokens):
            ids = input_ids[:, -self.cfg.max_seq_len :]
            logits, _ = self.forward(ids)
            logits = logits[:, -1, :] / max(temperature, 1e-8)

            if top_k > 0:
                v, _ = logits.topk(min(top_k, logits.size(-1)))
                logits[logits < v[:, -1:]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_tok], dim=1)

        return input_ids

    @torch.no_grad()
    def generate_analytical_cycle(
        self, input_ids: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        persona_loop_map = {
            "Mythos-Glasswing": 1,
            "DepthFirst-DevSecOps": 3,
            "Cyber-Decompiler": 5,
            "BigSleep-Mimic": 7,
        }

        results = {}
        for name, loop_idx in persona_loop_map.items():
            idx = min(loop_idx, self.cfg.max_loop_iters - 1)
            logits, _ = self.forward(input_ids, section_idx=idx)
            results[name] = logits.argmax(dim=-1)

        return results


# =========================================================================
# SecCoreRunner — End-to-End Analysis Pipeline
# =========================================================================

class SecCoreRunner:
    def __init__(self, cfg: Optional[MythosConfig] = None, device: str = "cpu"):
        self.cfg = cfg or MythosConfig()
        self.device = torch.device(device)
        self.model = SECCoreUnified(self.cfg).to(self.device).eval()
        self.tokenizer = get_tokenizer(self.cfg.vocab_size)
        self._param_count = sum(p.numel() for p in self.model.parameters())

    def analyze(
        self, target_code: str, max_output_tokens: int = 64
    ) -> Dict[str, object]:
        token_ids = self.tokenizer.encode(target_code)
        token_ids = token_ids[: self.cfg.max_seq_len]
        input_ids = torch.tensor([token_ids], dtype=torch.long, device=self.device)

        logits, telemetry = self.model(input_ids)
        cycle = self.model.generate_analytical_cycle(input_ids)
        output_ids = self.model.generate(
            input_ids, max_new_tokens=max_output_tokens
        )
        generated_ids = output_ids[0, len(token_ids) :].tolist()
        generated_text = self.tokenizer.decode(generated_ids)

        probs = F.softmax(logits, dim=-1)
        entropy = -(probs * (probs + 1e-10).log()).sum(-1).mean().item()
        top1_conf = probs.max(dim=-1).values.mean().item()

        return {
            "telemetry": {
                "input_tokens": input_ids.shape[1],
                "output_entropy": round(entropy, 4),
                "top1_confidence": round(top1_conf, 4),
                "loop_depth": telemetry["loops_executed"],
                "halt_probs": [round(p, 4) for p in telemetry["halt_probs_per_loop"]],
                "active_tokens": telemetry["active_tokens_per_loop"],
                "final_cum_halt_p": round(telemetry["final_cum_p"], 4),
                "active_experts_per_token": self.cfg.k1 * self.cfg.k2,
                "total_experts": self.cfg.n_experts,
            },
            "per_persona": {
                name: ids[0].tolist() for name, ids in cycle.items()
            },
            "generated_text": generated_text,
            "model_params": self._param_count,
        }

    def format_output(self, results: Dict[str, object]) -> str:
        t = results["telemetry"]
        lines = [
            "",
            "======================================================================",
            "  SEC-CORE UNIFIED ORCHESTRATION NETWORK — ANALYSIS REPORT",
            "  Version 5.4 | Self-Contained | Zero API Dependencies",
            "======================================================================",
            "",
            "  EXECUTIVE TELEMETRY MATRIX",
            "  " + "-" * 50,
            f"  | Input Tokens             | {t['input_tokens']}",
            f"  | Recurrent Loop Depth     | {t['loop_depth']} iterations (all executed)",
            f"  | Output Entropy           | {t['output_entropy']}",
            f"  | Top-1 Confidence         | {t['top1_confidence']}",
            f"  | Active Experts/Token     | {t['active_experts_per_token']}",
            f"  | Total Expert Pool        | {t['total_experts']}",
            f"  | Model Parameters         | {results['model_params']:,}",
            f"  | Final Cumulative Halt P  | {t['final_cum_halt_p']}",
            "",
            "  HALTING TELEMETRY (per loop iteration)",
            "  " + "-" * 50,
        ]

        for i, (p, a) in enumerate(zip(t["halt_probs"], t["active_tokens"])):
            persona = COUNCIL_PERSONAS.get(i, "???")
            lines.append(
                f"  | Loop {i} [{persona:25s}] | halt_p={p:.4f}  active={a}"
            )

        lines += [
            "",
            "  QUAD-AGENT COUNCIL ROUTING",
            "  " + "-" * 50,
        ]

        for name, role in PERSONA_ROLES.items():
            tok_ids = results["per_persona"].get(name, [])
            lines.append(f"  [{name}]")
            lines.append(f"    Role: {role}")
            lines.append(f"    Output tokens: {len(tok_ids)} | First 8: {tok_ids[:8]}")
            lines.append("")

        lines += [
            "  RAW MODEL OUTPUT",
            "  " + "-" * 50,
            "  " + (results["generated_text"][:500] or "(empty)"),
            "",
            "======================================================================",
            "  STATUS: Architecture fully operational at maximum loop depth.",
            "  NOTE:   Model weights are randomly initialized.",
            "          Train with training/3b_fine_web_edu.py for real analysis.",
            "======================================================================",
            "",
        ]

        return "\n".join(lines)


# ==========================================
# HIGH-SPEED INFERENCE GATEWAY
# ==========================================

class InferenceGateway:
    def __init__(self, repo_index=None):
        self.repo_index = repo_index
        self.api_key = os.getenv("CYBER_TOP_API_KEY") or os.getenv("OPENAI_API_KEY")
        self.api_url = os.getenv("CYBER_TOP_API_URL") or "https://api.openai.com/v1/chat/completions"
        self.model_target = os.getenv("CYBER_TOP_MODEL") or "gpt-4o"
        self.use_live_api = self.api_key is not None
        self.runner = None

    async def execute_orchestration(self, target_payload: str) -> str:
        if self.use_live_api:
            return await self._call_live_api(target_payload)
        return await self._execute_local_rdt(target_payload)

    async def _call_live_api(self, payload: str) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        # Prepend repo context
        if self.repo_index:
             summary = f"Scanned {len(self.repo_index['signatures'])} files. Critical logic in: {list(self.repo_index['critical_logic'].keys())[:5]}"
             payload = f"[REPO_CONTEXT: {summary}]\n\n{payload}"

        data = {
            "model": self.model_target,
            "messages": [
                {"role": "system", "content": MASTER_SYSTEM_PROMPT},
                {"role": "user", "content": f"[TARGET INPUT FOR EVALUATION]:\n{payload}"}
            ],
            "temperature": 0.15
        }
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(self.api_url, headers=headers, json=data) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result['choices'][0]['message']['content']
                    return f"[❌ SERVER ERROR {response.status}]"
        except Exception as e:
            return f"[❌ CRITICAL INFRASTRUCTURE FAILURE]: {str(e)}"

    async def _execute_local_rdt(self, payload: str) -> str:
        # Detect if conversational
        if len(payload.split()) < 5 and any(kw in payload.lower() for kw in ["hi", "hello", "who", "help", "hey"]):
            return self._conversational_response(payload)

        if self.runner is None:
            # CPU friendly config for fast evaluation
            cfg = MythosConfig(
                dim=256,
                n_heads=4,
                n_kv_heads=1,
                n_experts=16,
                max_loop_iters=8,
                act_threshold=1.0,
            )
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.runner = SecCoreRunner(cfg, device=device)

        loop = asyncio.get_running_loop()
        results = await loop.run_in_executor(None, self.runner.analyze, payload, 32)
        return self.runner.format_output(results)

    def _conversational_response(self, payload: str) -> str:
        repo_files = list(self.repo_index['critical_logic'].keys())[:3] if self.repo_index else ["sec_core_unified.py"]
        return f"""
# SEC-CORE OPERATIONAL STATUS
I am the **SEC-CORE Orchestrator [v5.4-CYBER-TOP]**. I have scanned your repository and identified {len(self.repo_index['signatures']) if self.repo_index else 'multiple'} critical entry points.

I am currently monitoring:
- `{repo_files[0] if len(repo_files) > 0 else 'N/A'}`
- `{repo_files[1] if len(repo_files) > 1 else 'N/A'}`

**Operational Mode:** Active Surveillance.
**Awaiting:** Technical payload or code snippet for Quad-Agent Council analysis.

How can I assist in hardening your architecture today?
"""


# --- WEB UI & SERVER ---
HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SEC-CORE // ORCHESTRATOR 5.4</title>
    <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;700&family=Inter:wght@400;900&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg: #030305;
            --surface: rgba(15, 15, 25, 0.98);
            --neon: #00ffcc;
            --border: rgba(0, 255, 204, 0.2);
            --text: #a0a0b0;
            --text-bright: #ffffff;
            --glitch: #ff0055;
        }
        body {
            background-color: var(--bg); color: var(--text); font-family: 'Inter', sans-serif;
            margin: 0; padding: 0; height: 100vh; display: flex; flex-direction: column;
            background: radial-gradient(circle at center, #0a0a1a 0%, #030305 100%);
            overflow: hidden;
        }
        .chat-view { flex: 1; overflow-y: auto; padding: 40px; display: flex; flex-direction: column; gap: 30px; scroll-behavior: smooth; }
        .bubble { background: var(--surface); border: 1px solid var(--border); padding: 30px; border-radius: 16px; backdrop-filter: blur(20px); position: relative; animation: slideIn 0.4s ease-out; max-width: 85%; }
        @keyframes slideIn { from { opacity: 0; transform: translateY(20px); } to { opacity: 1; transform: translateY(0); } }
        .user-bubble { align-self: flex-end; border-color: rgba(255, 255, 255, 0.1); background: rgba(30, 30, 45, 0.5); }
        .telemetry { font-family: 'JetBrains Mono', monospace; font-size: 11px; color: var(--neon); margin-bottom: 15px; letter-spacing: 2px; text-transform: uppercase; border-bottom: 1px solid var(--border); padding-bottom: 5px; }
        .markdown { line-height: 1.6; }
        .markdown h1, .markdown h2, .markdown h3 { color: var(--neon); font-family: 'JetBrains Mono', monospace; margin-top: 25px; }
        .markdown table { border-collapse: collapse; width: 100%; margin: 20px 0; background: rgba(0,0,0,0.3); }
        .markdown th, .markdown td { border: 1px solid var(--border); padding: 12px; text-align: left; font-family: 'JetBrains Mono', monospace; }
        .markdown pre { background: #000; padding: 20px; border-radius: 8px; border: 1px solid var(--border); color: var(--neon); overflow-x: auto; font-family: 'JetBrains Mono', monospace; }
        .markdown code { color: var(--neon); font-family: 'JetBrains Mono', monospace; background: rgba(0,255,204,0.1); padding: 2px 5px; border-radius: 4px; }
        .input-bar { padding: 30px 40px; background: rgba(10, 10, 15, 0.9); border-top: 1px solid var(--border); display: flex; gap: 20px; backdrop-filter: blur(10px); }
        input { flex: 1; background: rgba(0,0,0,0.3); border: 1px solid var(--border); border-radius: 8px; padding: 18px; color: var(--neon); font-family: 'JetBrains Mono', monospace; outline: none; transition: 0.3s; }
        input:focus { border-color: var(--neon); box-shadow: 0 0 15px rgba(0,255,204,0.2); }
        button { padding: 0 40px; background: var(--neon); border: none; border-radius: 8px; font-weight: 900; cursor: pointer; color: #000; text-transform: uppercase; letter-spacing: 1px; transition: 0.3s; }
        button:hover { background: #fff; transform: scale(1.02); }
        .thinking-overlay { display: none; position: absolute; top: 0; left: 0; right: 0; bottom: 0; background: var(--surface); z-index: 10; padding: 30px; border-radius: 16px; flex-direction: column; justify-content: center; align-items: center; }
        .spinner { width: 40px; height: 40px; border: 3px solid var(--border); border-top-color: var(--neon); border-radius: 50%; animation: spin 1s linear infinite; margin-bottom: 15px; }
        @keyframes spin { to { transform: rotate(360deg); } }
    </style>
</head>
<body>
    <div class="chat-view" id="chat">
        <div class="bubble">
            <div class="telemetry">SEC-CORE // ORCHESTRATOR ONLINE [5.4-CYBER-TOP]</div>
            <div class="markdown">
                Welcome to the **Superior SEC-CORE Unified Orchestration Network**.
                Repo-scanning complete. Ready for Quad-Agent deep reasoning.

                *Awaiting input payload...*
            </div>
        </div>
    </div>
    <div class="input-bar">
        <input type="text" id="inp" placeholder="Ingest code or ask a question..." autocomplete="off">
        <button id="btn">Analyze</button>
    </div>
    <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
    <script>
        const chat = document.getElementById('chat');
        const inp = document.getElementById('inp');
        const btn = document.getElementById('btn');

        const addMessage = (content, isUser = false) => {
            const b = document.createElement('div');
            b.className = `bubble ${isUser ? 'user-bubble' : ''}`;
            if(!isUser) {
                b.innerHTML = `<div class="telemetry">STREAMS INBOUND</div><div class="markdown">${marked.parse(content)}</div>`;
            } else {
                b.innerText = content;
            }
            chat.appendChild(b);
            chat.scrollTop = chat.scrollHeight;
            return b;
        };

        btn.onclick = async () => {
            const v = inp.value.trim(); if(!v) return;
            inp.value = '';
            addMessage(v, true);

            const load = document.createElement('div');
            load.className = 'bubble';
            load.innerHTML = `<div class="telemetry">REASONING CYCLE START</div><div id="thinking-text" style="font-family:'JetBrains Mono'">[1/4] Intent Parsing...</div>`;
            chat.appendChild(load);
            chat.scrollTop = chat.scrollHeight;

            const thinkingText = load.querySelector('#thinking-text');
            const steps = [
                "[1.2/4] Ingesting Repo Context: sec_core_unified.py...",
                "[1.5/4] Lookahead Falsification: Branching state space...",
                "[2/4] Initializing Quad-Agent Council...",
                "[2.2/4] Mythos-Glasswing analyzing dependency graph...",
                "[2.5/4] Cyber-Decompiler mapping memory segments...",
                "[3/4] Running Zero-Day Discovery Loop...",
                "[3.5/4] MoDA Attention: Refining mitigation weights...",
                "[4/4] Synthesizing Strategic Coda..."
            ];

            let i = 0;
            const timer = setInterval(() => {
                if(i < steps.length) {
                    thinkingText.innerText = steps[i++];
                } else {
                    clearInterval(timer);
                }
            }, 300);

            try {
                const r = await fetch('/api/v1/analyze', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({payload: v})
                });
                const d = await r.json();
                clearInterval(timer);
                load.innerHTML = `<div class="telemetry">ANALYSIS COMPLETE</div><div class="markdown">${marked.parse(d.output)}</div>`;
            } catch(e) {
                clearInterval(timer);
                load.innerHTML = `<div class="telemetry">SYSTEM ERROR</div><div class="markdown">Critical failure in inference bridge.</div>`;
            }
            chat.scrollTop = chat.scrollHeight;
        };
        inp.onkeydown = (e) => { if(e.key === 'Enter') btn.onclick(); };
    </script>
</body>
</html>
"""

class Handler(BaseHTTPRequestHandler):
    GATEWAY = None
    def _h(self, ct='text/html'):
        self.send_response(200)
        self.send_header('Content-type', ct)
        self.end_headers()
    def do_GET(self):
        self._h()
        self.wfile.write(HTML.encode())
    def do_POST(self):
        if self.path == '/api/v1/analyze':
            cl = int(self.headers['Content-Length'])
            data = json.loads(self.rfile.read(cl))
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            res = loop.run_until_complete(self.GATEWAY.execute_orchestration(data['payload']))
            self._h('application/json')
            self.wfile.write(json.dumps({"output": res}).encode())

async def cli_main(gateway, target_path):
    try:
        with open(target_path, "r", encoding="utf-8") as f:
            payload = f.read()
    except Exception as e:
        print(f"Failed to read target file: {e}")
        return

    print("\n[>>] INITIALIZING SEC-CORE UNIFIED ORCHESTRATION NETWORK - VERSION 5.4")
    print(f"[**] CONFIG: dim={gateway.runner.cfg.dim if gateway.runner else 256}")
    print("[>>] RUNNING: Quad-Agent Debate & Zero-Day Discovery Loop engaged...")

    output = await gateway.execute_orchestration(payload)
    print("\n" + "="*60 + "\n" + output + "\n" + "="*60 + "\n")

def run():
    scanner = RepoScanner()
    index = scanner.scan()
    gateway = InferenceGateway(index)

    if len(sys.argv) > 1 and os.path.isfile(sys.argv[1]):
        asyncio.run(cli_main(gateway, sys.argv[1]))
    else:
        Handler.GATEWAY = gateway
        port = int(os.environ.get("PORT", 7860))
        server = HTTPServer(('0.0.0.0', port), Handler)
        print(f"SEC-CORE 5.4 CHATBOT ACTIVE ON PORT {port}")
        server.serve_forever()

if __name__ == "__main__":
    run()
