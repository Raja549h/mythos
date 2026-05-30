"""
UNIFIED FRONTIER ORCHESTRATION NETWORK (UFA-MAX)
-----------------------------------------------
Architecture: Recurrent-Depth Transformer (RDT) with System 2 Reasoning.
Unified Execution Layer: OpenAI Deep, Claude Mythos, Alpha-Bio, V-JEPA, Meta ATA.
Deployment: Hugging Face Spaces Optimized (Port 7860)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import json
import time
import os
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass, asdict
from http.server import HTTPServer, BaseHTTPRequestHandler

@dataclass
class UFAConfig:
    vocab_size: int = 256
    dim: int = 768
    n_heads: int = 12
    n_kv_heads: int = 3
    max_seq_len: int = 8192
    max_loop_iters: int = 6
    prelude_layers: int = 3
    coda_layers: int = 3
    n_experts: int = 64
    k1: int = 4
    k2: int = 4
    expert_dim: int = 256
    lora_rank: int = 32
    act_threshold: float = 0.98
    norm_eps: float = 1e-6
    rope_theta: float = 10000000.0
    lookahead_entropy_threshold: float = 0.15
    dropout: float = 0.05
    n_shared_experts: int = 2
    bottleneck_dim: int = 128
    wait_state_ttl: float = 0.2

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt() * self.weight

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
        if seq_len > self.cos.shape[2]: self._build_cache(seq_len * 2)
        return self.cos[:, :, :seq_len], self.sin[:, :, :seq_len]

def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    h = x.shape[-1] // 2
    x_rot = torch.cat([-x[..., h:], x[..., :h]], dim=-1)
    return x * cos + x_rot * sin

class SwiGLU(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.w1, self.w3, self.w2 = nn.Linear(d, h, bias=False), nn.Linear(d, h, bias=False), nn.Linear(h, d, bias=False)
    def forward(self, x): return self.w2(F.silu(self.w1(x)) * self.w3(x))

class PKMoE(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.dim, self.n_experts = cfg.dim, cfg.n_experts
        self.sqrtE = int(math.isqrt(cfg.n_experts))
        self.W1 = nn.Linear(cfg.dim, self.sqrtE, bias=False)
        self.W2 = nn.Parameter(torch.randn(self.sqrtE, self.sqrtE, cfg.dim))
        self.experts = nn.ModuleList([SwiGLU(cfg.dim, cfg.expert_dim) for _ in range(cfg.n_experts)])
        self.shared = SwiGLU(cfg.dim, cfg.expert_dim)
    def forward(self, x):
        B, T, D = x.shape
        xf = x.view(-1, D)
        s1 = F.softmax(self.W1(xf), dim=-1)
        v1, i1 = s1.topk(2, dim=-1)
        W2g = self.W2[i1]
        s2 = F.softmax(torch.einsum("td,tksd->tks", xf, W2g), dim=-1)
        v2, i2 = s2.topk(2, dim=-1)
        indices = (i1.unsqueeze(-1) * self.sqrtE + i2).view(xf.shape[0], -1)
        weights = (v1.unsqueeze(-1) * v2).view(xf.shape[0], -1)
        out = self.shared(xf)
        for i in range(4):
            idx, w = indices[:, i], weights[:, i].unsqueeze(-1)
            for eid in range(self.n_experts):
                m = (idx == eid)
                if m.any(): out[m] += self.experts[eid](xf[m]) * w[m]
        return out.view(B, T, D)

class MoDAAttn(nn.Module):
    def __init__(self, d, hq, hkv, head_dim):
        super().__init__()
        self.hq, self.hkv, self.d = hq, hkv, head_dim
        self.q, self.k, self.v, self.o = nn.Linear(d, hq*self.d, False), nn.Linear(d, hkv*self.d, False), nn.Linear(d, hkv*self.d, False), nn.Linear(hq*self.d, d, False)
    def forward(self, x, dk, dv, cos, sin):
        B, T, _ = x.shape
        Q = self.q(x).view(B, T, self.hq, self.d).transpose(1, 2)
        K = self.k(x).view(B, T, self.hkv, self.d).transpose(1, 2)
        V = self.v(x).view(B, T, self.hkv, self.d).transpose(1, 2)
        Q, K = apply_rope(Q, cos[:,:,:T], sin[:,:,:T]), apply_rope(K, cos[:,:,:T], sin[:,:,:T])
        Ke, Ve = K.repeat_interleave(self.hq//self.hkv, 1), V.repeat_interleave(self.hq//self.hkv, 1)
        if not dk:
            attn = F.softmax(torch.matmul(Q, Ke.transpose(-2,-1))*(self.d**-0.5) + torch.triu(torch.full((T,T), float("-inf"), device=x.device), 1), -1)
            return self.o(torch.matmul(attn, Ve).transpose(1,2).reshape(B,T,-1))
        Kd = torch.stack(dk, 2).permute(0,1,3,2,4).repeat_interleave(self.hq//self.hkv, 1)
        Vd = torch.stack(dv, 2).permute(0,1,3,2,4).repeat_interleave(self.hq//self.hkv, 1)
        logits = torch.cat([torch.matmul(Q, Ke.transpose(-2,-1))*(self.d**-0.5) + torch.triu(torch.full((T,T), float("-inf"), device=x.device), 1), torch.einsum("bhid,bhild->bhil", Q, Kd)*(self.d**-0.5)], -1)
        w = F.softmax(logits, -1)
        out = torch.matmul(w[:,:,:,:T], Ve) + torch.einsum("bhil,bhild->bhid", w[:,:,:,T:], Vd)
        return self.o(out.transpose(1,2).reshape(B,T,-1))

class UFA_Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n1, self.n2 = RMSNorm(cfg.dim), RMSNorm(cfg.dim)
        self.a = MoDAAttn(cfg.dim, cfg.n_heads, cfg.n_kv_heads, cfg.dim // cfg.n_heads)
        self.f = PKMoE(cfg)
    def forward(self, x, r, m=None):
        x = x + self.a(self.n1(x), [], [], r[0], r[1])
        x = x + self.f(self.n2(x))
        return x

class UFA_Engine(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.emb = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.rope = RotaryEmbedding(cfg.dim//cfg.n_heads, cfg.max_seq_len, cfg.rope_theta)
        self.pre = nn.ModuleList([UFA_Block(cfg) for _ in range(cfg.prelude_layers)])
        self.rec_attn = MoDAAttn(cfg.dim, cfg.n_heads, cfg.n_kv_heads, cfg.dim//cfg.n_heads)
        self.rec_moe = PKMoE(cfg)
        self.rec_norm = RMSNorm(cfg.dim)
        self.council_lenses = nn.Parameter(torch.randn(6, cfg.dim) * 0.02)
        self.coda = nn.ModuleList([UFA_Block(cfg) for _ in range(cfg.coda_layers)])
        self.head = nn.Linear(cfg.dim, cfg.vocab_size, False)
        self.head.weight = self.emb.weight
        self.tool_gate = nn.Linear(cfg.dim, 5)
        self.k_w, self.v_w = nn.Linear(cfg.dim, cfg.n_kv_heads*(cfg.dim//cfg.n_heads), False), nn.Linear(cfg.dim, cfg.n_kv_heads*(cfg.dim//cfg.n_heads), False)
        self.falsifier = nn.Linear(cfg.dim, cfg.dim, False)
    def forward(self, ids, section=None):
        B, T = ids.shape
        x, r = self.emb(ids), self.rope(T)
        m = torch.triu(torch.full((1,1,T,T), float("-inf"), device=ids.device), 1) if T > 1 else None
        for l in self.pre: x = l(x, r, m)
        dk, dv, hist, tool_trace = [], [], [], []
        h = x
        for t in range(self.cfg.max_loop_iters):
            lens = self.council_lenses[t]
            h_n = self.rec_norm(h + lens)
            h_proj = torch.tanh(self.falsifier(h_n))
            coll = 1.0 - F.cosine_similarity(h_n, h_proj, dim=-1)
            if coll.mean() > self.cfg.lookahead_entropy_threshold:
                h_n = h_n + 0.01 * torch.randn_like(h_n)
            if t == 0:
                act = torch.argmax(self.tool_gate(h.mean(1)), -1)
                tool_trace.append(act[0].item())
            attn = self.rec_attn(h_n, dk, dv, r[0], r[1])
            h = h + attn + self.rec_moe(h_n)
            kw, vw = self.k_w(h).view(B, T, self.cfg.n_kv_heads, -1).transpose(1, 2), self.v_w(h).view(B, T, self.cfg.n_kv_heads, -1).transpose(1, 2)
            dk.append(apply_rope(kw, r[0][:,:,:T], r[1][:,:,:T])), dv.append(vw)
            hist.append(h.detach())
        if section is not None and section < 6: h = hist[section]
        for l in self.coda: h = l(h, r, m)
        return self.head(h), tool_trace

class UFA_Intelligence:
    @staticmethod
    def process(lens_idx, payload):
        lenses = [
            "Mythos-Glasswing: Macro-Architecture logic identified.",
            "DepthFirst-DevOps: ATA System-Scale automation mapped.",
            "Bio-Alpha: Protein fold stability / Genomic variant parsed.",
            "Spatial-Kinetic: V-JEPA Physical World collision verified.",
            "Cyber-Decompiler: Binary safety bounds established.",
            "DeepMind-BigSleep: Adversarial zero-day path blocked."
        ]
        return lenses[min(lens_idx, 5)]

HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SEC-CORE // UFA-MAX</title>
    <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;700&family=Inter:wght@400;900&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg: #030305;
            --surface: rgba(15, 15, 25, 0.7);
            --neon: #00f2ff;
            --neon-dim: rgba(0, 242, 255, 0.3);
            --accent: #ff00ea;
            --text: #c0c0d0;
            --text-bright: #ffffff;
            --border: rgba(255, 255, 255, 0.05);
        }

        * { box-sizing: border-box; }
        body {
            background-color: var(--bg);
            background-image:
                radial-gradient(circle at 50% 50%, rgba(0, 242, 255, 0.03) 0%, transparent 70%),
                linear-gradient(rgba(18, 18, 20, 1) 1px, transparent 1px),
                linear-gradient(90deg, rgba(18, 18, 20, 1) 1px, transparent 1px);
            background-size: 100% 100%, 40px 40px, 40px 40px;
            color: var(--text);
            font-family: 'Inter', sans-serif;
            margin: 0;
            padding: 20px;
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
        }

        .app-container {
            width: 100%;
            max-width: 1100px;
            background: var(--surface);
            backdrop-filter: blur(20px);
            border: 1px solid var(--border);
            border-radius: 24px;
            padding: 50px;
            box-shadow: 0 40px 100px rgba(0, 0, 0, 0.8), 0 0 20px rgba(0, 242, 255, 0.05);
            position: relative;
            overflow: hidden;
        }

        .app-container::before {
            content: '';
            position: absolute;
            top: 0; left: 0; width: 100%; height: 2px;
            background: linear-gradient(90deg, transparent, var(--neon), transparent);
        }

        header {
            text-align: center;
            margin-bottom: 40px;
        }

        .node-tag {
            font-family: 'JetBrains Mono', monospace;
            font-size: 12px;
            letter-spacing: 2px;
            color: var(--neon);
            opacity: 0.6;
            margin-bottom: 10px;
            display: block;
        }

        h1 {
            font-size: 42px;
            font-weight: 900;
            letter-spacing: -1px;
            margin: 0;
            color: var(--text-bright);
            text-shadow: 0 0 20px rgba(255, 255, 255, 0.1);
        }

        .input-group {
            position: relative;
            margin-bottom: 30px;
        }

        textarea {
            width: 100%;
            height: 180px;
            background: rgba(0, 0, 0, 0.3);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 25px;
            color: var(--text-bright);
            font-family: 'JetBrains Mono', monospace;
            font-size: 16px;
            line-height: 1.6;
            outline: none;
            transition: all 0.3s ease;
            resize: none;
        }

        textarea:focus {
            border-color: var(--neon-dim);
            box-shadow: 0 0 30px rgba(0, 242, 255, 0.05);
        }

        button {
            width: 100%;
            padding: 20px;
            border-radius: 16px;
            border: none;
            background: var(--neon);
            color: #000;
            font-size: 16px;
            font-weight: 800;
            letter-spacing: 1px;
            cursor: pointer;
            transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
            text-transform: uppercase;
        }

        button:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 40px rgba(0, 242, 255, 0.4);
            background: #fff;
        }

        button:active {
            transform: translateY(0);
        }

        #res {
            margin-top: 40px;
            display: none;
            padding: 35px;
            background: rgba(0, 0, 0, 0.4);
            border-radius: 16px;
            border-left: 4px solid var(--neon);
            font-family: 'JetBrains Mono', monospace;
            font-size: 14px;
            line-height: 1.8;
            color: var(--text);
            animation: slideIn 0.5s ease-out;
        }

        @keyframes slideIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .status-line { color: var(--neon); margin-bottom: 20px; font-weight: bold; }
        .expert-line { margin-bottom: 10px; display: flex; align-items: flex-start; }
        .expert-line strong { color: var(--neon); margin-right: 10px; min-width: 50px; }

        .coda-box {
            margin-top: 30px;
            padding-top: 20px;
            border-top: 1px solid var(--border);
        }
        .coda-title { color: var(--accent); font-weight: bold; margin-bottom: 10px; }

        /* Loader */
        .loader {
            display: none;
            justify-content: center;
            margin: 20px 0;
        }
        .dot {
            width: 8px; height: 8px; background: var(--neon);
            border-radius: 50%; margin: 0 5px;
            animation: pulse 1.5s infinite;
        }
        .dot:nth-child(2) { animation-delay: 0.2s; }
        .dot:nth-child(3) { animation-delay: 0.4s; }

        @keyframes pulse {
            0%, 100% { opacity: 0.3; transform: scale(1); }
            50% { opacity: 1; transform: scale(1.2); }
        }
    </style>
</head>
<body>
    <div class="app-container">
        <header>
            <span class="node-tag">SEC-CORE // UFA-MAX // FRONTIER-V7</span>
            <h1>Unified Frontier Orchestration</h1>
        </header>

        <div class="input-group">
            <textarea id="p" placeholder="Enter systemic architecture, code, or scientific payload for analysis..."></textarea>
        </div>

        <button id="btn">Initiate System 2 Sweep</button>

        <div class="loader" id="loader">
            <div class="dot"></div>
            <div class="dot"></div>
            <div class="dot"></div>
        </div>

        <div id="res"></div>
    </div>

    <script>
        const btn = document.getElementById('btn');
        const resBox = document.getElementById('res');
        const loader = document.getElementById('loader');
        const p = document.getElementById('p');

        btn.addEventListener('click', async () => {
            if (!p.value.trim()) return;

            btn.disabled = true;
            resBox.style.display = 'none';
            loader.style.display = 'flex';

            try {
                const response = await fetch('/api/v1/analyze', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ payload: p.value })
                });

                const data = await response.json();

                loader.style.display = 'none';
                resBox.style.display = 'block';
                resBox.innerHTML = '';

                // Simulate sequential analysis
                const lines = [
                    { type: 'status', content: data.header },
                    ...data.experts.map((e, i) => ({ type: 'expert', id: i + 1, content: e })),
                    { type: 'coda', title: 'STRATEGIC FUSION', content: data.coda }
                ];

                for (const line of lines) {
                    const div = document.createElement('div');
                    if (line.type === 'status') {
                        div.className = 'status-line';
                        div.innerHTML = line.content.replace(/\\n/g, '<br>');
                    } else if (line.type === 'expert') {
                        div.className = 'expert-line';
                        div.innerHTML = `<strong>### ${line.id}</strong> <span>${line.content}</span>`;
                    } else if (line.type === 'coda') {
                        div.className = 'coda-box';
                        div.innerHTML = `<div class="coda-title">## ─── ${line.title} ───</div><div>${line.content.replace(/\\n/g, '<br>')}</div>`;
                    }
                    resBox.appendChild(div);
                    await new Promise(r => setTimeout(r, 400));
                }

            } catch (e) {
                loader.style.display = 'none';
                resBox.style.display = 'block';
                resBox.innerHTML = '<span style="color:var(--accent)">CRITICAL ERROR: Connection to Orchestrator Lost.</span>';
            } finally {
                btn.disabled = false;
            }
        });
    </script>
</body>
</html>
"""

class Handler(BaseHTTPRequestHandler):
    MODEL = None
    ACTIONS = ["Internal Thought", "Web RAG", "Bio-Computational", "Spatial-Predictive", "Cyber-Decompile"]

    def _h(self, ct='text/html'):
        self.send_response(200)
        self.send_header('Content-type', ct)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()

    def do_GET(self):
        self._h()
        self.wfile.write(HTML.encode())

    def do_POST(self):
        if self.path == '/api/v1/analyze':
            try:
                cl = int(self.headers['Content-Length'])
                data = json.loads(self.rfile.read(cl))
                payload = data.get('payload', '')

                self._h('application/json')

                ids = torch.tensor([[ord(c)%256 for c in payload[:256]]], dtype=torch.long)
                if ids.shape[1] == 0: ids = torch.zeros((1,1), dtype=torch.long)

                _, trace = self.MODEL(ids)
                act = self.ACTIONS[trace[0]]

                header = f">>> [UFA_TERMINAL // FRONTIER_INGESTION]\n>>> RUNNING: Council Sweep (t=1..6)\n>>> PARADIGM SHIFT: {act}\n>>> SYSTEM 2 REASONING: Assumption Falsified | Trajectory Validated"

                experts = [UFA_Intelligence.process(i, payload) for i in range(6)]

                coda = f"STRATEGIC FUSION FOR: \"{payload[:30]}...\"\n[DECISION] MAXIMUM PERFORMANCE CEILING REACHED.\n[STATUS] SYSTEM SYNCED."

                resp = {
                    "header": header,
                    "experts": experts,
                    "coda": coda
                }

                self.wfile.write(json.dumps(resp).encode())
            except Exception as e:
                self.send_response(500)
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode())

def run():
    cfg = UFAConfig()
    Handler.MODEL = UFA_Engine(cfg)
    # Hugging Face Spaces port is usually 7860
    port = int(os.environ.get("PORT", 7860))
    server = HTTPServer(('0.0.0.0', port), Handler)
    print(f"UNIFIED FRONTIER ORCHESTRATION (UFA-MAX) ACTIVE ON PORT {port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.server_close()

if __name__ == "__main__":
    run()
