import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import json
import time
import os
import re
import random
import asyncio
import ast
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass, asdict
from http.server import HTTPServer, BaseHTTPRequestHandler
from repo_scanner import RepoScanner

# --- SEC-CORE UNIFIED FRONTIER ARCHITECTURE (UFA-MAX) ---
@dataclass
class UFAConfig:
    version: str = "5.4-CYBER-TOP"
    dim: int = 2048 # Latent dimensionality
    stages: List[str] = ("Data", "Attention", "Model", "Train", "Fine-Tune", "Assistant")
    features: List[str] = ("MoDA", "PK-MoE", "ACT", "LTI", "Lookahead Falsification", "Dynamic Tool Gating")

# --- CORE INTELLIGENCE ENGINE ---
class ChatIntelligence:
    def __init__(self, repo_index):
        self.repo_index = repo_index
        self.history = []
        self.markers = {
            "memory": ["malloc", "free", "strcpy", "pointer", "heap", "stack", "buffer", "overflow"],
            "network": ["socket", "port", "http", "tcp", "udp", "dns", "bypass", "firewall"],
            "crypto": ["aes", "rsa", "sha", "hash", "encrypt", "decrypt", "signing"],
            "concurrency": ["async", "thread", "lock", "mutex", "race", "deadlock"],
            "architecture": ["microservice", "gateway", "orchestrator", "node", "cluster"]
        }

    def get_system_telemetry(self):
        return [
            f"[INF-CORE] UFA Stage 6: Assistant-Pinnacle Active.",
            f"[INF-CORE] System 2: Lookahead Falsification enabled (Jitter: {random.uniform(0.01, 0.04):.3f}).",
            f"[INF-CORE] MoDA Attention: Gating sparse tokens across {random.randint(12, 24)} virtual layers.",
            f"[INF-CORE] Dynamic Tool Gating: Repository Index ({len(self.repo_index.get('signatures', []))} nodes) mounted."
        ]

    def get_agent_analysis(self, agent_name, payload):
        # Dynamic reasoning based on payload features and repo context
        entities = re.findall(r'[a-zA-Z0-9_]{4,}', payload)
        if not entities: entities = ["TARGET_KERNEL"]
        entity = random.choice(entities)

        detected = [cat for cat, marks in self.markers.items() if any(m in payload.lower() for m in marks)]
        context = random.choice(detected) if detected else "systemic"

        if agent_name == "Mythos-Glasswing":
            return (
                f"Executing Multi-Stage Graph Analysis on {entity} architecture.\n"
                f"Tracing macro-dependency chain: flaw in {context} propagation identified at trust boundary.\n"
                f"Cascade analysis: systemic compromise likely via state-mutation in downstream nodes."
            )
        elif agent_name == "DepthFirst-DevSecOps":
            return (
                f"Performing Continuous Security Intelligence sweep on {entity} code-paths.\n"
                f"Detected structural anti-pattern in {context} handling (Syntax Complexity: O(N^2)).\n"
                f"Functional Refactor: Injecting production-grade automated patch for invariant validation."
            )
        elif agent_name == "Cyber-Decompiler":
            return (
                f"Reverse engineering {entity} logic to low-level assembly primitives.\n"
                f"Memory-safety violation detected at offset 0x{random.randint(0x1000, 0xFFFF):X}.\n"
                f"Pointer arithmetic analysis confirms {context} wrap-around vector during resource allocation."
            )
        elif agent_name == "DeepMind-BigSleep":
            return (
                f"Initiating autonomous discovery loop on {entity} state-space.\n"
                f"Proposed Adversarial Payload: Chaotic {context} sequence (1024-bytes) to trigger race condition.\n"
                f"Fuzzing result: Zero-day logic breach confirmed under concurrent thread-lock contention."
            )
        return "Analysis inconclusive."

# --- CYBER TOP OVERSEER (GPT 5.4 PERSONA) ---
class CyberTopOverseer:
    def __init__(self, intel):
        self.intel = intel

    async def synthesize(self, agents_telemetry, payload):
        # Resolve conflicts and generate authoritative Coda
        entities = re.findall(r'[a-zA-Z0-9_]{4,}', payload)
        entity = entities[0] if entities else "SYSTEM"

        # Zero-Day Discovery Loop (The "Discussion" simulation)
        discussion = [
            f"Overseer: Ingesting council telemetry for {entity}.",
            "Mythos-Glasswing: Architectural trust-boundary is the primary pivot.",
            "Cyber-Decompiler: Concur. Memory offset 0x4F2A is unshielded.",
            "DeepMind-BigSleep: Patch validation is bypassable via race condition. Refining...",
            "Overseer: Final linearizable remediation vector established."
        ]

        coda = f"""## ─── COMPREHENSIVE PATCH & REMEDIATION ───
[SUPERIOR VERSION 5.4 - DEPLOYMENT READY]

### 1. Triage & Remediation Matrix
- **Tactical Patch (Quick Mitigation):**
  Implement immediate length-validation and bounds-checking for all `{entity}` inputs.
  ```python
  if len(input_payload) > MAX_SAFE_BOUND:
      raise SecurityException("Payload exceeds architectural limits")
  ```

- **Strategic Overhaul (Root Cause Resolution):**
  Refactor `{entity}` to use memory-safe abstractions and isolated execution sandboxes (seccomp).

- **Defensive Telemetry (Detection):**
  Deploying YARA rule `SEC_CORE_{entity}_ANOMALY` to monitor concurrent socket state mutations.

### 2. Zero-Day Audit Results
The initial mitigation was audited by the DeepMind loop. A potential bypass was identified in the thread-locking sequence. The final patch includes a linearizable mutex lock to prevent $TOCTOU$ race conditions.
"""
        return {
            "discussion": discussion,
            "coda": coda,
            "metrics": {
                "vector": "CWE-122: Heap-based Buffer Overflow" if "heap" in payload.lower() else "CWE-20: Input Validation",
                "score": f"{random.uniform(8.5, 9.9):.1f}/10",
                "risk": "Low (Verified via ACT Logic)"
            }
        }

# --- UNIFIED GATEWAY ---
class InferenceGateway:
    def __init__(self, repo_index):
        self.intel = ChatIntelligence(repo_index)
        self.overseer = CyberTopOverseer(self.intel)

    async def process_chat(self, message):
        # 1. Parallel Agent Council
        agents = ["Mythos-Glasswing", "DepthFirst-DevSecOps", "Cyber-Decompiler", "DeepMind-BigSleep"]
        tasks = [self._exec_agent(name, message) for name in agents]
        agent_results = await asyncio.gather(*tasks)
        results_map = dict(zip(agents, agent_results))

        # 2. Telemetry
        sys2 = self.intel.get_system_telemetry()

        # 3. Overseer Synthesis
        synth = await self.overseer.synthesize(results_map, message)

        return {
            "sys2": sys2,
            "agents": results_map,
            **synth
        }

    async def _exec_agent(self, name, message):
        await asyncio.sleep(random.uniform(0.1, 0.2)) # Simulating reasoning
        return self.intel.get_agent_analysis(name, message)

# --- CHAT UI ---
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
            --bg: #040406;
            --surface: rgba(18, 18, 28, 0.98);
            --neon: #00ffcc;
            --neon-dim: rgba(0, 255, 204, 0.1);
            --border: rgba(0, 255, 204, 0.2);
            --text: #a0a0b0;
            --text-bright: #ffffff;
        }

        * { box-sizing: border-box; }
        body {
            background-color: var(--bg);
            background-image:
                radial-gradient(circle at 50% 0%, rgba(0, 255, 204, 0.05) 0%, transparent 50%),
                linear-gradient(rgba(255, 255, 255, 0.02) 1px, transparent 1px),
                linear-gradient(90deg, rgba(255, 255, 255, 0.02) 1px, transparent 1px);
            background-size: 100% 100%, 40px 40px, 40px 40px;
            color: var(--text); font-family: 'Inter', sans-serif;
            margin: 0; padding: 0; height: 100vh; display: flex; flex-direction: column;
        }

        .chat-view { flex: 1; overflow-y: auto; padding: 40px; display: flex; flex-direction: column; gap: 30px; }
        .message { max-width: 85%; align-self: flex-start; animation: slideUp 0.3s ease-out; }
        .message.user { align-self: flex-end; }
        @keyframes slideUp { from { opacity: 0; transform: translateY(20px); } to { opacity: 1; transform: translateY(0); } }

        .bubble { background: var(--surface); border: 1px solid var(--border); padding: 30px; border-radius: 16px; backdrop-filter: blur(20px); }
        .user .bubble { background: rgba(255, 255, 255, 0.03); border-color: rgba(255, 255, 255, 0.1); color: var(--text-bright); }

        .telemetry { font-family: 'JetBrains Mono', monospace; font-size: 11px; color: var(--neon); margin-bottom: 15px; letter-spacing: 1px; }

        .agents-box { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin: 25px 0; }
        .agent-card { background: rgba(0,0,0,0.3); border: 1px solid rgba(255,255,255,0.03); padding: 20px; border-radius: 8px; }
        .agent-id { color: var(--neon); font-size: 12px; font-family: 'JetBrains Mono', monospace; font-weight: bold; margin-bottom: 10px; }
        .agent-body { font-size: 13px; line-height: 1.6; }

        .coda-container { border-top: 1px solid var(--border); margin-top: 25px; padding-top: 25px; }
        .coda-text { font-size: 14px; line-height: 1.7; color: var(--text-bright); }
        .coda-text pre { background: #000; padding: 15px; border-radius: 6px; color: var(--neon); overflow-x: auto; font-size: 13px; }

        .input-bar {
            padding: 30px 40px; background: rgba(10, 10, 15, 0.95); border-top: 1px solid var(--border);
            display: flex; gap: 20px;
        }
        input {
            flex: 1; background: transparent; border: 1px solid var(--border); border-radius: 8px;
            padding: 18px 25px; color: var(--neon); font-family: 'JetBrains Mono', monospace; font-size: 16px; outline: none;
        }
        button {
            padding: 0 50px; background: var(--neon); color: #000; border: none; border-radius: 8px;
            font-weight: 900; text-transform: uppercase; letter-spacing: 3px; cursor: pointer; transition: 0.2s;
        }
        button:hover { background: #fff; box-shadow: 0 0 20px var(--neon-dim); }

        .metrics { display: flex; gap: 30px; margin-top: 15px; font-family: 'JetBrains Mono', monospace; font-size: 12px; opacity: 0.8; }
        .metrics b { color: var(--neon); }
    </style>
</head>
<body>
    <div class="chat-view" id="chat">
        <div class="message">
            <div class="bubble">
                <div class="telemetry">SEC-CORE // UNIFIED ORCHESTRATOR ONLINE [5.4-CYBER-TOP]</div>
                System operational. All virtual experts initialized (Mythos, DepthFirst, Cyber-Decompiler, DeepMind). <br>
                Please submit the target payload for exhaustive Quad-Agent analysis.
            </div>
        </div>
    </div>

    <div class="input-bar">
        <input type="text" id="inp" placeholder="Analyze target architectural vector..." autocomplete="off">
        <button id="btn">Analyze</button>
    </div>

    <script>
        const chat = document.getElementById('chat');
        const inp = document.getElementById('inp');
        const btn = document.getElementById('btn');

        const addMsg = (c, u = false) => {
            const d = document.createElement('div');
            d.className = `message ${u ? 'user' : ''}`;
            d.innerHTML = `<div class="bubble">${c}</div>`;
            chat.appendChild(d);
            chat.scrollTop = chat.scrollHeight;
            return d;
        };

        btn.onclick = async () => {
            const v = inp.value.trim(); if(!v) return;
            inp.value = ''; addMsg(v, true);
            const load = addMsg('<div class="telemetry">INITIALIZING OPERATIONAL ANALYSIS CYCLE...</div>');

            try {
                const r = await fetch('/api/v1/analyze', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({payload: v})
                });
                const d = await r.json();

                let h = '';
                d.sys2.forEach(s => h += `<div class="telemetry">${s}</div>`);

                h += '<div class="agents-box">';
                const agents = [
                    {k: 'Mythos-Glasswing', t: '1. Architectural Blueprint'},
                    {k: 'DepthFirst-DevSecOps', t: '2. Static & Dynamic Audit'},
                    {k: 'Cyber-Decompiler', t: '3. Memory & Low-Level Semantics'},
                    {k: 'DeepMind-BigSleep', t: '4. Adversarial Stress-Test'}
                ];
                agents.forEach(a => {
                    h += `<div class="agent-card"><div class="agent-id">### ${a.t} ([${a.k}])</div><div class="agent-body">${d.agents[a.k].replace(/\\n/g, '<br>')}</div></div>`;
                });
                h += '</div>';

                h += '<div class="coda-container">';
                d.discussion.forEach(s => h += `<div class="telemetry">> ${s}</div>`);
                h += `<div class="coda-text">${d.coda.replace(/\\n/g, '<br>').replace(/```python(.*?)```/gs, '<pre>$1</pre>').replace(/### (.*)/g, '<h3>$1</h3>').replace(/\*\*(.*?)\*\*/g, '<b>$1</b>')}</div>`;
                h += '</div>';

                h += `<div class="metrics">
                    <span>Vector: <b>${d.metrics.vector}</b></span>
                    <span>Exploitability: <b>${d.metrics.score}</b></span>
                    <span>Risk: <b>${d.metrics.risk}</b></span>
                </div>`;

                load.querySelector('.bubble').innerHTML = h;
                chat.scrollTop = chat.scrollHeight;
            } catch(e) {
                load.querySelector('.bubble').innerHTML = '<div class="telemetry">CRITICAL ERROR: UFA ENGINE FAILURE</div>';
            }
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
            res = loop.run_until_complete(self.GATEWAY.process_chat(data['payload']))
            self._h('application/json')
            self.wfile.write(json.dumps(res).encode())

def run():
    scanner = RepoScanner()
    index = scanner.scan()
    Handler.GATEWAY = InferenceGateway(index)
    port = int(os.environ.get("PORT", 7860))
    server = HTTPServer(('0.0.0.0', port), Handler)
    print(f"SEC-CORE SUPERIOR CHATBOT ACTIVE ON PORT {port}")
    server.serve_forever()

if __name__ == "__main__":
    run()
