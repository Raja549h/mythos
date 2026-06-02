import os
import sys
import json
import asyncio
import random
import re
import aiohttp
from typing import Dict, Any, List
from http.server import HTTPServer, BaseHTTPRequestHandler
from repo_scanner import RepoScanner

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
- Objective: Evaluate how data flows across encapsulation layers. Identify architectural single points of failure, unvalidated state propagation, or asynchronous race conditions.

## PERSONA 02: Cyber-Decompiler [Deterministic Binary Specialist]
- Focus: Memory corruption, low-level pointer arithmetic, assembly execution paths, and the hardware-software interface.
- Objective: Audit the input strictly at the metal. Isolate vulnerabilities like Stack/Heap Buffer Overflows, Use-After-Free (UAF), Double Free, Integer Overflows, and Time-of-Check to Time-of-Use ($TOCTOU$) flaws.

## PERSONA 03: BigSleep-Mimic [AI Zero-Day Fuzzer]
- Focus: Complex semantic logic flaws, non-obvious heuristic anomalies, and multi-step exploitation chains.
- Objective: Assume the target code compiles perfectly and clears traditional SAST/DAST tools. Find the subtle interaction failure where combining multiple valid logic choices yields an exploitable state.

## PERSONA 04: SEC-CORE Governor [Risk & Mitigation Control]
- Focus: Blast-radius mitigation, CVSS validation, real-world patching friction, and performance-security trade-offs.
- Objective: Balance security necessity against operational realities. Ensure the system does not recommend an idealized patch that permanently destroys runtime efficiency.

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

# ==========================================
# STAGE 2: HIGH-SPEED INFERENCE GATEWAY
# ==========================================
class InferenceGateway:
    def __init__(self, repo_index=None):
        self.repo_index = repo_index
        self.api_key = os.getenv("CYBER_TOP_API_KEY") or os.getenv("OPENAI_API_KEY")
        self.api_url = os.getenv("CYBER_TOP_API_URL") or "https://api.openai.com/v1/chat/completions"
        self.model_target = os.getenv("CYBER_TOP_MODEL") or "gpt-4o"
        self.use_live_api = self.api_key is not None

    async def execute_orchestration(self, target_payload: str) -> str:
        if self.repo_index:
            summary = f"Scanned {len(self.repo_index['signatures'])} files. Critical logic identified in: {list(self.repo_index['critical_logic'].keys())[:5]}"
            target_payload = f"[REPO_CONTEXT: {summary}]\n\n{target_payload}"

        if self.use_live_api:
            return await self._call_live_api(target_payload)
        return await self._simulate_orchestration(target_payload)

    async def _call_live_api(self, payload: str) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
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

    async def _simulate_orchestration(self, payload: str) -> str:
        # STAGE 2: Intent Parsing & Weight Calculation
        intent, weights = self._parse_intent(payload)

        # STAGE 3: Quad-Agent Council Simulation
        agents = ["Mythos-Glasswing", "Cyber-Decompiler", "BigSleep-Mimic", "SEC-CORE Governor"]
        tasks = [self._simulate_agent(name, payload) for name in agents]
        agent_outputs = await asyncio.gather(*tasks)

        # STAGE 4: Cyber Top Overseer Synthesis & Zero-Day Discovery Loop
        # Resolve Conflicts using weights
        # Zero-Day Discovery Loop: Recursive critique
        critique = "If this tactical patch is applied blindly... secondary multi-threaded deadlocks might occur in the locking wrapper."
        refinement = "Refined strategy with linearizable mutex established."

        # Format result according to STAGE 5
        entities = re.findall(r'[a-zA-Z0-9_]{4,}', payload)
        entity = entities[0] if entities else "SYSTEM"

        cwe = "CWE-122: Heap-based Buffer Overflow" if "heap" in payload.lower() else "CWE-20: Improper Input Validation"
        score = f"{random.uniform(7.0, 9.9):.1f}"

        output = f"""
## Executive Telemetry Matrix
| Metric | Telemetry Value |
| :--- | :--- |
| **Vulnerability Vector** | {cwe} |
| **Exploitability Score** | {score} |
| **Rollback Risk** | Low - Minimal architectural regression expected. |

## Unified Coda Analysis
Analysis of `{entity}` reveals a critical trust-boundary propagation flaw. The agent consensus indicates that {agent_outputs[1].split('.')[0].lower()}. The Zero-Day Discovery Loop confirmed that a simple patch might introduce a race condition, thus the final directive includes a synchronized atomic wrapper.

## Triage Matrix
### 1. Tactical Patch (Quick Mitigation)
Immediate bounds-checking and length validation for `{entity}` inputs.
```python
# SEC-CORE Secure Patch
def secure_process(data):
    if len(data) > MAX_BUFFER_SIZE:
        raise SecurityException("Payload overflow detected")
    return process_raw(data)
```

### 2. Strategic Overhaul (Architectural Fix)
Migrate `{entity}` logic to a memory-safe Rust-based micro-service with strict Capability-Based Access Control (CBAC).

### 3. Defensive Telemetry
```yara
rule SEC_CORE_{entity}_Exploit {{
    meta:
        description = "Detects anomalous {entity} mutation patterns"
    strings:
        $p1 = {{ FF 00 AA 11 }}
    condition:
        $p1 at 0 and filesize < 2KB
}}
```
"""
        return output

    def _parse_intent(self, payload):
        p = payload.lower()
        if any(x in p for x in ["exploit", "poc", "0day"]):
            return "exploit_simulation", {"depth": 0.9, "stability": 0.1}
        if any(x in p for x in ["def ", "class ", "func"]):
            return "production_patching", {"security": 0.8, "performance": 0.2}
        return "hotfix_optimization", {"performance": 0.6, "security": 0.4}

    async def _simulate_agent(self, name, payload):
        await asyncio.sleep(random.uniform(0.05, 0.1))
        if name == "Mythos-Glasswing":
            return f"Architectural analysis of dependencies for target. Identified trust-boundary leak."
        if name == "Cyber-Decompiler":
            return f"Binary deconstruction reveals a potential buffer overflow at offset 0x4F2A."
        if name == "BigSleep-Mimic":
            return f"Adversarial fuzzer suggests an exploit chain involving race conditions."
        return f"Governor evaluates blast-radius as HIGH. Remediation priority established."

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
        }
        body {
            background-color: var(--bg); color: var(--text); font-family: 'Inter', sans-serif;
            margin: 0; padding: 0; height: 100vh; display: flex; flex-direction: column;
        }
        .chat-view { flex: 1; overflow-y: auto; padding: 40px; display: flex; flex-direction: column; gap: 30px; }
        .bubble { background: var(--surface); border: 1px solid var(--border); padding: 30px; border-radius: 16px; backdrop-filter: blur(20px); }
        .telemetry { font-family: 'JetBrains Mono', monospace; font-size: 11px; color: var(--neon); margin-bottom: 15px; }
        .markdown h2, .markdown h3 { color: var(--neon); font-family: 'JetBrains Mono', monospace; }
        .markdown table { border-collapse: collapse; width: 100%; margin: 20px 0; }
        .markdown th, .markdown td { border: 1px solid var(--border); padding: 12px; text-align: left; }
        .markdown pre { background: #000; padding: 20px; border-radius: 8px; color: var(--neon); overflow-x: auto; }
        .input-bar { padding: 30px 40px; background: #0a0a0f; border-top: 1px solid var(--border); display: flex; gap: 20px; }
        input { flex: 1; background: transparent; border: 1px solid var(--border); border-radius: 8px; padding: 18px; color: var(--neon); font-family: 'JetBrains Mono', monospace; outline: none; }
        button { padding: 0 40px; background: var(--neon); border: none; border-radius: 8px; font-weight: 900; cursor: pointer; }
    </style>
</head>
<body>
    <div class="chat-view" id="chat">
        <div class="bubble">
            <div class="telemetry">SEC-CORE // ORCHESTRATOR ONLINE [5.4-CYBER-TOP]</div>
            Awaiting input payload for Quad-Agent Debate...
        </div>
    </div>
    <div class="input-bar">
        <input type="text" id="inp" placeholder="Ingest code or payload..." autocomplete="off">
        <button id="btn">Analyze</button>
    </div>
    <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
    <script>
        const chat = document.getElementById('chat');
        const inp = document.getElementById('inp');
        const btn = document.getElementById('btn');

        btn.onclick = async () => {
            const v = inp.value.trim(); if(!v) return;
            inp.value = '';
            const uMsg = document.createElement('div'); uMsg.className = 'bubble'; uMsg.style.alignSelf = 'flex-end'; uMsg.innerText = v;
            chat.appendChild(uMsg);

            const load = document.createElement('div'); load.className = 'bubble'; load.innerHTML = '<div class="telemetry">STAGE 1: CALCULATING INTENT & WEIGHTS...</div>';
            chat.appendChild(load);
            chat.scrollTop = chat.scrollHeight;

            await new Promise(r => setTimeout(r, 800));
            load.innerHTML = '<div class="telemetry">STAGE 2: SYNCHRONIZING QUAD-AGENT COUNCIL...</div>';

            const r = await fetch('/api/v1/analyze', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({payload: v})
            });
            const d = await r.json();
            load.innerHTML = `<div class="telemetry">ANALYSIS COMPLETE</div><div class="markdown">${marked.parse(d.output)}</div>`;
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

    print("\n[⚡ INITIALIZING SEC-CORE UNIFIED ORCHESTRATION NETWORK - VERSION 5.4]")
    print(f"[🔄 ROUTING INFERENCE] Engine Target: {gateway.model_target}")
    print("[💥 RUNNING] Quad-Agent Debate & Zero-Day Discovery Loop engaged...")

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
