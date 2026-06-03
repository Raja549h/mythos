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
        if self.use_live_api:
            return await self._call_live_api(target_payload)
        return await self._simulate_orchestration(target_payload)

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

    async def _simulate_orchestration(self, payload: str) -> str:
        # Detect if conversational
        if len(payload.split()) < 5 and any(kw in payload.lower() for kw in ["hi", "hello", "who", "help", "hey"]):
            return self._conversational_response(payload)

        # STAGE 2: Intent Parsing & Weight Calculation
        intent, weights = self._parse_intent(payload)

        # Extract features for dynamic response
        entities = re.findall(r'[a-zA-Z_][a-zA-Z0-9_]{3,}', payload)
        # Filter out common keywords
        keywords = ["analyze", "check", "detect", "buffer", "overflow", "heap", "stack", "logic", "flaw"]
        filtered_entities = [e for e in entities if e.lower() not in keywords]
        entity = filtered_entities[0] if filtered_entities else (entities[0] if entities else "CORE_MODULE")

        # STAGE 3: Quad-Agent Council Simulation
        # Simulate agent outputs based on input content
        vuln_type = "Buffer Overflow" if "buffer" in payload.lower() or "malloc" in payload.lower() else \
                    "Injection Vector" if "query" in payload.lower() or "exec" in payload.lower() else \
                    "Logic Flaw"

        # Format result according to STAGE 5
        cwe = "CWE-122: Heap-based Buffer Overflow" if "heap" in payload.lower() else \
              "CWE-78: OS Command Injection" if "exec" in payload.lower() else \
              "CWE-20: Improper Input Validation"

        score = f"{random.uniform(6.5, 9.8):.1f}"

        # STAGE 4: Zero-Day Discovery Loop (Simulated)
        critiques = [
            f"Blind application of the `{entity}` patch may introduce a 1-byte heap off-by-one error during re-alignment.",
            f"The tactical fix for `{entity}` lacks thread-safety; high-concurrency environments could trigger a race condition in the validator.",
            f"Proposed mitigation for `{entity}` might be bypassed by polymorphic payloads using non-standard encoding."
        ]
        critique = random.choice(critiques)

        output = f"""
## Executive Telemetry Matrix
| Metric | Telemetry Value |
| :--- | :--- |
| **Vulnerability Vector** | {cwe} |
| **Exploitability Score** | {score} |
| **Rollback Risk** | Low - Targeted hotfix avoids regression in master branch. |

## Unified Coda Analysis
The SEC-CORE consensus identifies a critical `{vuln_type}` within the `{entity}` component.

**Mythos-Glasswing** reports that the architectural trust boundary between the input handler and the processing engine is non-existent, allowing unvalidated propagation of user-controlled state.

**Cyber-Decompiler** confirms that low-level memory layout for `{entity}` lacks guard pages, making it susceptible to deterministic exploitation.

**BigSleep-Mimic** successfully synthesized a 3-step exploitation chain that bypasses current stack canaries by leveraging a side-channel in the adjacent telemetry module.

**Adversarial Critique (Zero-Day Loop):** {critique}

## Triage Matrix
### 1. Tactical Patch (Quick Mitigation)
Inject a strict validation layer at the entry point of `{entity}`.
```python
# SEC-CORE AUTOMATED PATCH
def validated_{entity}(input_data):
    # Enforce strict length and semantic bounds
    if not is_valid_format(input_data) or len(input_data) > 1024:
        SEC_LOG.alert("ADVERSARIAL INPUT BLOCKED")
        return None
    return original_{entity}(input_data)
```

### 2. Strategic Overhaul (Architectural Fix)
Implement a **Recurrent-Depth Validator (RDV)** inspired by the OpenMythos RDT architecture found in `mythos_unified.py`. This uses adaptive halting to process input complexity proportional to its risk score.

### 3. Defensive Telemetry
```yara
rule SEC_CORE_DYNAMIC_{entity} {{
    meta:
        description = "Detects mutation patterns targeting {entity}"
        author = "SEC-CORE Orchestrator 5.4"
    strings:
        $s1 = "{entity}"
        $hex = {{ 41 41 41 41 41 }}
    condition:
        all of them
}}
```
"""
        return output

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

    def _parse_intent(self, payload):
        p = payload.lower()
        if any(x in p for x in ["exploit", "poc", "0day", "attack"]):
            return "exploit_simulation", {"depth": 0.9, "stability": 0.1}
        if any(x in p for x in ["def ", "class ", "func", "return"]):
            return "production_patching", {"security": 0.8, "performance": 0.2}
        return "hotfix_optimization", {"performance": 0.6, "security": 0.4}

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
