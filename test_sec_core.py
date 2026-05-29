import torch
import json
import http.client
from sec_core_unified import SECCoreUnified, SECConfig, run
import threading
import time

def test_standalone_model():
    print("Testing Standalone SECCoreUnified Model Logic...")
    cfg = SECConfig(dim=128, n_heads=4, n_kv_heads=1, n_experts=8)
    model = SECCoreUnified(cfg)
    input_ids = torch.randint(0, 256, (1, 8))

    # Forward pass
    logits, trace = model(input_ids)
    assert logits.shape == (1, 8, 256)
    assert len(trace) > 0
    print("Standalone Model Logic OK.")

def test_server_and_model_link():
    print("\nTesting Server-to-Model Link (Port 3000)...")
    # Kill any existing server
    import os
    import subprocess
    os.system("kill $(lsof -t -i :3000) 2>/dev/null || true")

    # Start server in a thread
    server_thread = threading.Thread(target=run, daemon=True)
    server_thread.start()
    time.sleep(5) # Wait for weight space initialization

    # Send actual analytical payload
    conn = http.client.HTTPConnection("127.0.0.1", 3000)
    headers = {'Content-type': 'application/json'}
    body = json.dumps({'payload': 'analyze stack buffer overflow in libc'})
    conn.request('POST', '/api/v1/analyze', body, headers)

    response = conn.getresponse()
    data = response.read().decode()

    print(f"Server Response Status: {response.status}")
    assert response.status == 200
    assert "SEC_CORE_TERMINAL" in data
    assert "Council Lens" in data
    assert "THE COMPREHENSIVE CODA" in data
    # Verify that the mock intelligence was triggered
    assert "CVE-2026-X" in data or "Internal Knowledge" in data
    print("Server-to-Model Link OK.")

if __name__ == "__main__":
    try:
        test_standalone_model()
        test_server_and_model_link()
        print("\n--- SEC-CORE VALIDATION SUITE PASSED ---")
    except Exception as e:
        print(f"\n--- VALIDATION FAILED: {e} ---")
        import traceback
        traceback.print_exc()
        exit(1)
