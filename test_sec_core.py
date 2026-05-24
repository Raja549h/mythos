import torch
import json
import http.client
from sec_core_unified import SECCoreUnified, SECConfig, run_server
import threading
import time

def test_standalone_model():
    print("Testing Standalone SECCoreUnified Model...")
    cfg = SECConfig(dim=128, n_heads=4, n_kv_heads=1, n_experts=16)
    model = SECCoreUnified(cfg)
    input_ids = torch.randint(0, cfg.vocab_size, (1, 8))

    # Forward pass
    logits, trace = model(input_ids)
    assert logits.shape == (1, 8, cfg.vocab_size)
    assert len(trace) > 0
    print("Standalone Model Logic OK.")

def test_server_endpoint():
    print("\nTesting Standalone Server Endpoint...")
    # Start server in a thread
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()
    time.sleep(5) # Wait for init

    # Send request
    conn = http.client.HTTPConnection("127.0.0.1", 8000)
    headers = {'Content-type': 'application/json'}
    body = json.dumps({'payload': 'test buffer overflow'})
    conn.request('POST', '/api/v1/analyze', body, headers)

    response = conn.getresponse()
    data = response.read().decode()

    print(f"Server Response Status: {response.status}")
    assert response.status == 200
    assert "SEC_CORE_TERMINAL" in data
    assert "THE COMPREHENSIVE CODA" in data
    print("Standalone Server Endpoint OK.")

if __name__ == "__main__":
    try:
        test_standalone_model()
        test_server_endpoint()
        print("\n--- STANDALONE SEC-CORE VALIDATION PASSED ---")
    except Exception as e:
        print(f"\n--- TEST FAILED: {e} ---")
        import traceback
        traceback.print_exc()
        exit(1)
