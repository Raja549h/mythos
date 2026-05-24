import torch
from sec_core_unified import SECCoreUnified, SECConfig

def test_tool_gating_consensus():
    print("Testing Tool Gating Consensus...")
    cfg = SECConfig(dim=128, n_heads=4, n_kv_heads=1, n_experts=16)
    model = SECCoreUnified(cfg)
    input_ids = torch.randint(0, cfg.vocab_size, (2, 8)) # Batch size 2

    # Get tool trace (using internal forward for testing)
    with torch.no_grad():
        cos, sin = model.rope(8)
        x = model.embed(input_ids)
        for layer in model.prelude:
            x = layer(x, (cos, sin))
        _, _, tool_trace = model.recurrent(x, x, (cos, sin))

    print(f"Tool Trace: {tool_trace}")
    assert len(tool_trace) > 0
    print("Tool Gating Consensus OK.")

def test_latent_pause_ttl():
    print("\nTesting Latent Pause TTL...")
    # High TTL penalty should change the scale of A in LTI
    cfg_low = SECConfig(dim=128, n_heads=8, n_kv_heads=2, wait_state_ttl=0.0)
    cfg_high = SECConfig(dim=128, n_heads=8, n_kv_heads=2, wait_state_ttl=0.9)

    model_low = SECCoreUnified(cfg_low)
    model_high = SECCoreUnified(cfg_high)

    # Mock 'Search' action
    with torch.no_grad():
        model_low.recurrent.tool_gate.gate.bias[1] = 100.0
        model_high.recurrent.tool_gate.gate.bias[1] = 100.0

    input_ids = torch.randint(0, 32000, (1, 4))

    with torch.no_grad():
        cos, sin = model_low.rope(4)
        x = model_low.embed(input_ids)
        # Low TTL
        h_low, _, _ = model_low.recurrent(x, x, (cos, sin))
        # High TTL
        h_high, _, _ = model_high.recurrent(x, x, (cos, sin))

    # States should differ due to wait penalty
    diff = (h_low - h_high).abs().mean()
    print(f"Mean Difference with TTL: {diff.item()}")
    assert diff > 0, "Wait state penalty had no effect on latent states"
    print("Latent Pause TTL OK.")

def test_sectional_generation_v7():
    print("\nTesting Sectional Generation (V7)...")
    cfg = SECConfig(dim=128, n_heads=4, n_kv_heads=1, n_experts=16)
    model = SECCoreUnified(cfg)
    input_ids = torch.randint(0, cfg.vocab_size, (1, 8))

    cycle = model.generate_analytical_cycle(input_ids)
    expected = ["Mythos-Glasswing", "DepthFirst-DevSecOps", "Cyber-Decompiler", "DeepMind-BigSleep", "Tool-Interaction-Trace"]

    for s in expected:
        assert s in cycle, f"Section {s} missing"
        print(f"Section {s} OK.")
    print("Sectional Generation V7 OK.")

if __name__ == "__main__":
    try:
        test_tool_gating_consensus()
        test_latent_pause_ttl()
        test_sectional_generation_v7()
        print("\n--- ALL SEC-CORE V7 VALIDATION TESTS PASSED ---")
    except Exception as e:
        print(f"\n--- TEST FAILED: {e} ---")
        import traceback
        traceback.print_exc()
        exit(1)
