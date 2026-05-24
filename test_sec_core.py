import torch
import torch.nn as nn
from sec_core_unified import SECCoreUnified, MythosConfig

def test_dimensions():
    print("Testing Dimensions...")
    cfg = MythosConfig(dim=256, n_heads=8, n_kv_heads=2, n_experts=16)
    model = SECCoreUnified(cfg)
    B, T = 1, 16
    input_ids = torch.randint(0, cfg.vocab_size, (B, T))

    logits = model(input_ids)
    assert logits.shape == (B, T, cfg.vocab_size), f"Logits shape mismatch: {logits.shape}"
    print("Logits shape OK.")

def test_history_access():
    print("\nTesting History Access & Sectional Generation...")
    cfg = MythosConfig(dim=128, n_heads=4, n_kv_heads=1, n_experts=16)
    model = SECCoreUnified(cfg)
    input_ids = torch.randint(0, cfg.vocab_size, (1, 8))

    sections = ["Mythos-Glasswing", "DepthFirst-DevSecOps", "Cyber-Decompiler", "DeepMind-BigSleep"]
    cycle = model.generate_analytical_cycle(input_ids)

    for s in sections:
        assert s in cycle, f"Section {s} missing from cycle"
        assert cycle[s].shape == (1, 8), f"Section {s} shape mismatch: {cycle[s].shape}"
        print(f"Section {s} OK.")

def test_backtracking_signal():
    print("\nTesting Backtracking Signal (Mock Collision)...")
    cfg = MythosConfig(dim=128, n_heads=4, n_kv_heads=1, n_experts=16, lookahead_entropy_threshold=0.0001)
    # Low threshold should trigger backtracking more often
    model = SECCoreUnified(cfg)
    input_ids = torch.randint(0, cfg.vocab_size, (1, 4))

    # We can't easily see if backtracking happened without hooks, but we ensure it doesn't crash
    logits = model(input_ids)
    assert torch.isfinite(logits).all()
    print("Backtracking pass OK (no crashes).")

def test_moda_cross_loop():
    print("\nTesting MoDA Cross-Loop Dependency...")
    cfg = MythosConfig(dim=128, n_heads=4, n_kv_heads=1, n_experts=16)
    model = SECCoreUnified(cfg)
    input_ids = torch.randint(0, cfg.vocab_size, (1, 4))

    # Forward pass uses MoDA with depth_cache populated loop-by-loop
    logits = model(input_ids)
    assert torch.isfinite(logits).all()
    print("MoDA cross-loop OK.")

if __name__ == "__main__":
    try:
        test_dimensions()
        test_history_access()
        test_backtracking_signal()
        test_moda_cross_loop()
        print("\n--- ALL SEC-CORE VALIDATION TESTS PASSED ---")
    except Exception as e:
        print(f"\n--- TEST FAILED: {e} ---")
        import traceback
        traceback.print_exc()
        exit(1)
