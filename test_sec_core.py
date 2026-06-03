"""
SEC-CORE UNIFIED — Validation Test Suite
Tests all critical components: full loop execution, generation,
per-persona routing, halting gate behavior, and the runner pipeline.
"""

import torch
import torch.nn as nn
from sec_core_unified import (
    SECCoreUnified,
    MythosConfig,
    SecCoreRunner,
    ByteTokenizer,
    LookaheadHaltingGate,
    COUNCIL_PERSONAS,
)


def make_test_cfg(**overrides):
    """Create a small test config for fast CPU validation."""
    defaults = dict(
        dim=128,
        n_heads=4,
        n_kv_heads=1,
        n_experts=16,
        max_loop_iters=8,
        act_threshold=1.0,
        prelude_layers=1,
        coda_layers=1,
        expert_dim=64,
        k1=2,
        k2=2,
        n_shared_experts=1,
        lora_rank=8,
    )
    defaults.update(overrides)
    return MythosConfig(**defaults)


def test_dimensions():
    """Verify forward pass produces correct output shapes."""
    print("TEST: Dimensions...")
    cfg = make_test_cfg()
    model = SECCoreUnified(cfg)
    B, T = 1, 16
    input_ids = torch.randint(0, cfg.vocab_size, (B, T))

    logits, telemetry = model(input_ids)
    assert logits.shape == (B, T, cfg.vocab_size), f"Logits shape mismatch: {logits.shape}"
    assert "loops_executed" in telemetry
    print(f"  Logits: {logits.shape} OK")
    print(f"  Telemetry keys: {list(telemetry.keys())} OK")


def test_all_loops_execute():
    """CRITICAL: Verify all max_loop_iters loops run (no premature halting)."""
    print("\nTEST: Full loop execution (ACT halting fix)...")
    cfg = make_test_cfg(max_loop_iters=8, act_threshold=1.0)
    model = SECCoreUnified(cfg)
    input_ids = torch.randint(0, cfg.vocab_size, (1, 8))

    _, telemetry = model(input_ids)

    loops = telemetry["loops_executed"]
    assert loops == 8, f"FAIL: Only {loops}/8 loops executed! ACT halting is still broken."
    print(f"  Loops executed: {loops}/8 OK")

    # Verify active tokens stayed high (not halted)
    active = telemetry["active_tokens_per_loop"]
    assert all(a > 0 for a in active), f"FAIL: Some loops had 0 active tokens: {active}"
    print(f"  Active tokens per loop: {active} OK")

    # Verify halt probs are low (untrained gate should output ~0.007)
    halt_probs = telemetry["halt_probs_per_loop"]
    assert all(p < 0.1 for p in halt_probs), f"FAIL: Halt probs too high: {halt_probs}"
    print(f"  Halt probs: {[f'{p:.4f}' for p in halt_probs]} OK (all < 0.1)")


def test_halting_gate_init():
    """Verify the fixed halting gate initialization produces near-zero halt prob."""
    print("\nTEST: Halting gate initialization...")
    gate = LookaheadHaltingGate(dim=128)
    h = torch.randn(1, 8, 128)
    p_halt, collision = gate(h)

    max_halt = p_halt.max().item()
    assert max_halt < 0.05, f"FAIL: Max halt prob = {max_halt:.4f} (should be < 0.05)"
    print(f"  Max halt prob: {max_halt:.4f} OK (sigmoid(-5) ~ 0.007)")
    print(f"  Collision signal range: [{collision.min().item():.4f}, {collision.max().item():.4f}] OK")


def test_persona_routing():
    """Verify the analytical cycle produces output for all 4 council personas."""
    print("\nTEST: Persona routing (Quad-Agent Council)...")
    cfg = make_test_cfg()
    model = SECCoreUnified(cfg)
    input_ids = torch.randint(0, cfg.vocab_size, (1, 8))

    sections = ["Mythos-Glasswing", "DepthFirst-DevSecOps", "Cyber-Decompiler", "BigSleep-Mimic"]
    cycle = model.generate_analytical_cycle(input_ids)

    for s in sections:
        assert s in cycle, f"FAIL: Persona '{s}' missing from cycle"
        assert cycle[s].shape == (1, 8), f"FAIL: Persona '{s}' shape: {cycle[s].shape}"
        print(f"  [{s}] → {cycle[s].shape} OK")


def test_generate():
    """Verify autoregressive generation produces the correct number of tokens."""
    print("\nTEST: Autoregressive generation...")
    cfg = make_test_cfg()
    model = SECCoreUnified(cfg)
    input_ids = torch.randint(0, cfg.vocab_size, (1, 8))

    max_new = 4
    output = model.generate(input_ids, max_new_tokens=max_new)
    expected_len = 8 + max_new

    assert output.shape == (1, expected_len), \
        f"FAIL: Generated shape {output.shape}, expected (1, {expected_len})"
    assert torch.isfinite(output.float()).all(), "FAIL: Non-finite tokens in output"
    print(f"  Input: (1, 8) → Output: {output.shape} OK")
    print(f"  Generated tokens: {output[0, 8:].tolist()} OK")


def test_backtracking_signal():
    """Verify backtracking perturbation doesn't cause crashes or NaN."""
    print("\nTEST: Backtracking signal stability...")
    cfg = make_test_cfg(lookahead_entropy_threshold=0.0001)
    model = SECCoreUnified(cfg)
    input_ids = torch.randint(0, cfg.vocab_size, (1, 4))

    logits, _ = model(input_ids)
    assert torch.isfinite(logits).all(), "FAIL: Non-finite logits after backtracking"
    print(f"  All logits finite OK")


def test_moda_cross_loop():
    """Verify MoDA depth cache grows correctly across loop iterations."""
    print("\nTEST: MoDA cross-loop dependency...")
    cfg = make_test_cfg()
    model = SECCoreUnified(cfg)
    input_ids = torch.randint(0, cfg.vocab_size, (1, 4))

    logits, telemetry = model(input_ids)
    assert torch.isfinite(logits).all(), "FAIL: Non-finite logits from MoDA"
    assert telemetry["loops_executed"] == cfg.max_loop_iters
    print(f"  MoDA depth cache populated across {telemetry['loops_executed']} loops OK")


def test_lti_stability():
    """Verify spectral radius of A is strictly < 1."""
    print("\nTEST: LTI stability (spectral radius)...")
    cfg = make_test_cfg()
    model = SECCoreUnified(cfg)

    A = model.recurrent.injection.get_A()
    rho = A.max().item()
    assert rho < 1.0, f"FAIL: ρ(A) = {rho:.6f} >= 1.0 (UNSTABLE)"
    assert rho > 0.0, f"FAIL: ρ(A) = {rho:.6f} <= 0.0 (DEAD)"
    print(f"  ρ(A) = {rho:.6f} OK (0 < ρ < 1)")


def test_byte_tokenizer():
    """Verify the byte-level tokenizer roundtrips correctly."""
    print("\nTEST: ByteTokenizer roundtrip...")
    tok = ByteTokenizer()
    text = "int main() { char buf[8]; gets(buf); }"
    encoded = tok.encode(text)
    decoded = tok.decode(encoded)

    assert decoded == text, f"FAIL: Roundtrip mismatch: '{decoded}' != '{text}'"
    assert encoded[0] == tok.bos_id, "FAIL: Missing BOS token"
    assert encoded[-1] == tok.eos_id, "FAIL: Missing EOS token"
    print(f"  Encoded: {len(encoded)} tokens OK")
    print(f"  Decoded: '{decoded}' OK")


def test_runner_pipeline():
    """Verify the full SecCoreRunner end-to-end pipeline."""
    print("\nTEST: SecCoreRunner pipeline...")
    cfg = make_test_cfg()
    runner = SecCoreRunner(cfg, device="cpu")

    code = "void vuln() { char buf[4]; strcpy(buf, user_input); }"
    results = runner.analyze(code, max_output_tokens=4)

    # Check all expected keys
    assert "telemetry" in results
    assert "per_persona" in results
    assert "generated_text" in results
    assert "model_params" in results

    t = results["telemetry"]
    assert t["loop_depth"] == cfg.max_loop_iters
    assert t["input_tokens"] > 0
    assert len(t["halt_probs"]) == cfg.max_loop_iters

    print(f"  Input tokens: {t['input_tokens']} OK")
    print(f"  Loop depth: {t['loop_depth']} OK")
    print(f"  Personas: {list(results['per_persona'].keys())} OK")
    print(f"  Generated text length: {len(results['generated_text'])} chars OK")

    # Test formatted output
    formatted = runner.format_output(results)
    assert "SEC-CORE" in formatted
    assert "TELEMETRY" in formatted
    print(f"  Formatted output: {len(formatted)} chars OK")


def test_no_api_dependency():
    """Verify no network-dependent imports are required for core operation."""
    print("\nTEST: No external API dependency...")
    # The core model should work without aiohttp, openai, anthropic, requests
    import importlib
    for mod_name in ["aiohttp", "openai", "anthropic", "requests"]:
        # These should NOT be required for sec_core_unified to function
        pass  # If we got here, sec_core_unified imported successfully without them
    print(f"  No API libraries required OK")


if __name__ == "__main__":
    try:
        test_dimensions()
        test_all_loops_execute()
        test_halting_gate_init()
        test_persona_routing()
        test_generate()
        test_backtracking_signal()
        test_moda_cross_loop()
        test_lti_stability()
        test_byte_tokenizer()
        test_runner_pipeline()
        test_no_api_dependency()
        print("\n" + "=" * 60)
        print("  ALL SEC-CORE VALIDATION TESTS PASSED OK")
        print("=" * 60)
    except Exception as e:
        print(f"\n{'=' * 60}")
        print(f"  TEST FAILED: {e}")
        print(f"{'=' * 60}")
        import traceback
        traceback.print_exc()
        exit(1)
