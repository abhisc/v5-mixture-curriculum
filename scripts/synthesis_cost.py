"""Cost the synthetic lanes in GPU-days.

Claiming "we will generate 186B agentic tokens" is wishful accounting unless the cost of
generating them is stated. This script converts the token targets into GPU-days using
(a) the measured throughput of the real generator in proxy/agentic_synth.py for the
execution-grounding half, and (b) explicit, editable inference assumptions for the
teacher-policy half.

Every assumption is a named constant. Change one and the conclusion changes; that is the
point. Nothing here is tuned to make the answer look good.

Token targets are derived from inventory/mixture.json so a catalog change moves the bill.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from budget import audit, lanes_of, load

ROOT = Path(__file__).resolve().parents[1]
# The generator writes into the (gitignored) data directory; results/ holds the committed
# copy so the cost table is reproducible from a fresh clone without regenerating traces.
SYNTH_STATS_CANDIDATES = [
    ROOT / "proxy" / "data" / "raw" / "agentic" / "synthesis_stats.json",
    ROOT / "results" / "synthesis_stats.json",
]

BYTES_PER_TOKEN = 4.0          # bytes per token for the code/terminal-heavy agentic mix
SECONDS_PER_DAY = 86_400


@dataclass
class Assumptions:
    # Teacher policy that proposes the plan and the next action at each step.
    teacher_decode_tok_per_s_per_gpu: float = 3_000.0
    # Fraction of generated rollouts that pass their verification assertion.
    accept_rate: float = 0.55
    # Fraction of accepted rollouts removed by near-duplicate filtering.
    dedup_loss: float = 0.20
    # Translation model for the Indic translated tier (IndicTrans2-class, ~1B params).
    mt_tok_per_s_per_gpu: float = 20_000.0
    # Round-trip quality gate rejection rate on machine translation.
    mt_reject_rate: float = 0.25
    # Dialogue generator for the Indic synthetic tier top-up.
    dialogue_tok_per_s_per_gpu: float = 3_000.0
    dialogue_accept_rate: float = 0.70


def measured_execution_throughput() -> dict:
    path = next((p for p in SYNTH_STATS_CANDIDATES if p.exists()), None)
    if path is None:
        return {}
    s = json.loads(path.read_text())
    return {
        "bytes_per_sec_per_core": s["bytes_per_sec"],
        "tokens_per_sec_per_core": s["bytes_per_sec"] / BYTES_PER_TOKEN,
        "verified_rate_measured": s["verified_rate"],
        "recovery_rate_measured": s["failure_recovery_rate"],
    }


def gpu_days(target_tokens_b: float, tok_per_s: float, yield_frac: float) -> tuple[float, float]:
    """Returns (tokens that must be generated in B, GPU-days)."""
    must_generate_b = target_tokens_b / max(1e-9, yield_frac)
    seconds = must_generate_b * 1e9 / tok_per_s
    return must_generate_b, seconds / SECONDS_PER_DAY


def report(a: Assumptions | None = None) -> dict:
    a = a or Assumptions()
    exec_stats = measured_execution_throughput()
    inv, mix = load()
    aud = audit(inv, mix)

    agentic = aud["lanes"]["agentic"]
    agentic_target = agentic.synthetic_target_b
    agentic_yield = a.accept_rate * (1 - a.dedup_loss)
    agentic_gen, agentic_gpu_days = gpu_days(
        agentic_target, a.teacher_decode_tok_per_s_per_gpu, agentic_yield
    )

    # The execution half runs on CPU, in parallel containers, and is measured not assumed.
    core_days = None
    if exec_stats:
        core_days = (
            agentic_gen * 1e9 * BYTES_PER_TOKEN
            / exec_stats["bytes_per_sec_per_core"]
            / SECONDS_PER_DAY
        )

    indic_cfg = lanes_of(mix)["indic"]
    indic_alloc = aud["pretrain_b"] * indic_cfg["share"]
    tiers = indic_cfg["tiers"]
    tier_supply = aud["indic_tiers"]

    mt_need = indic_alloc * tiers["translated"]["share"]
    mt_have = tier_supply.get("translated", 0.0)
    mt_target = max(0.0, mt_need - mt_have)
    mt_gen, mt_gpu_days = gpu_days(mt_target, a.mt_tok_per_s_per_gpu, 1 - a.mt_reject_rate)

    syn_need = indic_alloc * tiers["synthetic"]["share"]
    syn_have = tier_supply.get("synthetic", 0.0)
    dlg_target = max(0.0, syn_need - syn_have)
    dlg_gen, dlg_gpu_days = gpu_days(
        dlg_target, a.dialogue_tok_per_s_per_gpu, a.dialogue_accept_rate
    )

    total_gpu_days = agentic_gpu_days + mt_gpu_days + dlg_gpu_days
    out = {
        "measured_execution": exec_stats,
        "agentic": {
            "target_b": round(agentic_target, 1),
            "yield": round(agentic_yield, 3),
            "must_generate_b": round(agentic_gen, 1),
            "gpu_days": round(agentic_gpu_days, 0),
            "cpu_core_days_for_execution": round(core_days, 0) if core_days else None,
        },
        "indic_translated": {
            "target_b": round(mt_target, 1),
            "yield": round(1 - a.mt_reject_rate, 3),
            "must_generate_b": round(mt_gen, 1),
            "gpu_days": round(mt_gpu_days, 0),
        },
        "indic_synthetic_dialogue": {
            "target_b": round(dlg_target, 1),
            "yield": a.dialogue_accept_rate,
            "must_generate_b": round(dlg_gen, 1),
            "gpu_days": round(dlg_gpu_days, 0),
        },
        "total_gpu_days": round(total_gpu_days, 0),
        "wall_clock_days_on_512_gpus": round(total_gpu_days / 512, 1),
        "wall_clock_days_on_2048_gpus": round(total_gpu_days / 2048, 1),
    }
    return out


def markdown(r: dict) -> str:
    rows = [
        "| Synthetic slot | Tokens needed | Yield after verify + dedup | Tokens actually generated | GPU-days |",
        "|---|---:|---:|---:|---:|",
    ]
    for key, label in (
        ("agentic", "Agentic rollouts"),
        ("indic_translated", "Indic translated tier"),
        ("indic_synthetic_dialogue", "Indic synthetic dialogue"),
    ):
        d = r[key]
        rows.append(
            f"| {label} | {d['target_b']:.1f}B | {d['yield']*100:.0f}% | "
            f"{d['must_generate_b']:.1f}B | {d['gpu_days']:,.0f} |"
        )
    rows.append(
        f"| **Total** | | | | **{r['total_gpu_days']:,.0f}** "
        f"({r['wall_clock_days_on_2048_gpus']} days on 2048 GPUs) |"
    )
    return "\n".join(rows)


if __name__ == "__main__":
    r = report()
    print(json.dumps(r, indent=2))
    print()
    print(markdown(r))
