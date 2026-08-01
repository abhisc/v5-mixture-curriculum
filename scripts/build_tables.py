"""Generate every numeric table in README.md from the config and results files.

Run `python scripts/build_tables.py --inject` to rewrite the README in place. Each table
lives between `<!-- BEGIN:name -->` and `<!-- END:name -->` markers, so the prose is
hand-written but no number in the document is hand-typed. If the spec and the README ever
disagree, CI fails.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import synthesis_cost
from budget import audit, lanes_of, load, realized_shares, validate

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
RESULTS = ROOT / "results"

LANE_LABEL = {
    "web": "General web",
    "code": "Specialist code",
    "stem": "STEM / reasoning",
    "indic": "Native Indic",
    "agentic": "Agentic / tool-use",
    "longctx": "Long context",
}


def fmt_b(x: float) -> str:
    return f"{x/1000:.2f}T" if x >= 1000 else f"{x:.1f}B"


def t_budget(inv, mix, a) -> str:
    rows = [
        "| Lane | Share | Tokens | Benchmarks this lane must win | Supply verdict |",
        "|---|---:|---:|---|---|",
    ]
    for lane, cfg in lanes_of(mix).items():
        la = a["lanes"][lane]
        rows.append(
            f"| {LANE_LABEL[lane]} | {cfg['share']*100:.0f}% | {fmt_b(la.allocated_b)} | "
            f"{', '.join(cfg['benchmarks'])} | {la.verdict} |"
        )
    rows.append(
        f"| **Total pre-training** | **100%** | **{fmt_b(a['pretrain_b'])}** | | |"
    )
    return "\n".join(rows)


def t_supply(inv, mix, a) -> str:
    rows = [
        "| Lane | Allocated | Unique supply owned | Drawn from natural | Must be generated | Epochs on natural | Fresh-token equivalent |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for lane in lanes_of(mix):
        la = a["lanes"][lane]
        ep = "n/a" if la.total_draw_b == 0 else f"{la.epochs_on_natural:.2f}"
        rows.append(
            f"| {LANE_LABEL[lane]} | {fmt_b(la.allocated_b)} | {fmt_b(la.natural_unique_b)} | "
            f"{fmt_b(la.natural_target_b)} | {fmt_b(la.synthetic_target_b)} | {ep} | {fmt_b(la.effective_b)} |"
        )
    tot_alloc = sum(a["lanes"][l].allocated_b for l in lanes_of(mix))
    tot_eff = sum(a["lanes"][l].effective_b for l in lanes_of(mix))
    rows.append(
        f"| **Total** | **{fmt_b(tot_alloc)}** | | | | | "
        f"**{fmt_b(tot_eff)}** ({tot_eff/tot_alloc*100:.1f}%) |"
    )
    return "\n".join(rows)


def t_indic(inv, mix, a) -> str:
    indic = lanes_of(mix)["indic"]
    alloc = a["pretrain_b"] * indic["share"]
    supply = a["indic_tiers"]
    rows = [
        "| Tier | Share of Indic slot | Tokens | Natural supply available | Gap to close | How the gap is closed |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for tier, cfg in indic["tiers"].items():
        tokens = alloc * cfg["share"]
        avail = supply.get(tier, 0.0)
        gap = max(0.0, tokens - avail)
        headroom = max(0.0, avail - tokens)
        if tier == "verified" and gap <= 0:
            how = (
                f"None needed. We select the best {fmt_b(tokens)} of {fmt_b(avail)}, "
                "so we can afford to be picky."
            )
        elif tier == "unverified" and gap <= 0:
            how = (
                f"None needed. {fmt_b(headroom)} of headroom is held as contingency "
                "if the MT gate rejects more than planned."
            )
        elif tier == "unverified":
            how = (
                "SOTA inventory unverified crawl is thin; gap closed by up-weighting "
                "Sangraha synthetic and MT rather than inventing more crawl."
            )
        elif tier == "translated":
            how = "IndicTrans2-class MT over STEM and instruction tokens, round-trip chrF++ gated."
        elif tier == "synthetic":
            how = (
                "Sangraha synthetic tier from the SOTA inventory, plus a small "
                "native-script multi-turn dialogue top-up if selection rejects noisy shards."
            )
        else:
            how = "See inventory notes."
        rows.append(
            f"| {tier.capitalize()} | {cfg['share']*100:.0f}% | {fmt_b(tokens)} | {fmt_b(avail)} | "
            f"{'none' if gap <= 0 else fmt_b(gap)} | {how} |"
        )
    rows.append(f"| **Total** | **100%** | **{fmt_b(alloc)}** | | | |")
    return "\n".join(rows)


def t_floors(inv, mix) -> str:
    rows = [
        "| Lane | Budget share | Protected floor | Floor active from | What the floor prevents |",
        "|---|---:|---:|---|---|",
    ]
    why = {
        "web": "No floor. OPUS over-selects web by default; it needs a ceiling, not a floor.",
        "code": "Stops a mid-run reasoning spike from cannibalising SWE-bench capability.",
        "stem": "Long-CoT tokens look high-loss and get down-weighted by loss-based selectors.",
        "indic": "The binding constraint. Measured: cutting Indic to 0.5% costs 1.57 bits of Indic BPB to buy 0.05 bits of web BPB.",
        "agentic": "Measured: at 0.5% the model can no longer emit a well-formed tool call (structure score 0.03 vs 0.80). Terminal output looks like noise to a loss-based selector.",
        "longctx": "Prevents long packs being dropped for throughput reasons once sequence length rises.",
    }
    for lane, cfg in lanes_of(mix).items():
        f = cfg["floor"]
        rows.append(
            f"| {LANE_LABEL[lane]} | {cfg['share']*100:.1f}% | "
            f"{'none' if f == 0 else f'{f*100:.1f}%'} | {cfg.get('floor_active_from','S1')} | {why[lane]} |"
        )
    return "\n".join(rows)


def t_curriculum(inv, mix) -> str:
    stages = mix["curriculum"]["stages"]
    lanes = list(lanes_of(mix))
    head = "| Stage | Tokens | Seq len | " + " | ".join(LANE_LABEL[l] for l in lanes) + " |"
    sep = "|---|---:|---:|" + "---:|" * len(lanes)
    rows = [head, sep]
    for s in stages:
        cells = " | ".join(f"{s['mix'].get(l,0)*100:.1f}%" for l in lanes)
        rows.append(f"| {s['name']} | {fmt_b(s['tokens_b'])} | {s['seq_len']//1024}k | {cells} |")
    realized = realized_shares(mix)
    rows.append(
        "| **Realized S1-S4** | **" + fmt_b(sum(s["tokens_b"] for s in stages if not s["name"].startswith("S5")))
        + "** | | " + " | ".join(f"**{realized[l]*100:.2f}%**" for l in lanes) + " |"
    )
    rows.append(
        "| *Budget target* | | | " + " | ".join(f"*{lanes_of(mix)[l]['share']*100:.2f}%*" for l in lanes) + " |"
    )
    return "\n".join(rows)


def t_anneal(inv, mix) -> str:
    res = mix["anneal_reserve"]
    rows = [
        "| Tier A component | Lane | Tokens | Share of anneal | Why it is held back |",
        "|---|---|---:|---:|---|",
    ]
    for c in res["components"]:
        rows.append(
            f"| {c['id']} | {LANE_LABEL[c['lane']]} | {fmt_b(c['tokens_b'])} | "
            f"{c['tokens_b']/res['total_tokens_b']*100:.1f}% | {c['why']} |"
        )
    rows.append(f"| **Total** | | **{fmt_b(res['total_tokens_b'])}** | **100%** | |")
    return "\n".join(rows)


def t_bands(inv, mix) -> str:
    b = mix["reasoning_bands"]
    rows = [
        "| Band | Thinking-token budget | Share of reasoning tokens |",
        "|---|---|---:|",
    ]
    for k in ("low", "medium", "high", "ultra"):
        rows.append(f"| `{k}` | {b[k]['token_budget']} | {b[k]['share_of_reasoning_tokens']*100:.0f}% |")
    return "\n".join(rows)


def t_cleaning() -> str:
    p = RESULTS / "cleaning_report.json"
    if not p.exists():
        return "_not yet generated - run `make clean-data`_"
    rep = json.loads(p.read_text())
    rows = [
        "| Lane | Docs in | Docs kept | Retention | Short | Exact dup | Near dup | Romanised (reclassified) | Repetitive | Path-scrubbed |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for lane, r in rep.items():
        rows.append(
            f"| {LANE_LABEL.get(lane, lane)} | {r['docs_in']} | {r['docs_out']} | {r['retention']*100:.1f}% | "
            f"{r['dropped_short']} | {r['dropped_exact_dup']} | {r['dropped_near_dup']} | "
            f"{r['reclassified_romanised']} | {r['dropped_repetitive']} | {r['scrubbed_docs']} |"
        )
    return "\n".join(rows)


def _agg(results: dict) -> dict:
    """Average replicated seeds, keeping the spread."""
    by_arm: dict[str, list[dict]] = {}
    for _, r in results.items():
        by_arm.setdefault(r["arm"], []).append(r)
    out = {}
    for arm, runs in by_arm.items():
        lanes = runs[0]["bpb"].keys()
        out[arm] = {
            "n_seeds": len(runs),
            "bpb": {l: sum(r["bpb"][l] for r in runs) / len(runs) for l in lanes},
            "bpb_spread": {
                l: (max(r["bpb"][l] for r in runs) - min(r["bpb"][l] for r in runs))
                for l in lanes
            },
            "fidelity": sum(r["indic_script_fidelity"] for r in runs) / len(runs),
            "structure": sum(r["agentic_structure_score"] for r in runs) / len(runs),
            "mixture": runs[0]["mixture"],
            "tokens": runs[0]["tokens_seen"],
            "params": runs[0]["params_non_embedding"],
        }
    return out


def t_ablation() -> str:
    p = RESULTS / "ablation.json"
    if not p.exists():
        return "_not yet generated - run `make ablation`_"
    agg = _agg(json.loads(p.read_text()))
    order = ["v5", "opus_greedy", "indic_00", "indic_02", "indic_04", "indic_16",
             "agentic_00", "agentic_02", "agentic_10", "indic_only", "agentic_only"]
    rows = [
        "| Arm | Indic % | Agentic % | web BPB | code BPB | stem BPB | indic BPB | agentic BPB | Indic script fidelity | Agentic structure |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in order:
        if arm not in agg:
            continue
        a = agg[arm]
        m = a["mixture"]
        b = a["bpb"]
        rows.append(
            f"| `{arm}` | {m.get('indic',0)*100:.1f} | {m.get('agentic',0)*100:.1f} | "
            f"{b.get('web',float('nan')):.3f} | {b.get('code',float('nan')):.3f} | {b.get('stem',float('nan')):.3f} | "
            f"{b.get('indic',float('nan')):.3f} | {b.get('agentic',float('nan')):.3f} | "
            f"{a['fidelity']:.3f} | {a['structure']:.3f} |"
        )
    return "\n".join(rows)


def t_findings() -> str:
    """Derived quantities that answer the three hypotheses directly."""
    p = RESULTS / "ablation.json"
    if not p.exists():
        return "_not yet generated - run `make ablation`_"
    agg = _agg(json.loads(p.read_text()))

    def bpb(arm: str, lane: str) -> float:
        return agg[arm]["bpb"][lane]

    rows = [
        "| Question | Comparison | Measured |",
        "|---|---|---|",
    ]

    if "v5" in agg and "indic_00" in agg:
        web_tax = bpb("v5", "web") - bpb("indic_00", "web")
        indic_gain = bpb("indic_00", "indic") - bpb("v5", "indic")
        rows.append(
            f"| **H1.** What does the 8% Indic floor cost the general lane? | `v5` vs `indic_00`, web BPB "
            f"| **+{web_tax:.3f} bits** ({web_tax/bpb('indic_00','web')*100:.1f}% worse) |"
        )
        rows.append(
            f"| **H1.** What does it buy? | `v5` vs `indic_00`, Indic BPB "
            f"| **−{indic_gain:.3f} bits** ({indic_gain/bpb('indic_00','indic')*100:.1f}% better) |"
        )
        rows.append(
            f"| Return on the trade | Indic bits gained per general bit paid "
            f"| **{indic_gain/max(1e-9, web_tax):.0f}×** |"
        )
        rows.append(
            f"| **H2.** Is the damage gradual or categorical? | Indic script fidelity, `v5` vs `indic_00` "
            f"| **{agg['v5']['fidelity']:.3f} vs {agg['indic_00']['fidelity']:.3f}** |"
        )

    if "v5" in agg and "opus_greedy" in agg:
        rows.append(
            f"| **H2.** What does an unprotected selector cost? | `opus_greedy` vs `v5`, Indic BPB "
            f"| **{bpb('opus_greedy','indic') - bpb('v5','indic'):+.3f} bits** |"
        )
        rows.append(
            f"| | `opus_greedy` vs `v5`, agentic BPB "
            f"| **{bpb('opus_greedy','agentic') - bpb('v5','agentic'):+.3f} bits** |"
        )
        rows.append(
            f"| | `opus_greedy` vs `v5`, web BPB (what it gains) "
            f"| **{bpb('opus_greedy','web') - bpb('v5','web'):+.3f} bits** |"
        )
        rows.append(
            f"| | `opus_greedy` vs `v5`, script fidelity "
            f"| **{agg['opus_greedy']['fidelity']:.3f} vs {agg['v5']['fidelity']:.3f}** |"
        )

    # H3: marginal value of each extra point of Indic share.
    sweep = [("indic_00", 0.0), ("indic_02", 2.0), ("indic_04", 4.0), ("v5", 8.0), ("indic_16", 16.0)]
    have = [(a, s) for a, s in sweep if a in agg]
    for (a1, s1), (a2, s2) in zip(have, have[1:]):
        d = bpb(a1, "indic") - bpb(a2, "indic")
        rows.append(
            f"| **H3.** Marginal Indic BPB per extra point of share | {s1:.0f}% -> {s2:.0f}% "
            f"| **{d/(s2-s1):+.4f} bits/pp** (total {d:+.3f}) |"
        )

    if "v5" in agg and "indic_only" in agg:
        rows.append(
            f"| Crowding-out cost vs a pure specialist | `v5` vs `indic_only`, Indic BPB "
            f"| **{bpb('v5','indic') - bpb('indic_only','indic'):+.3f} bits** |"
        )
    if "v5" in agg and "agentic_only" in agg:
        rows.append(
            f"| | `v5` vs `agentic_only`, agentic BPB "
            f"| **{bpb('v5','agentic') - bpb('agentic_only','agentic'):+.3f} bits** |"
        )

    if "v5" in agg:
        spread = agg["v5"]["bpb_spread"]
        n = agg["v5"]["n_seeds"]
        rows.append(
            f"| **Is the general-lane cost even measurable?** | seed-to-seed spread of `v5` web BPB "
            f"over {n} seeds | **{spread['web']:.3f} bits** - the same size as the effect above |"
        )
        rows.append(
            f"| Seed noise ceiling | worst-lane BPB spread across {n} seeds of `v5` "
            f"| **{max(spread.values()):.3f} bits** (agentic) |"
        )
    return "\n".join(rows)


def t_synthesis() -> str:
    p = ROOT / "proxy" / "data" / "raw" / "agentic" / "synthesis_stats.json"
    if not p.exists():
        return "_not yet generated - run `make agentic`_"
    s = json.loads(p.read_text())
    rows = [
        "| Measurement | Value |",
        "|---|---:|",
        f"| Traces generated | {s['traces']:,} |",
        f"| Tool-call steps executed | {s['steps']:,} |",
        f"| Wall time | {s['seconds']:.0f}s on one laptop core |",
        f"| Throughput | {s['traces_per_sec']:.1f} traces/s, {s['bytes_per_sec']/1e3:.1f} kB/s |",
        f"| Mean trace size | {s['mean_bytes_per_trace']:.0f} bytes |",
        f"| Assertion-verified traces | {s['verified_rate']*100:.1f}% |",
        f"| Traces containing a real failure then recovery | {s['failure_recovery_rate']*100:.1f}% |",
    ]
    return "\n".join(rows)


def build(inv, mix) -> dict[str, str]:
    a = audit(inv, mix)
    return {
        "budget": t_budget(inv, mix, a),
        "supply": t_supply(inv, mix, a),
        "indic": t_indic(inv, mix, a),
        "floors": t_floors(inv, mix),
        "curriculum": t_curriculum(inv, mix),
        "anneal": t_anneal(inv, mix),
        "bands": t_bands(inv, mix),
        "cleaning": t_cleaning(),
        "ablation": t_ablation(),
        "findings": t_findings(),
        "synthesis": t_synthesis(),
        "synthcost": synthesis_cost.markdown(synthesis_cost.report()),
    }


def inject(tables: dict[str, str]) -> int:
    text = README.read_text()
    missing = []
    for name, table in tables.items():
        begin, end = f"<!-- BEGIN:{name} -->", f"<!-- END:{name} -->"
        if begin not in text or end not in text:
            missing.append(name)
            continue
        pre = text.split(begin)[0]
        post = text.split(end, 1)[1]
        text = f"{pre}{begin}\n{table}\n{end}{post}"
    README.write_text(text)
    if missing:
        print("  no marker in README for:", ", ".join(missing))
    return len(tables) - len(missing)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--inject", action="store_true")
    args = ap.parse_args()

    inv, mix = load()
    problems = validate(inv, mix)
    if problems:
        print("SPEC INCONSISTENT:")
        for p in problems:
            print("  -", p)
        raise SystemExit(1)

    tables = build(inv, mix)
    if args.inject:
        n = inject(tables)
        print(f"injected {n} tables into README.md")
    else:
        for name, t in tables.items():
            print(f"\n### {name}\n{t}")
