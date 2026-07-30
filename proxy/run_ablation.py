"""Run the mixture ablation.

Arms are constructed so that each comparison isolates one variable.

  v5           the proposed mixture, floors on
  opus_greedy  what an unconstrained selector does when its scoring proxy is English and
               maths heavy: it starves Indic and agentic down to noise and spends the
               tokens on web/code/stem instead. This is the arm the protected floors exist
               to prevent.
  indic_XX     Indic share swept from 0% to 16%. The token difference is always taken from
               the web lane and never from code/stem/agentic, so any movement in Indic
               metrics is attributable to the Indic share alone.
  agentic_XX   same sweep design for the agentic lane.
  *_only       single-lane specialists. Not a proposal, a CEILING: the best this
               architecture and token budget can do on that lane. The gap between a
               mixture arm and its specialist is the crowding-out cost.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from model import Config
from train import train_arm

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

V5 = {"web": 0.35, "code": 0.32, "stem": 0.18, "indic": 0.08, "agentic": 0.05, "longctx": 0.02}


def swap(lane: str, value: float, base: dict[str, float] = V5) -> dict[str, float]:
    """Set `lane` to `value`, taking the difference from the web lane."""
    m = dict(base)
    delta = value - m[lane]
    m[lane] = value
    m["web"] = round(m["web"] - delta, 6)
    assert m["web"] >= 0, f"web lane would go negative for {lane}={value}"
    assert abs(sum(m.values()) - 1.0) < 1e-9
    return m


ARMS: dict[str, dict[str, float]] = {
    "v5": V5,
    "opus_greedy": {"web": 0.42, "code": 0.37, "stem": 0.18, "indic": 0.005, "agentic": 0.005, "longctx": 0.02},
    "indic_00": swap("indic", 0.00),
    "indic_02": swap("indic", 0.02),
    "indic_04": swap("indic", 0.04),
    "indic_16": swap("indic", 0.16),
    "agentic_00": swap("agentic", 0.00),
    "agentic_02": swap("agentic", 0.02),
    "agentic_10": swap("agentic", 0.10),
    "indic_only": {"indic": 1.0},
    "agentic_only": {"agentic": 1.0},
}

# Arms replicated across seeds to give the headline comparison an error bar.
REPLICATED = ("v5", "opus_greedy")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--dmodel", type=int, default=384)
    ap.add_argument("--seeds", type=int, default=2, help="seeds for the replicated arms")
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()

    cfg = Config(n_layer=args.layers, d_model=args.dmodel, seq_len=args.seq)
    arms = {k: v for k, v in ARMS.items() if not args.only or k in args.only}

    RESULTS.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS / "ablation.json"
    results = json.loads(out_path.read_text()) if out_path.exists() else {}

    plan = []
    for name, mix in arms.items():
        n_seeds = args.seeds if name in REPLICATED else 1
        for seed in range(n_seeds):
            plan.append((name, mix, seed))

    print(f"{len(plan)} runs planned: {args.steps} steps x {args.batch} x {args.seq} "
          f"= {args.steps*args.batch*args.seq/1e6:.1f}M tokens each\n")

    t0 = time.time()
    for i, (name, mix, seed) in enumerate(plan, 1):
        key = f"{name}_s{seed}"
        if key in results:
            print(f"[{i}/{len(plan)}] {key} already done, skipping")
            continue
        print(f"[{i}/{len(plan)}] {key}  mixture={mix}")
        r = train_arm(
            name=name, mixture=mix, steps=args.steps, batch_size=args.batch,
            seq=args.seq, lr=args.lr, seed=seed, cfg=cfg,
        )
        results[key] = {k: v for k, v in r.items() if k != "history"}
        results[key]["final_train_loss"] = r["history"][-1]["loss"]
        results[key]["max_grad_norm"] = max(h["grad_norm"] for h in r["history"])
        out_path.write_text(json.dumps(results, indent=2))
        el = time.time() - t0
        print(f"    done in {r['wall_seconds']}s   elapsed {el/60:.1f}m   "
              f"eta {(el/i)*(len(plan)-i)/60:.1f}m\n", flush=True)

    print(f"all runs complete in {(time.time()-t0)/60:.1f} minutes -> {out_path}")


if __name__ == "__main__":
    main()
