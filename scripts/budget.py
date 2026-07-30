"""Budget, supply-audit and repetition math for the V5 mixture.

Everything the README asserts numerically is computed here. Run `python scripts/build_tables.py`
to regenerate the markdown tables.

The one non-obvious piece of modelling is `effective_tokens`, which implements the
data-constrained scaling law from Muennighoff et al. (2023):

    D' = U * (1 + R* * (1 - exp(-R / R*)))

where U is unique tokens, R is the number of *repetitions* (epochs - 1) and R* ~= 15.4.
It says repeated tokens are worth less than fresh ones, with the value of an extra epoch
decaying exponentially. We use it to convert a raw token allocation into the "fresh-token
equivalent" the model actually benefits from, which is what makes over-allocation visible
instead of invisible.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INVENTORY_PATH = ROOT / "inventory" / "inventory.json"
MIXTURE_PATH = ROOT / "inventory" / "mixture.json"


def lanes_of(mixture: dict) -> dict[str, dict]:
    """Lane configs, skipping the `_`-prefixed documentation keys."""
    return {k: v for k, v in mixture["lanes"].items() if not k.startswith("_")}


def load() -> tuple[dict, dict]:
    return (
        json.loads(INVENTORY_PATH.read_text()),
        json.loads(MIXTURE_PATH.read_text()),
    )


# --------------------------------------------------------------------------------------
# supply
# --------------------------------------------------------------------------------------

@dataclass
class LaneSupply:
    lane: str
    raw_tokens_b: float = 0.0
    unique_tokens_b: float = 0.0          # after overlap discount
    derived_tokens_b: float = 0.0         # re-packings; contribute no new supply
    datasets: list[dict] = field(default_factory=list)


def lane_supply(inventory: dict) -> dict[str, LaneSupply]:
    out: dict[str, LaneSupply] = {}
    for ds in inventory["datasets"]:
        lane = ds["lane"]
        sup = out.setdefault(lane, LaneSupply(lane=lane))
        raw = ds["unique_tokens_b"]
        effective = raw * (1.0 - ds["overlap_discount"])
        sup.raw_tokens_b += raw
        if ds["provenance"] == "derived":
            sup.derived_tokens_b += raw
        else:
            sup.unique_tokens_b += effective
        sup.datasets.append(ds)
    return out


def indic_tier_supply(inventory: dict) -> dict[str, float]:
    """Unique (post-discount) supply available per Indic tier from *natural* sources."""
    tiers: dict[str, float] = {}
    for ds in inventory["datasets"]:
        if ds["lane"] != "indic":
            continue
        tier = ds.get("indic_tier", "unverified")
        tiers[tier] = tiers.get(tier, 0.0) + ds["unique_tokens_b"] * (1.0 - ds["overlap_discount"])
    return tiers


# --------------------------------------------------------------------------------------
# repetition
# --------------------------------------------------------------------------------------

def effective_tokens(allocated_b: float, unique_b: float, r_star: float) -> float:
    """Fresh-token-equivalent value of `allocated_b` tokens drawn from `unique_b` unique tokens."""
    if unique_b <= 0:
        return 0.0
    if allocated_b <= unique_b:
        return allocated_b
    repetitions = allocated_b / unique_b - 1.0
    return unique_b * (1.0 + r_star * (1.0 - math.exp(-repetitions / r_star)))


def epochs(allocated_b: float, unique_b: float) -> float:
    return float("inf") if unique_b <= 0 else allocated_b / unique_b


# --------------------------------------------------------------------------------------
# the audit
# --------------------------------------------------------------------------------------

@dataclass
class LaneAudit:
    lane: str
    share: float
    allocated_b: float          # tokens this lane is budgeted
    natural_unique_b: float     # unique supply this lane owns, after overlap discount
    synthesis_fraction: float
    synthetic_target_b: float   # must be generated
    natural_target_b: float     # drawn from this lane's own natural supply
    repacked_in_b: float        # tokens borrowed from other lanes (already paid for there)
    lent_out_b: float           # tokens this lane supplies to a re-packed lane
    total_draw_b: float         # natural_target + lent_out: the real pressure on this lane's supply
    epochs_on_natural: float
    effective_b: float          # fresh-token-equivalent, counted once globally
    waste_b: float
    verdict: str


def audit(inventory: dict, mixture: dict) -> dict:
    total = mixture["budget"]["total_tokens_b"]
    pretrain_b = total * mixture["budget"]["splits"]["pretrain"]
    anneal_b = total * mixture["budget"]["splits"]["anneal"]
    post_b = total * mixture["budget"]["splits"]["post_training"]
    r_star = mixture["repetition_model"]["r_star"]
    soft_cap = mixture["repetition_model"]["epoch_soft_cap"]

    supply = lane_supply(inventory)
    lanes: dict[str, LaneAudit] = {}

    # A re-packed lane (long context) does not own most of its tokens: it borrows them from
    # other lanes. Compute those debits first so the lending lanes are charged for them.
    borrowed: dict[str, float] = {name: 0.0 for name in lanes_of(mixture)}
    native_only: dict[str, float] = {}
    for name, cfg in lanes_of(mixture).items():
        sm = cfg.get("supply_model")
        if not sm or sm.get("type") != "repacked":
            continue
        allocated = pretrain_b * cfg["share"]
        native_only[name] = allocated * sm["native_fraction"]
        for lender, frac in sm["borrows_from"].items():
            borrowed[lender] += allocated * frac

    eff_rates: dict[str, float] = {}
    # Lending lanes first, so the re-packed lane can price its borrowed tokens at the
    # lender's marginal value.
    order = sorted(lanes_of(mixture), key=lambda n: n in native_only)
    for name in order:
        cfg = lanes_of(mixture)[name]
        allocated = pretrain_b * cfg["share"]
        syn_frac = cfg["synthesis_fraction"]
        nat_unique = supply[name].unique_tokens_b
        repacked_in = 0.0

        if name in native_only:
            # A re-packed lane draws on its own supply only for the natively-long slice.
            # The rest is the same tokens as the lending lanes, packed at long sequence
            # length, and is charged to those lanes instead.
            nat_target = native_only[name]
            repacked_in = allocated - nat_target
            syn_target = 0.0
        else:
            syn_target = allocated * syn_frac
            nat_target = allocated - syn_target

        # Tokens lent to a re-packed lane come out of this lane's natural and synthetic
        # pools in the same ratio as the lane itself. Long agentic sessions are synthesised,
        # so they must not be charged against the lane's 1.2B of natural data.
        lent_out = borrowed[name]
        lent_natural = lent_out * (1.0 - syn_frac)
        lent_synthetic = lent_out * syn_frac
        total_draw = nat_target + lent_natural

        ep = epochs(total_draw, nat_unique)
        eff_draw = effective_tokens(total_draw, nat_unique, r_star)
        # Each drawn token is worth eff_draw/total_draw fresh tokens. Attribute that value to
        # whichever lane's budget actually spends it, so no token is counted twice globally.
        rate = eff_draw / total_draw if total_draw > 0 else 1.0
        # Value of one token lent to a re-packed lane: natural tokens are discounted by the
        # lender's repetition rate, synthesised tokens are fresh by construction.
        eff_rates[name] = rate * (1.0 - syn_frac) + syn_frac
        # A lane's effective total covers only the tokens its OWN budget line spends.
        # Lent tokens are spent by the borrowing lane and are counted there.
        eff = rate * nat_target + syn_target
        if repacked_in > 0:
            sm = cfg["supply_model"]
            eff += sum(
                allocated * frac * eff_rates[lender]
                for lender, frac in sm["borrows_from"].items()
            )

        if repacked_in > 0:
            verdict = f"RE-PACKED ({repacked_in:.0f}B borrowed, {nat_target:.1f}B native)"
        elif syn_frac >= 0.5:
            verdict = f"SYNTHESIS-DEPENDENT ({syn_frac*100:.0f}% generated)"
        elif ep <= 1.0:
            verdict = "MET with unique tokens"
        elif ep <= soft_cap:
            verdict = f"MET by repetition ({ep:.2f} epochs)"
        else:
            verdict = f"OVER-DRAWN ({ep:.2f} epochs, exceeds {soft_cap:g}-epoch soft cap)"

        lanes[name] = LaneAudit(
            lane=name,
            share=cfg["share"],
            allocated_b=allocated,
            natural_unique_b=nat_unique,
            synthesis_fraction=syn_frac,
            synthetic_target_b=syn_target + lent_synthetic,
            natural_target_b=nat_target,
            repacked_in_b=repacked_in,
            lent_out_b=lent_out,
            total_draw_b=total_draw,
            epochs_on_natural=ep,
            effective_b=eff,
            waste_b=allocated - eff,
            verdict=verdict,
        )

    return {
        "total_b": total,
        "pretrain_b": pretrain_b,
        "anneal_b": anneal_b,
        "post_b": post_b,
        "lanes": lanes,
        "supply": supply,
        "indic_tiers": indic_tier_supply(inventory),
    }


# --------------------------------------------------------------------------------------
# curriculum consistency
# --------------------------------------------------------------------------------------

def realized_shares(mixture: dict) -> dict[str, float]:
    """Token-weighted lane shares actually delivered by the staged curriculum."""
    stages = mixture["curriculum"]["stages"]
    pretrain_stages = [s for s in stages if not s["name"].startswith("S5")]
    total = sum(s["tokens_b"] for s in pretrain_stages)
    out: dict[str, float] = {lane: 0.0 for lane in lanes_of(mixture)}
    for s in pretrain_stages:
        for lane, frac in s["mix"].items():
            out[lane] += frac * s["tokens_b"]
    return {lane: v / total for lane, v in out.items()}


def validate(inventory: dict, mixture: dict, tolerance_pp: float = 0.15) -> list[str]:
    """Returns a list of problems. Empty list means the spec is internally consistent."""
    problems: list[str] = []

    share_sum = sum(c["share"] for c in lanes_of(mixture).values())
    if abs(share_sum - 1.0) > 1e-9:
        problems.append(f"lane shares sum to {share_sum:.6f}, not 1.0")

    split_sum = sum(mixture["budget"]["splits"].values())
    if abs(split_sum - 1.0) > 1e-9:
        problems.append(f"budget splits sum to {split_sum:.6f}, not 1.0")

    for name, cfg in lanes_of(mixture).items():
        if cfg["floor"] > cfg["share"] + 1e-9:
            problems.append(f"lane {name}: floor {cfg['floor']} exceeds share {cfg['share']}")

    stage_order = [s["name"].split()[0] for s in mixture["curriculum"]["stages"]]
    for stage in mixture["curriculum"]["stages"]:
        s = sum(stage["mix"].values())
        if abs(s - 1.0) > 1e-6:
            problems.append(f"stage {stage['name']} mix sums to {s:.6f}, not 1.0")
        if stage["name"].startswith("S5"):
            continue
        stage_idx = stage_order.index(stage["name"].split()[0])
        for lane, cfg in lanes_of(mixture).items():
            if stage_idx < stage_order.index(cfg.get("floor_active_from", "S1")):
                continue
            if stage["mix"].get(lane, 0.0) + 1e-9 < cfg["floor"]:
                problems.append(
                    f"stage {stage['name']} violates {lane} floor: "
                    f"{stage['mix'].get(lane, 0.0):.3f} < {cfg['floor']:.3f}"
                )

    pretrain_stage_tokens = sum(
        s["tokens_b"] for s in mixture["curriculum"]["stages"] if not s["name"].startswith("S5")
    )
    expected_pretrain = mixture["budget"]["total_tokens_b"] * mixture["budget"]["splits"]["pretrain"]
    if abs(pretrain_stage_tokens - expected_pretrain) > 1e-6:
        problems.append(
            f"curriculum stages total {pretrain_stage_tokens:g}B but pretrain budget is {expected_pretrain:g}B"
        )

    realized = realized_shares(mixture)
    for lane, cfg in lanes_of(mixture).items():
        delta_pp = abs(realized[lane] - cfg["share"]) * 100
        if delta_pp > tolerance_pp:
            problems.append(
                f"curriculum delivers {lane} at {realized[lane]*100:.2f}% but budget claims "
                f"{cfg['share']*100:.2f}% (delta {delta_pp:.2f}pp > {tolerance_pp}pp)"
            )

    reserve = mixture["anneal_reserve"]
    comp_sum = sum(c["tokens_b"] for c in reserve["components"])
    if abs(comp_sum - reserve["total_tokens_b"]) > 1e-6:
        problems.append(
            f"anneal components sum to {comp_sum:g}B, declared total is {reserve['total_tokens_b']:g}B"
        )
    anneal_budget = mixture["budget"]["total_tokens_b"] * mixture["budget"]["splits"]["anneal"]
    if abs(reserve["total_tokens_b"] - anneal_budget) > 1e-6:
        problems.append(
            f"anneal reserve {reserve['total_tokens_b']:g}B != anneal budget split {anneal_budget:g}B"
        )

    indic_tiers = lanes_of(mixture)["indic"]["tiers"]
    tier_sum = sum(t["share"] for t in indic_tiers.values())
    if abs(tier_sum - 1.0) > 1e-9:
        problems.append(f"indic tier shares sum to {tier_sum:.6f}, not 1.0")

    band_sum = sum(
        b["share_of_reasoning_tokens"]
        for k, b in mixture["reasoning_bands"].items()
        if not k.startswith("_")
    )
    if abs(band_sum - 1.0) > 1e-9:
        problems.append(f"reasoning band shares sum to {band_sum:.6f}, not 1.0")

    return problems


if __name__ == "__main__":
    inv, mix = load()
    issues = validate(inv, mix)
    if issues:
        print("SPEC INCONSISTENT:")
        for p in issues:
            print("  -", p)
        raise SystemExit(1)
    print("spec is internally consistent")
    a = audit(inv, mix)
    tot_alloc = tot_eff = 0.0
    for lane in mix["lanes"]:
        if lane.startswith("_"):
            continue
        la = a["lanes"][lane]
        tot_alloc += la.allocated_b
        tot_eff += la.effective_b
        print(
            f"{lane:9s} {la.share*100:5.2f}%  alloc={la.allocated_b:7.1f}B  "
            f"unique={la.natural_unique_b:7.1f}B  draw={la.total_draw_b:7.1f}B  "
            f"epochs={la.epochs_on_natural:5.2f}  eff={la.effective_b:7.1f}B  {la.verdict}"
        )
    print(f"{'TOTAL':9s}        alloc={tot_alloc:7.1f}B  "
          f"eff={tot_eff:7.1f}B  ({tot_eff/tot_alloc*100:.1f}% of allocation is fresh-token-equivalent)")
    print("\nIndic tier supply (natural, post-discount):")
    for tier, tokens in a["indic_tiers"].items():
        print(f"  {tier:11s} {tokens:7.1f}B")
