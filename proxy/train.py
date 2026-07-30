"""Train one arm of the mixture ablation and evaluate it per lane.

Every arm is identical except for the sampling proportions over lanes: same seed, same
architecture, same optimizer, same number of optimizer steps, same total tokens. That is
what makes the resulting differences attributable to the mixture rather than to compute.

Metrics
-------
bpb[lane]
    Held-out bits per byte on that lane's validation split. Lower is better. This is the
    pretraining-loss analogue of the benchmark the lane is meant to win.

indic_script_fidelity
    Feed the model real Devanagari prompts and let it continue. Measure the fraction of
    generated alphabetic characters that are in an Indic script. A model that has been
    starved of Indic data does not merely get a worse Indic loss, it code-switches back
    into Latin script mid-sentence. This measures that failure directly, and it is the
    proxy-scale analogue of script integrity on IndicGenBench.

agentic_structure_score
    Feed the model an agentic goal prefix and check whether the continuation reproduces
    the action/observation protocol. This is the proxy-scale analogue of BFCL schema
    adherence: can the model emit a well-formed tool call at all.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from pathlib import Path

import numpy as np
import torch

from model import ByteLM, Config

ROOT = Path(__file__).resolve().parents[1]
SHARDS = ROOT / "proxy" / "data" / "shards"
RUNS = ROOT / "proxy" / "runs"

LANES = ("web", "code", "stem", "indic", "agentic", "longctx")

INDIC_RANGES = [
    (0x0900, 0x097F), (0x0980, 0x09FF), (0x0A00, 0x0A7F), (0x0A80, 0x0AFF),
    (0x0B00, 0x0B7F), (0x0B80, 0x0BFF), (0x0C00, 0x0C7F), (0x0C80, 0x0CFF),
    (0x0D00, 0x0D7F), (0x0600, 0x06FF),
]


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class LaneData:
    def __init__(self, device: torch.device):
        self.train: dict[str, np.ndarray] = {}
        self.val: dict[str, np.ndarray] = {}
        for lane in LANES:
            tp, vp = SHARDS / f"{lane}_train.npy", SHARDS / f"{lane}_val.npy"
            if tp.exists():
                self.train[lane] = np.load(tp)
                self.val[lane] = np.load(vp)
        self.device = device

    def batch(self, lane_counts: dict[str, int], seq: int, rng: np.random.Generator):
        xs, ys = [], []
        for lane, count in lane_counts.items():
            if count <= 0 or lane not in self.train:
                continue
            arr = self.train[lane]
            hi = len(arr) - seq - 1
            starts = rng.integers(0, hi, size=count)
            for s in starts:
                chunk = arr[s : s + seq + 1].astype(np.int64)
                xs.append(chunk[:-1])
                ys.append(chunk[1:])
        x = torch.from_numpy(np.stack(xs)).to(self.device, non_blocking=True)
        y = torch.from_numpy(np.stack(ys)).to(self.device, non_blocking=True)
        return x, y


def sample_counts(
    lanes: list[str], probs: np.ndarray, batch_size: int, rng: np.random.Generator
) -> dict[str, int]:
    """Draw each sequence's lane from the mixture distribution.

    Deterministic largest-remainder allocation cannot represent a lane whose share is
    below 1/batch_size: at batch 24 a 2% lane rounds to either 0% or 4.2%, which would
    make the low end of the Indic sweep unmeasurable. Multinomial sampling makes the
    expected proportion exact for any share, with sampling noise that vanishes over
    thousands of steps.
    """
    draw = rng.multinomial(batch_size, probs)
    return dict(zip(lanes, (int(c) for c in draw)))


@torch.no_grad()
def eval_bpb(model: ByteLM, data: LaneData, seq: int, batches: int, seed: int) -> dict[str, float]:
    model.eval()
    rng = np.random.default_rng(seed)
    out = {}
    for lane, arr in data.val.items():
        if len(arr) < seq + 2:
            continue
        losses = []
        n = min(batches, max(1, (len(arr) - seq - 1) // seq))
        for _ in range(n):
            starts = rng.integers(0, len(arr) - seq - 1, size=8)
            xs = np.stack([arr[s : s + seq + 1].astype(np.int64) for s in starts])
            x = torch.from_numpy(xs[:, :-1]).to(model.tok.weight.device)
            y = torch.from_numpy(xs[:, 1:]).to(model.tok.weight.device)
            _, loss = model(x, y)
            losses.append(loss.item())
        out[lane] = float(np.mean(losses) / math.log(2))  # nats -> bits per byte
    model.train()
    return out


def _is_indic(ch: str) -> bool:
    return any(lo <= ord(ch) <= hi for lo, hi in INDIC_RANGES)


@torch.no_grad()
def eval_script_fidelity(model: ByteLM, data: LaneData, n_prompts: int, gen_len: int, seed: int) -> float:
    """Fraction of generated letters that stay in an Indic script."""
    if "indic" not in data.val:
        return float("nan")
    model.eval()
    rng = np.random.default_rng(seed)
    arr = data.val["indic"]
    prompt_len = 128
    prompts = []
    for _ in range(n_prompts):
        s = int(rng.integers(0, len(arr) - prompt_len - 1))
        prompts.append(arr[s : s + prompt_len].astype(np.int64))
    x = torch.from_numpy(np.stack(prompts)).to(model.tok.weight.device)
    out = model.generate(x, max_new=gen_len, temperature=0.8)
    gen = out[:, prompt_len:].cpu().numpy().astype(np.uint8)
    indic = total = 0
    for row in gen:
        text = bytes(row).decode("utf-8", errors="ignore")
        for ch in text:
            if ch.isalpha():
                total += 1
                indic += int(_is_indic(ch))
    model.train()
    return indic / total if total else 0.0


AGENTIC_PROMPT = b"<|agentic|>\n<|goal|> Count the ERROR lines in logs/ and report the count.\n"


@torch.no_grad()
def eval_agentic_structure(model: ByteLM, n: int, gen_len: int, seed: int) -> float:
    """Does the continuation reproduce the think/action/observation protocol?"""
    model.eval()
    torch.manual_seed(seed)
    x = torch.from_numpy(
        np.stack([np.frombuffer(AGENTIC_PROMPT, dtype=np.uint8).astype(np.int64)] * n)
    ).to(model.tok.weight.device)
    out = model.generate(x, max_new=gen_len, temperature=0.8)
    gen = out[:, x.shape[1] :].cpu().numpy().astype(np.uint8)
    score = 0.0
    for row in gen:
        text = bytes(row).decode("utf-8", errors="ignore")
        hits = sum(
            1 for tag in ("<|think|>", "<|action|>", "<|observation|>", "exit=")
            if tag in text
        )
        score += hits / 4.0
    model.train()
    return score / n


def train_arm(
    name: str,
    mixture: dict[str, float],
    steps: int,
    batch_size: int,
    seq: int,
    lr: float,
    seed: int,
    cfg: Config,
    eval_batches: int = 12,
    log_every: int = 100,
) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = pick_device()
    data = LaneData(device)

    mixture = {k: v for k, v in mixture.items() if k in data.train}
    total = sum(mixture.values())
    mixture = {k: v / total for k, v in mixture.items()}
    lane_names = list(mixture)
    probs = np.array([mixture[k] for k in lane_names], dtype=np.float64)
    probs = probs / probs.sum()
    realized = {k: 0 for k in lane_names}

    model = ByteLM(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1)
    warmup = max(20, steps // 20)

    def lr_at(it: int) -> float:
        if it < warmup:
            return lr * it / warmup
        p = (it - warmup) / max(1, steps - warmup)
        return 0.1 * lr + 0.9 * lr * 0.5 * (1 + math.cos(math.pi * p))

    rng = np.random.default_rng(seed)
    t0 = time.time()
    history = []
    for it in range(steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(it)
        counts = sample_counts(lane_names, probs, batch_size, rng)
        for k, v in counts.items():
            realized[k] += v
        x, y = data.batch(counts, seq, rng)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if it % log_every == 0 or it == steps - 1:
            history.append({"step": it, "loss": loss.item(), "grad_norm": float(gnorm)})
            print(
                f"    [{name}] step {it:5d}/{steps}  loss {loss.item():.4f}  "
                f"gn {float(gnorm):.2f}  {time.time()-t0:6.1f}s",
                flush=True,
            )

    tokens_seen = steps * batch_size * seq
    bpb = eval_bpb(model, data, seq, eval_batches, seed=1234)
    fidelity = eval_script_fidelity(model, data, n_prompts=16, gen_len=192, seed=1234)
    structure = eval_agentic_structure(model, n=8, gen_len=192, seed=1234)

    result = {
        "arm": name,
        "seed": seed,
        "mixture": mixture,
        "realized_mixture": {k: v / max(1, sum(realized.values())) for k, v in realized.items()},
        "steps": steps,
        "batch_size": batch_size,
        "seq_len": seq,
        "tokens_seen": tokens_seen,
        "params_non_embedding": model.n_params(),
        "wall_seconds": round(time.time() - t0, 1),
        "bpb": bpb,
        "indic_script_fidelity": fidelity,
        "agentic_structure_score": structure,
        "history": history,
    }
    RUNS.mkdir(parents=True, exist_ok=True)
    (RUNS / f"{name}_s{seed}.json").write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="smoke")
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--dmodel", type=int, default=384)
    args = ap.parse_args()

    cfg = Config(n_layer=args.layers, d_model=args.dmodel, seq_len=args.seq)
    mix = {"web": 0.35, "code": 0.32, "stem": 0.18, "indic": 0.08, "agentic": 0.05, "longctx": 0.02}
    r = train_arm(args.name, mix, args.steps, args.batch, args.seq, args.lr, args.seed, cfg)
    print(json.dumps({k: v for k, v in r.items() if k != "history"}, indent=2))
