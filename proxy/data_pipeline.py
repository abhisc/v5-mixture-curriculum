"""Turn cleaned lane text into byte shards, with a held-out split per lane.

The long-context lane is built here rather than downloaded, mirroring the specification's
claim that long context is a packing strategy over other lanes' tokens rather than a
source of new tokens: `longctx` is produced by concatenating whole code files and whole
STEM documents until they exceed the model's context window.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CLEAN = ROOT / "proxy" / "data" / "clean"
SHARDS = ROOT / "proxy" / "data" / "shards"

LANES = ("web", "code", "stem", "indic", "agentic", "longctx")
VAL_FRACTION = 0.06
LONGCTX_TARGET_BYTES = 3_000_000
LONGCTX_DOC_BYTES = 8192


def build_longctx() -> str:
    """Pack whole documents from the code and STEM lanes into long concatenations."""
    parts: list[str] = []
    pools = []
    for lane, weight in (("code", 0.7), ("stem", 0.3)):
        p = CLEAN / f"{lane}.txt"
        if p.exists():
            docs = [d for d in p.read_text(encoding="utf-8").split("\n\n") if len(d) > 300]
            pools.append((docs, weight))
    if not pools:
        return ""
    total = 0
    idx = [0] * len(pools)
    while total < LONGCTX_TARGET_BYTES:
        buf: list[str] = []
        size = 0
        progressed = False
        while size < LONGCTX_DOC_BYTES:
            for i, (docs, _) in enumerate(pools):
                if idx[i] >= len(docs):
                    continue
                d = docs[idx[i]]
                idx[i] += 1
                buf.append(d)
                size += len(d)
                progressed = True
                if size >= LONGCTX_DOC_BYTES:
                    break
            if not progressed:
                break
        if not progressed:
            break
        parts.append("<|pack|>\n" + "\n<|sep|>\n".join(buf))
        total += size
    return "\n\n".join(parts)


def main() -> None:
    SHARDS.mkdir(parents=True, exist_ok=True)
    manifest = {}

    lc = build_longctx()
    if lc:
        (CLEAN / "longctx.txt").write_text(lc, encoding="utf-8")

    for lane in LANES:
        p = CLEAN / f"{lane}.txt"
        if not p.exists():
            print(f"  {lane}: missing, skipped")
            continue
        raw = p.read_bytes()
        arr = np.frombuffer(raw, dtype=np.uint8)
        n_val = max(4096, int(len(arr) * VAL_FRACTION))
        # Take validation from the tail so training never sees it.
        train, val = arr[:-n_val], arr[-n_val:]
        np.save(SHARDS / f"{lane}_train.npy", train)
        np.save(SHARDS / f"{lane}_val.npy", val)
        manifest[lane] = {"train_bytes": int(train.size), "val_bytes": int(val.size)}
        print(f"  {lane:8s} train {train.size/1e6:7.2f} MB   val {val.size/1e6:6.3f} MB")

    (SHARDS / "manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
