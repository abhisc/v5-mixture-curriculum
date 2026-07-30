"""Targeted cleaning pipeline. Emits results/cleaning_report.json.

This is the small-scale implementation of the cleaning rules the specification commits to.
Cleaning effort is aimed at the lanes the supply audit shows to be starved, because that
is where a rejected document is expensive: throwing away a duplicate web page costs
nothing when the web lane runs at 0.25 epochs, but throwing away an Indic document
directly raises the Indic lane's epoch count.

Filters, in order:
  1. length      - documents too short to contain a complete thought
  2. exact dedup - sha1 over normalised whitespace
  3. near dedup  - 5-gram MinHash with banded LSH, Jaccard >= 0.8
  4. script gate - Indic docs must be predominantly in a native Indic script. Documents
                   that fail are RECLASSIFIED into the romanised tier, not deleted:
                   transliterated Indic text is genuinely useful (a large share of real
                   Indian user input is Romanised) but it is not native-script data and
                   must not be counted as such. This filter exists because language ID
                   alone cannot tell the difference - it labels romanised Sanskrit as
                   "Sanskrit" and lets it into the verified native tier.
  5. scrub       - absolute local paths and $HOME leakage in generated agentic traces
  6. repetition  - documents whose top line or top 3-gram dominates the document
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field, asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "proxy" / "data" / "raw"
CLEAN = ROOT / "proxy" / "data" / "clean"
RESULTS = ROOT / "results"

# Unicode blocks for the scripts used by major Indian languages.
INDIC_RANGES = [
    (0x0900, 0x097F),  # Devanagari  (Hindi, Marathi, Sanskrit)
    (0x0980, 0x09FF),  # Bengali
    (0x0A00, 0x0A7F),  # Gurmukhi
    (0x0A80, 0x0AFF),  # Gujarati
    (0x0B00, 0x0B7F),  # Odia
    (0x0B80, 0x0BFF),  # Tamil
    (0x0C00, 0x0C7F),  # Telugu
    (0x0C80, 0x0CFF),  # Kannada
    (0x0D00, 0x0D7F),  # Malayalam
    (0x0600, 0x06FF),  # Arabic      (Urdu)
]

MIN_CHARS = {"web": 200, "code": 200, "stem": 80, "indic": 60, "agentic": 200, "longctx": 200}
INDIC_SCRIPT_MIN = 0.55
NEAR_DUP_THRESHOLD = 0.80
NUM_HASHES = 64
BANDS = 16

HOME_PAT = re.compile(r"/Users/[^/\s\"']+|/home/[^/\s\"']+")
ABS_SCRATCH_PAT = re.compile(r"(/[\w.\-]+)+/proxy/data/scratch")


@dataclass
class LaneReport:
    lane: str
    docs_in: int = 0
    docs_out: int = 0
    bytes_in: int = 0
    bytes_out: int = 0
    dropped_short: int = 0
    dropped_exact_dup: int = 0
    dropped_near_dup: int = 0
    reclassified_romanised: int = 0
    dropped_repetitive: int = 0
    scrubbed_docs: int = 0
    notes: list[str] = field(default_factory=list)

    @property
    def retention(self) -> float:
        return self.docs_out / self.docs_in if self.docs_in else 0.0


def indic_ratio(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    hits = sum(
        1 for c in letters if any(lo <= ord(c) <= hi for lo, hi in INDIC_RANGES)
    )
    return hits / len(letters)


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip().lower()


def shingles(text: str, k: int = 5) -> set[int]:
    toks = norm(text).split()
    if len(toks) < k:
        return {hash(" ".join(toks))} if toks else set()
    return {hash(" ".join(toks[i : i + k])) for i in range(len(toks) - k + 1)}


def minhash(sh: set[int], n: int = NUM_HASHES) -> tuple[int, ...]:
    if not sh:
        return tuple([0] * n)
    # n independent hash permutations via multiply-xor mixing.
    out = []
    for i in range(n):
        salt = (i * 0x9E3779B1) & 0xFFFFFFFF
        out.append(min(((h ^ salt) * 0x85EBCA6B) & 0xFFFFFFFF for h in sh))
    return tuple(out)


def jaccard_est(a: tuple[int, ...], b: tuple[int, ...]) -> float:
    return sum(1 for x, y in zip(a, b) if x == y) / len(a)


def is_repetitive(text: str) -> bool:
    lines = [l for l in text.splitlines() if l.strip()]
    if len(lines) >= 8:
        top = Counter(lines).most_common(1)[0][1]
        if top / len(lines) > 0.5:
            return True
    toks = text.split()
    if len(toks) >= 60:
        grams = Counter(" ".join(toks[i : i + 3]) for i in range(len(toks) - 2))
        if grams.most_common(1)[0][1] / max(1, len(grams)) > 0.25:
            return True
    return False


def read_docs(lane: str) -> list[str]:
    """Split each lane's raw files into documents using the lane's natural boundary."""
    d = RAW / lane
    if not d.exists():
        return []
    docs: list[str] = []
    if lane == "code":
        for f in d.glob("*.txt"):
            text = f.read_text(encoding="utf-8", errors="ignore")
            parts = re.split(r"(?=<\|file:)", text) if "<|file:" in text else text.split("\n\n\n")
            docs += [p for p in parts if p.strip()]
    elif lane == "agentic":
        for f in d.glob("*.txt"):
            text = f.read_text(encoding="utf-8", errors="ignore")
            docs += [p for p in re.split(r"(?=<\|agentic\|>)", text) if p.strip()]
    elif lane == "indic":
        f = d / "indic_text.txt"
        if f.exists():
            docs += [p for p in f.read_text(encoding="utf-8").split("\n\n") if p.strip()]
    elif lane == "stem":
        f = d / "stem_text.txt"
        if f.exists():
            docs += [p for p in f.read_text(encoding="utf-8").split("\n\n") if p.strip()]
    elif lane == "web":
        f = d / "wikitext2.txt"
        if f.exists():
            text = f.read_text(encoding="utf-8")
            # WikiText marks articles with ` = Title = ` headings.
            parts = re.split(r"\n(?= = [^=])", text)
            docs += [p for p in parts if p.strip()]
    return docs


def clean_lane(lane: str) -> tuple[list[str], LaneReport]:
    rep = LaneReport(lane=lane)
    docs = read_docs(lane)
    rep.docs_in = len(docs)
    rep.bytes_in = sum(len(d.encode()) for d in docs)

    seen_exact: set[str] = set()
    bands: dict[tuple[int, int, tuple[int, ...]], list[tuple[int, ...]]] = {}
    kept: list[str] = []
    romanised: list[str] = []
    rows_per_band = NUM_HASHES // BANDS

    for doc in docs:
        if len(doc) < MIN_CHARS[lane]:
            rep.dropped_short += 1
            continue

        if lane == "agentic":
            scrubbed = ABS_SCRATCH_PAT.sub("/workspace", doc)
            scrubbed = HOME_PAT.sub("/home/user", scrubbed)
            if scrubbed != doc:
                rep.scrubbed_docs += 1
            doc = scrubbed

        if lane == "indic" and indic_ratio(doc) < INDIC_SCRIPT_MIN:
            # Not native script. Route to the romanised tier instead of discarding.
            rep.reclassified_romanised += 1
            romanised.append(doc)
            continue

        h = hashlib.sha1(norm(doc).encode()).hexdigest()
        if h in seen_exact:
            rep.dropped_exact_dup += 1
            continue
        seen_exact.add(h)

        if is_repetitive(doc):
            rep.dropped_repetitive += 1
            continue

        sig = minhash(shingles(doc))
        dup = False
        keys = [(b, 0, sig[b * rows_per_band : (b + 1) * rows_per_band]) for b in range(BANDS)]
        for key in keys:
            for other in bands.get(key, ()):
                if jaccard_est(sig, other) >= NEAR_DUP_THRESHOLD:
                    dup = True
                    break
            if dup:
                break
        if dup:
            rep.dropped_near_dup += 1
            continue
        for key in keys:
            bands.setdefault(key, []).append(sig)

        kept.append(doc)

    rep.docs_out = len(kept)
    rep.bytes_out = sum(len(d.encode()) for d in kept)
    if romanised:
        rep.notes.append(
            f"{len(romanised)} documents reclassified from native-script to romanised tier"
        )
        (CLEAN / "indic_romanised.txt").write_text("\n\n".join(romanised), encoding="utf-8")
    return kept, rep


def main() -> None:
    CLEAN.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    reports = {}
    for lane in ("web", "code", "stem", "indic", "agentic"):
        kept, rep = clean_lane(lane)
        if not kept:
            print(f"{lane}: no documents found, skipping")
            continue
        (CLEAN / f"{lane}.txt").write_text("\n\n".join(kept), encoding="utf-8")
        reports[lane] = asdict(rep) | {"retention": round(rep.retention, 4)}
        print(
            f"{lane:8s} docs {rep.docs_in:6d} -> {rep.docs_out:6d} "
            f"({rep.retention*100:5.1f}%)  bytes {rep.bytes_in/1e6:6.2f} -> {rep.bytes_out/1e6:6.2f} MB  "
            f"[short {rep.dropped_short}, exact {rep.dropped_exact_dup}, near {rep.dropped_near_dup}, "
            f"romanised {rep.reclassified_romanised}, repetitive {rep.dropped_repetitive}, scrubbed {rep.scrubbed_docs}]"
        )
    (RESULTS / "cleaning_report.json").write_text(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
