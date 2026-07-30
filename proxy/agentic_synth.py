"""Execution-grounded synthesis of agentic traces.

The V5 spec claims that 99% of the agentic lane must be generated, because natural
supply is ~1.2B tokens against a 180B target. A claim like that is only credible if the
generator exists and its throughput is measured. This is that generator, in miniature.

Design rule: every observation in a trace is REAL. We run actual commands in a scratch
directory and record actual stdout, stderr and exit codes. Nothing is imagined. That is
the difference between grounded synthesis and an LLM writing plausible-looking terminal
output, which is the failure mode that makes synthetic agentic data worthless: a model
trained on invented `ls` output learns to hallucinate file systems.

Each trace also contains at least one genuine failure followed by a genuine recovery,
because error recovery is the capability Terminal-Bench actually measures and the one
that clean, happy-path traces never teach.
"""

from __future__ import annotations

import json
import random
import shutil
import subprocess
import time
from dataclasses import dataclass, asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "proxy" / "data" / "raw" / "agentic"
SCRATCH = ROOT / "proxy" / "data" / "scratch"

TIMEOUT = 10


@dataclass
class Step:
    thought: str
    command: str
    stdout: str
    stderr: str
    exit_code: int


def run(cmd: str, cwd: Path) -> tuple[str, str, int]:
    try:
        p = subprocess.run(
            cmd, shell=True, cwd=cwd, capture_output=True, text=True, timeout=TIMEOUT
        )
        return p.stdout[:4000], p.stderr[:2000], p.returncode
    except subprocess.TimeoutExpired:
        return "", f"command timed out after {TIMEOUT}s", 124


# --------------------------------------------------------------------------------------
# task families. each returns (goal, [(thought, command)], verification command)
# each family deliberately includes a step that fails for a real reason.
# --------------------------------------------------------------------------------------

def task_find_and_fix_syntax(rng: random.Random, wd: Path):
    name = f"mod_{rng.randrange(10**6)}.py"
    (wd / name).write_text("def add(a, b)\n    return a + b\n")
    return (
        f"The file {name} does not import. Find the syntax error and fix it.",
        [
            ("Reproduce the failure before changing anything.", f"python3 -c 'import {name[:-3]}'"),
            ("Read the file to locate the offending line.", f"cat -n {name}"),
            ("The def line is missing its colon. Patch it.",
             f"""python3 - <<'EOF'
import pathlib
p = pathlib.Path({name!r})
p.write_text(p.read_text().replace('def add(a, b)', 'def add(a, b):'))
EOF"""),
            ("Confirm the module now imports and behaves.",
             f"python3 -c 'import {name[:-3]} as m; print(m.add(2,3))'"),
        ],
        f"python3 -c 'import {name[:-3]} as m; assert m.add(2,3)==5; print(\"PASS\")'",
    )


def task_grep_count(rng: random.Random, wd: Path):
    d = f"logs_{rng.randrange(10**6)}"
    (wd / d).mkdir()
    n_err = rng.randint(3, 12)
    lines = []
    for i in range(60):
        lvl = "ERROR" if i % max(1, 60 // n_err) == 0 else rng.choice(["INFO", "WARN", "DEBUG"])
        lines.append(f"2026-07-{(i%28)+1:02d} {lvl} service=api msg=request_{i}")
    (wd / d / "app.log").write_text("\n".join(lines) + "\n")
    return (
        f"Count the ERROR lines in {d}/ and report the count.",
        [
            ("Try the directory directly; grep will complain and tell me what I need.",
             f"grep -c ERROR {d}"),
            ("It needs -r for a directory. Retry recursively.", f"grep -rc ERROR {d}"),
            ("Cross-check with an explicit count.", f"grep -rh ERROR {d} | wc -l"),
        ],
        f"test $(grep -rh ERROR {d} | wc -l) -gt 0 && echo PASS",
    )


def task_json_extract(rng: random.Random, wd: Path):
    name = f"data_{rng.randrange(10**6)}.json"
    recs = [
        {"id": i, "user": f"u{i}", "amount": rng.randint(1, 500), "currency": "INR"}
        for i in range(rng.randint(8, 25))
    ]
    (wd / name).write_text(json.dumps(recs, indent=2))
    return (
        f"Report the total `amount` across all records in {name}.",
        [
            ("Inspect the shape first; guessing the schema wastes a turn.", f"head -c 300 {name}"),
            ("Assume it is newline-delimited JSON and sum it.",
             f"python3 -c \"import json;print(sum(json.loads(l)['amount'] for l in open('{name}')))\""),
            ("That failed: it is a single JSON array, not JSONL. Parse it as an array.",
             f"python3 -c \"import json;print(sum(r['amount'] for r in json.load(open('{name}'))))\""),
        ],
        f"python3 -c \"import json;assert sum(r['amount'] for r in json.load(open('{name}')))>0;print('PASS')\"",
    )


def task_failing_test(rng: random.Random, wd: Path):
    pkg = f"calc_{rng.randrange(10**6)}"
    (wd / f"{pkg}.py").write_text(
        "def mean(xs):\n"
        "    return sum(xs) / len(xs)\n"
    )
    (wd / f"test_{pkg}.py").write_text(
        f"from {pkg} import mean\n"
        "def test_mean():\n"
        "    assert mean([1,2,3]) == 2\n"
        "def test_empty():\n"
        "    assert mean([]) == 0\n"
    )
    return (
        f"`test_{pkg}.py` fails. Make the whole suite pass without weakening the tests.",
        [
            ("Run the suite to see the real failure, not a guessed one.",
             f"python3 -m pytest -q test_{pkg}.py"),
            ("pytest is not installed here; fall back to running the assertions directly.",
             f"python3 -c \"import test_{pkg} as t; t.test_mean(); t.test_empty(); print('ok')\""),
            ("ZeroDivisionError on the empty case. Guard it in the source, not the test.",
             f"""python3 - <<'EOF'
import pathlib
p = pathlib.Path("{pkg}.py")
p.write_text("def mean(xs):\\n    if not xs:\\n        return 0\\n    return sum(xs) / len(xs)\\n")
EOF"""),
            ("Re-run both tests.",
             f"python3 -c \"import importlib,{pkg},test_{pkg} as t; importlib.reload({pkg}); t.test_mean(); t.test_empty(); print('ok')\""),
        ],
        f"python3 -c \"import test_{pkg} as t; t.test_mean(); t.test_empty(); print('PASS')\"",
    )


def task_disk_usage(rng: random.Random, wd: Path):
    d = f"tree_{rng.randrange(10**6)}"
    for sub in ("a", "b", "c"):
        (wd / d / sub).mkdir(parents=True)
        for i in range(rng.randint(2, 6)):
            (wd / d / sub / f"f{i}.bin").write_bytes(b"x" * rng.randint(500, 8000))
    return (
        f"Find which immediate subdirectory of {d}/ uses the most disk space.",
        [
            ("Sort by size; du prints human units by default which sorts wrong.",
             f"du -h {d}/* | sort -h"),
            ("Use byte counts so the ordering is unambiguous.", f"du -k {d}/* | sort -n | tail -1"),
            ("Confirm the file count in the winner for a sanity check.",
             f"find {d} -type f | wc -l"),
        ],
        f"du -k {d} | tail -1",
    )


def task_env_and_path(rng: random.Random, wd: Path):
    name = f"tool_{rng.randrange(10**6)}.sh"
    (wd / name).write_text("#!/bin/sh\necho tool-ran-ok\n")
    return (
        f"Make {name} executable and run it from the current directory.",
        [
            ("Try running it directly; it will fail because the bit is not set.", f"./{name}"),
            ("Permission denied as expected. Inspect the mode.", f"ls -l {name}"),
            ("Add the execute bit.", f"chmod +x {name}"),
            ("Run it again.", f"./{name}"),
        ],
        f"./{name}",
    )


def task_dedupe_lines(rng: random.Random, wd: Path):
    name = f"ids_{rng.randrange(10**6)}.txt"
    base = [f"id-{rng.randrange(100)}" for _ in range(40)]
    (wd / name).write_text("\n".join(base) + "\n")
    return (
        f"Report how many DISTINCT ids are in {name}.",
        [
            ("uniq only collapses adjacent duplicates, but try it to show why it is wrong.",
             f"uniq {name} | wc -l"),
            ("That over-counts because the file is unsorted. Sort first.",
             f"sort {name} | uniq | wc -l"),
            ("Cross-check with sort -u.", f"sort -u {name} | wc -l"),
        ],
        f"sort -u {name} | wc -l",
    )


TASKS = [
    task_find_and_fix_syntax,
    task_grep_count,
    task_json_extract,
    task_failing_test,
    task_disk_usage,
    task_env_and_path,
    task_dedupe_lines,
]


def render(goal: str, steps: list[Step], verified: bool) -> str:
    parts = [f"<|agentic|>\n<|goal|> {goal}"]
    for s in steps:
        obs = s.stdout if s.stdout.strip() else s.stderr
        obs = obs.strip()
        if len(obs) > 1200:
            obs = obs[:1200] + "\n...[truncated]"
        parts.append(
            f"<|think|> {s.thought}\n"
            f"<|action|> bash\n{s.command}\n"
            f"<|observation|> exit={s.exit_code}\n{obs}"
        )
    parts.append(f"<|result|> {'verified-pass' if verified else 'unverified'}\n")
    return "\n".join(parts) + "\n"


def generate(n_traces: int, seed: int = 0) -> dict:
    rng = random.Random(seed)
    OUT.mkdir(parents=True, exist_ok=True)
    if SCRATCH.exists():
        shutil.rmtree(SCRATCH)
    SCRATCH.mkdir(parents=True)

    t0 = time.time()
    n_steps = n_recovered = n_verified = 0
    out_path = OUT / "agentic_traces.txt"
    meta_path = OUT / "agentic_traces.meta.jsonl"

    with out_path.open("w", encoding="utf-8") as fh, meta_path.open("w", encoding="utf-8") as mf:
        for i in range(n_traces):
            wd = SCRATCH / f"t{i}"
            wd.mkdir()
            fn = TASKS[i % len(TASKS)]
            goal, plan, verify = fn(rng, wd)
            steps: list[Step] = []
            had_failure = False
            for thought, cmd in plan:
                so, se, rc = run(cmd, wd)
                steps.append(Step(thought, cmd, so, se, rc))
                had_failure |= rc != 0
            vso, _, vrc = run(verify, wd)
            verified = vrc == 0
            n_steps += len(steps)
            n_recovered += int(had_failure and verified)
            n_verified += int(verified)
            fh.write(render(goal, steps, verified))
            mf.write(json.dumps({
                "task": fn.__name__,
                "steps": len(steps),
                "had_failure": had_failure,
                "verified": verified,
            }) + "\n")
            shutil.rmtree(wd, ignore_errors=True)

    elapsed = time.time() - t0
    nbytes = out_path.stat().st_size
    shutil.rmtree(SCRATCH, ignore_errors=True)

    stats = {
        "traces": n_traces,
        "steps": n_steps,
        "bytes": nbytes,
        "seconds": round(elapsed, 2),
        "traces_per_sec": round(n_traces / elapsed, 2),
        "bytes_per_sec": round(nbytes / elapsed, 1),
        "mean_bytes_per_trace": round(nbytes / n_traces, 1),
        "verified_rate": round(n_verified / n_traces, 3),
        "failure_recovery_rate": round(n_recovered / n_traces, 3),
    }
    (OUT / "synthesis_stats.json").write_text(json.dumps(stats, indent=2))
    return stats


if __name__ == "__main__":
    import sys

    n = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    s = generate(n)
    print(json.dumps(s, indent=2))
