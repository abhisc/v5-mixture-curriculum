# Learning Guide: What Happened in This Repo

This is a beginner-friendly tour of the `v5-mixture-curriculum` repository.

The formal submission document is [`README.md`](README.md). That file is written like a
professional tech spec for reviewers who already know the field. **This guide is for you**
if terms like *token budget*, *data mixture*, *annealing*, *proxy experiment*, or *BPB*
are still fuzzy.

Read this first. Then open the README. Then poke the scripts.

---

## 1. The one-sentence version

We designed a **recipe for what data a large AI model should read during training**, and
we **tested a miniature version of that recipe** on a tiny model to see whether the recipe
actually works.

That recipe is called a **data mixture**. The staged order of reading is called a
**curriculum**. Together they are the Session 5 assignment.

---

## 2. Background you need before anything else

### What is a language model being trained on?

A model learns by predicting the next piece of text, over and over, on a huge pile of
documents. Those documents are turned into **tokens** (roughly: pieces of words).

When people say "we trained on 3.6 trillion tokens", they mean:

> the model saw 3.6 trillion of those pieces, in sequence, and tried to predict each next one.

Tokens are the unit of the training budget, the same way rupees are the unit of a money
budget.

### What is a "data mixture"?

Imagine you have a fixed food budget for a year, and you must decide:

- how much rice
- how much protein
- how much fruit
- how much spice

If you buy only rice, you survive but you are not strong. If you buy only spice, you are
interesting for a week and then sick.

A **data mixture** is the same idea for AI training:

| Food analogy | Training analogy |
|---|---|
| Rice | General web text (Wikipedia, articles) |
| Protein | Code |
| Vegetables | Math / science / reasoning |
| Specialty cuisine | Indian languages (Indic) |
| Cooking practice | Agentic / tool-use traces (using a terminal, calling tools) |

You cannot give every category 100%, because the total token budget is fixed. Every
percent you give to code is a percent you take from something else. That is why mixture
design is an argument, not a shopping list.

### What is a "curriculum"?

Even with the right mix of food, *order* matters. You do not feed a baby steak on day one.

In training:

1. Early: lots of general language and basic code
2. Middle: more specialist code and STEM
3. Later: longer reasoning traces and tool-use
4. End ("annealing"): a small amount of the *very best* data, while learning slows down

That progression is the **curriculum**.

### What is "wishful accounting"?

This is the trap the assignment specifically punishes.

Wishful accounting looks like:

> "We'll allocate 15% of training to agentic data."

…without checking whether 15% of agentic data even *exists*.

If the real world only has 1% as much agentic data as you need, your plan is fiction unless
you also explain how you will **create** the missing 99% (usually by synthesis).

This repo spends a lot of energy avoiding that trap.

---

## 3. What was the assignment asking for?

Session 5 asked for a **written plan** (a GitHub README), not a fancy web app.

The plan had to defend:

1. Exact % of the token budget for each capability lane
2. A detailed Indic split (verified / unverified / translated / synthetic)
3. Named datasets for agentic, reasoning, and long-context slots
4. Honest comparison of targets vs real supply
5. Protected floors so scarce data is never starved
6. An annealing reserve of best-tier data held until the end
7. Reasoning effort bands (low / medium / high / ultra) with examples
8. A testable proxy experiment at small scale
9. Targeted data cleaning for starved lanes

The grading philosophy: **tight reasoning beats vague padding**, and **running a small
experiment beats only describing one**.

---

## 4. What this repo actually contains

```
v5-mixture-curriculum/
├── README.md                 ← the formal specification (submit this)
├── LEARNING_GUIDE.md         ← this file
├── inventory/
│   ├── inventory.json        ← "how much real data exists?" (SOTA Dataset Inventory catalog)
│   └── mixture.json          ← "how will we spend the budget?"
├── scripts/
│   ├── budget.py             ← checks the math and supply
│   ├── synthesis_cost.py     ← costs generating missing data
│   └── build_tables.py       ← fills tables into README.md
├── proxy/
│   ├── fetch_data.py         ← downloads real tiny corpora
│   ├── agentic_synth.py      ← creates tool-use traces by running real commands
│   ├── clean.py              ← cleans / dedups / script-gates data
│   ├── data_pipeline.py      ← packs data into training shards
│   ├── model.py              ← tiny transformer
│   ├── train.py              ← trains one mixture
│   └── run_ablation.py       ← trains several mixtures and compares them
└── results/
    ├── ablation.json         ← measured experiment outcomes
    ├── cleaning_report.json  ← what cleaning removed / reclassified
    └── synthesis_stats.json  ← how fast agentic synthesis ran
```

There are two layers:

| Layer | Purpose |
|---|---|
| **Specification layer** (`inventory/`, `scripts/`, `README.md`) | The full-scale V5 plan for ~4 trillion tokens, sized from the SOTA Dataset Inventory |
| **Proxy layer** (`proxy/`, `results/`) | A tiny real experiment that tests whether the *idea* of protected floors works |

The proxy cannot prove the full plan. It can only **falsify** bad assumptions cheaply.

---

## 5. Core vocabulary (in plain language)

| Term | Plain meaning |
|---|---|
| **Token** | A chunk of text the model reads (often ~¾ of a word in English) |
| **Lane / capability lane** | A category of data (web, code, STEM, Indic, agentic, long-context) |
| **Budget share** | Percentage of total training tokens given to a lane |
| **Unique supply** | How many *distinct* tokens you actually own for that lane |
| **Epoch / repetition** | Reading the same unique data more than once |
| **Synthetic data** | Data generated by models/scripts because natural data is scarce |
| **Protected floor** | A minimum % a lane must always get, even if an automatic selector would drop it |
| **OPUS** | The imagined global data selector that picks "useful" samples during training |
| **Annealing / cooldown** | Final training phase with lower learning rate and best-quality data only |
| **Tier A** | The highest-quality held-out data saved for annealing |
| **Proxy experiment** | A cheap small-model run used to test a hypothesis before full-scale training |
| **Ablation** | Train almost-identical runs that differ in *one* thing, then compare |
| **BPB (bits per byte)** | A loss metric: lower = the model predicts that lane's text better |
| **Script fidelity** | Whether the model keeps writing in Devanagari/Tamil/etc. instead of switching to English letters |
| **Agentic** | Behaviour involving tools: shell commands, APIs, multi-step plans, recovery from errors |

---

## 6. The plan in one picture

```text
Total budget: 4.0 trillion tokens
│
├── 90%  Pre-training (3.6T)     ← learn capabilities
│         web 35% | code 32% | STEM 18% | Indic 8% | agentic 5% | longctx 2%
│
├──  2%  Annealing (80B)         ← consolidate with Tier A only
│
└──  8%  Post-training (320B)    ← SFT + RL (elicitation, not new capability)
```

### Why these numbers (intuition, not the full defence)

- **Code is huge (32%)** because coding is win-condition #1, and good code also helps
  reasoning.
- **Web is still large (35%)** so the model does not forget common knowledge.
- **Indic is only 8%** but fully protected, because English data would otherwise crowd it out.
- **Agentic is only 5%** on purpose. Natural agentic data barely exists. Claiming 15%
  without a generation plan would be wishful accounting.
- **Long context is 2%** because long context is mostly a *packing strategy* (glue
  existing docs into long sequences), not a new data mountain.

The full defence of every percentage is in README sections II–IV.

---

## 7. The hardest honesty problem: supply vs target

This is the heart of the repo.

For each lane we asked:

1. How many tokens do we *want*?
2. How many unique tokens do we *have*?
3. If we need more, do we **repeat**, **translate**, or **synthesise**?

Rough verdicts:

| Lane | Can we meet the target? |
|---|---|
| Web | Yes, easily, with unique tokens |
| Code | Yes, with mild repetition (~1.2 epochs) |
| STEM | Yes, with mild repetition |
| Indic | Partially with native text; translated + synthetic tiers fill the rest |
| Agentic | Almost entirely synthetic (~99%) |
| Long context | Mostly re-packed tokens borrowed from code/STEM |

`scripts/budget.py` is the calculator that checks this.

Try it:

```bash
make check
```

You should see each lane's allocation, unique supply, epochs, and a verdict string.

---

## 8. Why "protected floors" exist

During training, a selector like OPUS tries to pick data that reduces loss the most.

Problem: that selector is usually judged on English + math-ish proxies. So it tends to
think:

- Indic text is "annoying" (higher loss)
- Terminal / tool traces look "messy" and not very useful

Left alone, it can quietly starve the lanes you care about.

A **protected floor** says:

> No matter what OPUS wants, Indic must get at least 8%, agentic at least 4%, etc.

The proxy experiment was designed to measure what happens if you remove those floors.

---

## 9. What the proxy experiment actually did

### Scale

Not a giant production model. A **tiny** model:

- ~10.7 million parameters
- trained on ~14.7 million tokens per arm
- on your laptop GPU

This is like testing a bridge design with a foam model. Foam is not steel, but if the foam
model collapses under a predictable load, you learned something cheap.

### Real data used

| Lane | Source used in the proxy |
|---|---|
| Web | WikiText-2 |
| Code | Local Python files + MBPP tasks |
| STEM | GSM8K math word problems |
| Indic | Universal Dependencies treebanks (Hindi, Marathi, Tamil, Telugu, Urdu…) |
| Agentic | Traces generated by actually running shell commands |
| Long context | Packed concatenations of code + STEM docs |

### Arms compared

The important completed comparison:

| Arm | Meaning |
|---|---|
| `v5` | Proposed mixture with floors (Indic 8%, agentic 5%) |
| `opus_greedy` | What happens if scarce lanes are cut to ~0.5% and the rest goes to web/code |

### Headline measured result

Removing floors:

- gained about **0.05 bits** on general web prediction
- lost about **1.57 bits** on Indic
- lost about **1.88 bits** on agentic
- almost destroyed the model's ability to emit a tool-call structure (0.80 → 0.03)

Translation: the greedy trade looks slightly better on English metrics and is a disaster
on the scarce lanes. That is exactly why floors exist.

### Two honest caveats already written into the README

1. The general-lane tax is small (and under the 0.10 BPB falsification bar), but it is now
   larger than seed noise — still not something to over-precision.
2. Indic *script* stayed mostly fine at 0.5% starvation, while *quality* got much worse.
   So monitoring only for "did it stop writing Devanagari?" is not enough.

The Indic share sweep (0% / 2% / 4% / 8% / 16%) **did run**. Almost all gains arrive by 2%;
8%→16% buys almost nothing. So 8% sits on the flat part of the curve.

---

## 10. The surprising cleaning discovery

While cleaning Indic data, nearly **half** of the Indic documents failed a native-script
check.

Why? The Sanskrit treebank was written in **romanised Latin letters** (IAST), like:

```text
pūrvasya medhājananāni...
```

A language-ID tool can still say "this is Sanskrit." But it is **not** Devanagari native
script.

If you count romanised text as "verified native Indic", you overstate your native supply.
This repo reclassified those docs into a **romanised tier** instead of deleting them or
pretending they were native script.

That is a miniature version of a full-scale risk: bad taxonomy creates fake supply.

---

## 11. How the numbers stay honest

A common failure mode in long specs is:

> someone edits a percentage in prose, forgets a table, and the document contradicts itself.

This repo avoids that:

1. Human edits `inventory/mixture.json` and `inventory/inventory.json`
2. `scripts/budget.py` validates consistency
3. `scripts/build_tables.py` regenerates every table inside `README.md`
4. CI (`.github/workflows/check.yml`) fails if README tables drift

So README tables are outputs, not handwritten claims.

---

## 12. Suggested learning path (do this in order)

### Step A — Read for orientation (30–40 min)

1. This file (you are here)
2. README sections I and II only
3. Skim section VIII (proxy results)

Do not try to memorize every dataset name yet.

### Step B — Look at the config (20 min)

Open:

- `inventory/mixture.json` → the plan
- `inventory/inventory.json` → the supply

Ask yourself for one lane, e.g. `agentic`:

- What share did we allocate?
- What floor protects it?
- Which datasets are listed?
- Does the inventory make the 5% claim look honest or fictional?

### Step C — Run the calculators (10 min)

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python numpy torch
make check
```

Read the printed verdicts.

### Step D — Inspect one real agentic trace (10 min)

Open a bit of:

`proxy/data/raw/agentic/agentic_traces.txt`

You should see:

- a goal
- a thought
- a real bash command
- a real observation / exit code
- sometimes a failure, then a fix

That file is the concrete answer to "what does agentic data look like?"

### Step E — Read the cleaning report (10 min)

Open `results/cleaning_report.json`.

Notice Indic retention around 50% and the large romanised reclassification count.

### Step F — Read the ablation numbers (15 min)

Open `results/ablation.json`, or just re-read README section VIII.

Compare `v5` vs `opus_greedy` on:

- `indic` BPB
- `agentic` BPB
- `indic_script_fidelity`
- `agentic_structure_score`

### Step G — Optional deeper coding (as curiosity allows)

| If you want to understand… | Read |
|---|---|
| How the tiny model works | `proxy/model.py` |
| How mixture sampling works | `proxy/train.py` |
| How arms are defined | `proxy/run_ablation.py` |
| How synthesis is costed | `scripts/synthesis_cost.py` |

You do **not** need to understand every line of the transformer to understand the
assignment. The assignment is about mixture reasoning; the model is just the measuring
instrument.

---

## 13. Common confusions

### "Did we train the real V5 model?"

No. We wrote a plan for V5-scale training and tested a tiny proxy.

### "Are the proxy BPB numbers the same as SWE-bench / MMLU scores?"

No. BPB is a training-loss style metric. It is useful for relative comparisons between
mixtures. It does not equal a public benchmark score.

### "Why is agentic only 5% if it is so important?"

Because importance ≠ available supply. The plan protects agentic with a floor and
concentrates it late in the curriculum, while staying honest about synthesis cost.

### "Why generate agentic data by running real commands?"

If an LLM invents fake terminal output, the student model learns to hallucinate
filesystems. Grounded synthesis records real outputs so the lessons are true.

### "What does 'annealing reserve' mean in practice?"

It means: take some of your best data, **do not show it during normal pre-training**, and
only show it at the end when the learning rate is low. The idea is consolidation of rare,
high-value skills.

---

## 14. If you only remember five things

1. **Mixture = budget allocation across data types under a fixed token limit.**
2. **Curriculum = the order and staging of that mixture over training.**
3. **Wishful accounting = allocating tokens to a lane that has no real supply plan.**
4. **Protected floors exist because automatic selectors starve scarce but important data.**
5. **We tested the floor idea on a tiny model and measured a huge asymmetric tradeoff.**

---

## 15. Where to go next

| Goal | Open |
|---|---|
| Submit / review the formal plan | [`README.md`](README.md) |
| Change a percentage and regenerate tables | edit `inventory/mixture.json`, then `make tables` |
| Re-run proxy arms / regenerate result tables | `make ablation && make tables` |
| Re-check consistency | `make check` |

If something in the README still feels opaque after this guide, it is usually one of:

- a supply-accounting detail (read section III)
- a curriculum stage table (read section VI)
- a measured result caveat (read section VIII)

Those three sections are the densest, and they are where most of the intellectual work of
this repo lives.
