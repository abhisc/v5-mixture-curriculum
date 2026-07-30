# V5 Mixture and Curriculum Specification

A 4.0T-token data plan for a model that must win on **coding**, **agentic workflows**,
**controllable reasoning**, and **native Indic languages** — four capabilities that
compete for the same fixed budget.

Every number in this document is generated from [`inventory/inventory.json`](inventory/inventory.json)
and [`inventory/mixture.json`](inventory/mixture.json) by [`scripts/build_tables.py`](scripts/build_tables.py).
Nothing is hand-typed, and `make check` fails if the spec stops being internally consistent.

**The mixture was tested, not just asserted.** A 10.7M-parameter proxy was trained on real
corpora, one run per candidate mixture, to measure what the protected floors actually buy.
The headline result: removing the floors gains **0.041 bits** on the general lane and
loses **1.47 bits** on Indic and **1.90 bits** on agentic.
[Section VIII](#viii-proxy-experiment-hypothesis-and-measured-results) reports the numbers,
including the one that contradicted our expectations and changed the spec, and states
plainly which hypotheses remain untested.

---

## Contents

| # | Section |
|---|---|
| I | [Executive summary and win conditions](#i-executive-summary-and-win-conditions) |
| II | [Defended budget allocation](#ii-defended-budget-allocation) |
| III | [Inventory and honest supply sizing](#iii-inventory-and-honest-supply-sizing) |
| IV | [Native Indic breakdown](#iv-native-indic-breakdown) |
| V | [OPUS routing and protected floors](#v-opus-routing-and-protected-floors) |
| VI | [Curriculum and stable transitions](#vi-curriculum-and-stable-transitions) |
| VII | [Reasoning-effort control bands](#vii-reasoning-effort-control-bands) |
| VIII | [Proxy experiment: hypothesis and measured results](#viii-proxy-experiment-hypothesis-and-measured-results) |
| IX | [Targeted data cleaning](#ix-targeted-data-cleaning) |
| X | [Risk register and reproduction](#x-risk-register-and-reproduction) |

---

## I. Executive summary and win conditions

V5 is a **coding-and-agentic specialist that is also the strongest native-Indic model in
its weight class**. It is not a general-knowledge maximiser, and the budget reflects that.

**Budget split of 4.0T tokens**

| Phase | Tokens | Share | Purpose |
|---|---:|---:|---|
| Pre-training (S1–S4) | 3.60T | 90% | Capability formation |
| Anneal / cooldown (S5) | 80B | 2% | Tier A consolidation at low LR |
| Post-training (SFT + RL) | 320B | 8% | Elicitation, not capability creation |

**What we are trying to win, and what we are willing to lose.**

| Priority | Capability | Primary benchmarks | Stance |
|---|---|---|---|
| 1 | Coding | SWE-bench Verified, LiveCodeBench | Win outright |
| 2 | Agentic | Terminal-Bench, BFCL v3, τ-bench | Win outright |
| 3 | Controllable reasoning | AIME, MATH-500, GPQA-Diamond | Match frontier at matched effort |
| 4 | Native Indic | IndicMMLU-Pro, MILU, IndicGenBench | Win outright in class |
| — | General knowledge | MMLU, MMLU-Pro | **Accept 1–2 points below a web-maximal twin.** This is the bill for lanes 1–4 and we pay it deliberately. |

The single most important design claim: **three of our four target capabilities are
data-poor, and a loss-based data selector will quietly defund all three.** The plan is
therefore built around protected floors and a synthesis pipeline, not around clever
re-weighting of abundant data.

---

## II. Defended budget allocation

<!-- BEGIN:budget -->
| Lane | Share | Tokens | Benchmarks this lane must win | Supply verdict |
|---|---:|---:|---|---|
| General web | 35% | 1.26T | MMLU, MMLU-Pro, GPQA-Diamond, HellaSwag | MET with unique tokens |
| Specialist code | 32% | 1.15T | SWE-bench Verified, LiveCodeBench, HumanEval+, MultiPL-E | MET by repetition (1.24 epochs) |
| STEM / reasoning | 18% | 648.0B | AIME, MATH-500, GPQA-Diamond, ARC-AGI | MET by repetition (1.43 epochs) |
| Native Indic | 8% | 288.0B | IndicMMLU-Pro, MILU, IndicGenBench, Indic-XNLI, IN22 (chrF++) | SYNTHESIS-DEPENDENT (60% generated) |
| Agentic / tool-use | 5% | 180.0B | Terminal-Bench, BFCL v3, SWE-bench Verified (agent scaffold), tau-bench | SYNTHESIS-DEPENDENT (99% generated) |
| Long context | 2% | 72.0B | RULER 128k, LongBench v2, repo-level SWE-bench | RE-PACKED (66B borrowed, 5.8B native) |
| **Total pre-training** | **100%** | **3.60T** | | |
<!-- END:budget -->

### Why each number

**General web — 35% (1.26T).** This is a *floor on knowledge retention*, not an
aspiration. Web is our cheapest lane by far (4.96T unique available, so we run it at
0.25 epochs and can select the top quartile by classifier score). We cut it from a
conventional ~60–70% because the marginal MMLU point from web tokens 1.26T→2.0T is
small, while the same tokens moved into code and Indic buy capabilities that no amount
of web data produces. We do not cut it below 35%: the proxy run shows a measurable
general-lane cost even at these ratios, and MMLU regressions are hard to recover in
post-training because they are knowledge, not skill.

**Specialist code — 32% (1.15T).** The largest specialist lane, because SWE-bench
Verified is win condition #1 and because code transfers: it is the densest available
source of long-range dependency, state tracking, and formal structure, which is why it
lifts reasoning benchmarks too. Why not 45%? Two hard reasons. (1) **Supply.** We own
958B unique permissive-licensed tokens; 32% already puts us at 1.24 epochs once the
long-context lane's repo-packing draw is charged. 45% would mean 1.7 epochs and force us
into non-permissive or low-quality code, which raises legal risk for zero benchmark gain.
(2) **Diminishing returns.** Code-maximal models (60–90% code) buy HumanEval and lose
everything else; the models that actually win *agentic* SWE-bench are generalists with a
heavy code lane, because resolving a real issue requires reading English issue text and
planning, not just emitting syntax.

**STEM and reasoning — 18% (648B).** Sized by the *long-CoT bottleneck*, not by ambition.
Raw math and science text is plentiful (396B unique), but the reasoning-trace pool that
actually teaches AIME-style behaviour is only ~37B tokens and heavily cross-distilled.
Pushing this lane above 18% would mean repeating the same distilled traces, which the
data-constrained scaling law says is worth progressively less, and which in practice
teaches trace *style* rather than reasoning. The right place to spend on reasoning is the
anneal (27% of Tier A) and RL, not more pre-training epochs.

**Native Indic — 8% (288B).** The most-contested number, so it gets the most defence.
Against ~22 scheduled languages, 288B is ~13B tokens per language on a
population-and-supply weighted split. That is roughly the scale at which a language stops
being a transliteration veneer over English and starts carrying its own morphology.
Section VIII measures what starvation costs: cutting Indic to 0.5% degrades Indic BPB by
**1.47 bits** while improving the general lane by **0.041 bits**, an amount indistinguishable
from seed noise. We have *not* yet measured whether 8% is optimal as opposed to merely
safe — the sweep that would establish that is specified and unrun, and Section VIII says so.

**Agentic and tool-use — 5% (180B).** Deliberately *not* larger, and this is the
"wishful accounting" trap the whole plan is built to avoid. Natural agentic supply is
**1.2B unique tokens** against a 180B target. Every token above that must be generated.
5% is what we can credibly synthesise and verify (costed in Section III at 1,638
GPU-days); 10% would be a number with nothing behind it. The curriculum compensates by
concentrating the lane where it matters: agentic runs at 10.5% during S4 and 22.5% of the
anneal, so peak exposure is 4.5× the headline share.

**Long context — 2% (72B).** Small because long context is **a packing strategy, not a
token source**. 92% of this lane is whole-repository and whole-paper packing of tokens
that already belong to the code and STEM lanes; only 5.8B is natively-long text (PG-19).
Those borrowed tokens are debited from the lending lanes so nothing is double-counted.
Long context is expensive in FLOPs (attention at 64k), not in tokens, so the budget line
understates the compute commitment.

---

## III. Inventory and honest supply sizing

The audit below charges every lane against real, catalogued supply. Two mechanisms make
it honest:

- **Overlap discounts.** FineWeb-Edu and DCLM are both CommonCrawl derivatives; counting
  both at face value would invent ~700B tokens that do not exist. Each dataset carries an
  explicit discount, set high where we are unsure so that we under-claim.
- **Repetition is priced, not ignored.** Repeated tokens are converted to fresh-token
  equivalents using the data-constrained scaling law
  (Muennighoff et al. 2023), `D' = U·(1 + R*·(1 − e^{−R/R*}))` with `R* = 15.4`. A lane
  cannot hide over-allocation by simply looping.

<!-- BEGIN:supply -->
| Lane | Allocated | Unique supply owned | Drawn from natural | Must be generated | Epochs on natural | Fresh-token equivalent |
|---|---:|---:|---:|---:|---:|---:|
| General web | 1.26T | 4.96T | 1.26T | 0.0B | 0.25 | 1.26T |
| Specialist code | 1.15T | 958.4B | 1.15T | 0.0B | 1.24 | 1.15T |
| STEM / reasoning | 648.0B | 396.2B | 550.8B | 99.9B | 1.43 | 645.7B |
| Native Indic | 288.0B | 241.1B | 115.2B | 172.8B | 0.48 | 288.0B |
| Agentic / tool-use | 180.0B | 1.2B | 1.8B | 186.8B | 1.61 | 180.0B |
| Long context | 72.0B | 1.7B | 5.8B | 0.0B | 3.37 | 71.6B |
| **Total** | **3.60T** | | | | | **3.60T** (99.9%) |
<!-- END:supply -->

### Where the targets are met, plainly

**Met with unique tokens, no repetition:** general web (0.25 epochs — we are selecting the
best quarter of what we have, not scraping the barrel) and both natural Indic tiers
(0.48 epochs).

**Met by repetition, within the 4-epoch soft cap:** code at **1.24 epochs** and STEM at
**1.43 epochs**. Neither is a problem — up to ~4 epochs, repeated tokens are worth close
to fresh ones. The code repetition is targeted, not uniform: we re-draw the Stack-Edu top
decile and dependency-complete repositories, and we do not repeat the long tail.

**Cannot be met without synthesis:** the agentic lane, which is short by **185B tokens**,
and the Indic translated and synthetic tiers, short by **129B**. This is the plan's
largest exposure, so it is costed rather than asserted.

### The agentic lane is 99% synthetic. Here is the evidence it is feasible.

A synthesis claim is only as good as the generator behind it, so we built one and measured
it. [`proxy/agentic_synth.py`](proxy/agentic_synth.py) produces **execution-grounded**
traces: it runs real commands in a scratch directory and records real stdout, real stderr
and real exit codes. Nothing is imagined. That distinction is the whole ballgame — a model
trained on LLM-invented `ls` output learns to hallucinate filesystems, which is precisely
the failure Terminal-Bench punishes. Every trace also contains at least one genuine
failure followed by a genuine recovery, because clean happy-path traces never teach error
handling.

Measured, on one laptop core:

<!-- BEGIN:synthesis -->
| Measurement | Value |
|---|---:|
| Traces generated | 2,600 |
| Tool-call steps executed | 8,914 |
| Wall time | 228s on one laptop core |
| Throughput | 11.4 traces/s, 10.8 kB/s |
| Mean trace size | 945 bytes |
| Assertion-verified traces | 100.0% |
| Traces containing a real failure then recovery | 71.5% |
<!-- END:synthesis -->

Scaling that to the 186.8B target, with explicit yield assumptions
([`scripts/synthesis_cost.py`](scripts/synthesis_cost.py)):

<!-- BEGIN:synthcost -->
| Synthetic slot | Tokens needed | Yield after verify + dedup | Tokens actually generated | GPU-days |
|---|---:|---:|---:|---:|
| Agentic rollouts | 186.8B | 44% | 424.5B | 1,638 |
| Indic translated tier | 115.2B | 75% | 153.6B | 89 |
| Indic synthetic dialogue | 15.7B | 70% | 22.4B | 87 |
| **Total** | | | | **1,813** (0.9 days on 2048 GPUs) |
<!-- END:synthcost -->

So the agentic lane costs roughly **1,638 GPU-days of teacher inference plus ~1,800
CPU-core-days of sandboxed execution** — about a day of wall-clock on a 2,048-GPU
allocation with a few thousand parallel containers. That is a real, payable bill. It is
also why the lane is 5% and not 15%: at 15% the generation cost alone would rival a
meaningful fraction of the training run.

The honest caveat: our measured 100% verification rate comes from templated task families
with known-good solutions. Open-ended generation will be far lower, which is why the cost
model assumes a **55% accept rate**, not 100%.

---

## IV. Native Indic breakdown

The 288B Indic slot is not one thing. Tiers differ by an order of magnitude in cost and
in trustworthiness, and conflating them is how Indic plans quietly become
English-translated plans.

<!-- BEGIN:indic -->
| Tier | Share of Indic slot | Tokens | Natural supply available | Gap to close | How the gap is closed |
|---|---:|---:|---:|---:|---|
| Verified | 10% | 28.8B | 74.6B | none | None needed. We select the best 28.8B of 74.6B, so we can afford to be picky. |
| Unverified | 30% | 86.4B | 122.3B | none | None needed. 35.9B of headroom is held as contingency if the MT gate rejects more than planned. |
| Translated | 40% | 115.2B | 2.2B | 113.0B | IndicTrans2-class MT over STEM and instruction tokens, round-trip chrF++ gated. |
| Synthetic | 20% | 57.6B | 41.9B | 15.7B | Romanised Sangraha tier plus generated native-script multi-turn dialogue. |
| **Total** | **100%** | **288.0B** | | | |
<!-- END:indic -->

**Verified (10%, 28.8B).** Human- or heuristically-audited native text: Sangraha's
verified tier, editorially-clean news, government gazettes, public-domain books. We have
74.6B available and only spend 28.8B, so this tier is *selection-limited, not
supply-limited* — we take the best 39% and can afford strict standards. This is also the
tier the anneal draws from.

**Unverified (30%, 86.4B).** LID- and perplexity-filtered crawl. 122.3B available against
an 86.4B draw, leaving **35.9B of contingency** — this headroom is the plan's insurance
policy, and Section X explains when we spend it.

**Translated (40%, 115.2B) — the number that needs the most defence.** Only 2.2B of
genuine parallel corpora exists, so ~113B must be machine-translated. Why spend the
largest tier on generated data when 35.9B of *real* native text is sitting unused?

Because they buy different things. The benchmarks we must win — IndicMMLU-Pro, MILU — are
*academic and technical* evaluations in Indian languages. Native Indic web crawl is
overwhelmingly news, entertainment and social content; it contains almost no university
physics, no competitive-programming discussion, no formal mathematics. No quantity of
additional crawl produces a model that can do STEM reasoning in Telugu. Translating our
own high-quality STEM and instruction tokens is the only mechanism that transfers *task
capability* across the language boundary. The unverified tier teaches the model to
*speak* the language; the translated tier teaches it to *think* in it.

Quality gating is mandatory and non-negotiable: round-trip chrF++ thresholds per language
pair, plus a native-speaker-audited sample per batch. Translationese is a real risk, which
is why translated tokens are barred from the anneal reserve — Tier A Indic is verified
native only.

**Synthetic (20%, 57.6B).** Sangraha's romanised tier (41.9B) plus ~15.7B of generated
native-script multi-turn dialogue seeded from verified documents. Romanised text is not a
compromise: a large share of real Indian user input is Latin-script Hindi or code-mixed,
so this tier is target-distribution data, not a substitute for it.

### Per-language split of the 288B

Weighted by speaker population × benchmark coverage × available supply, with a hard floor
so no scheduled language falls below the level at which script competence fails to form.

| Band | Languages | Share each | Tokens each |
|---|---|---:|---:|
| Anchor | Hindi | 22% | 63.4B |
| Major | Bengali | 11% | 31.7B |
| Major | Tamil, Telugu | 9% | 25.9B |
| Major | Marathi | 8% | 23.0B |
| Mid | Gujarati | 7% | 20.2B |
| Mid | Kannada, Malayalam | 6% | 17.3B |
| Mid | Urdu | 5% | 14.4B |
| Long tail | Odia, Punjabi | 4% | 11.5B |
| Long tail | Assamese | 3% | 8.6B |
| Floor | 10 remaining scheduled languages | 0.6% | 1.7B |

The 1.7B floor is set for *competence*, not for script survival, and Section VIII explains
why that distinction is load-bearing: a starved lane keeps its script long after it has
lost its quality, so a floor tuned to prevent visible code-switching would be set far too
low. 1.7B is ~0.6% of the Indic slot, the same order as the `opus_greedy` allocation that
measurably hollowed the lane out, and it should be treated as a lower bound pending the
per-language sweep.

---

## V. OPUS routing and protected floors

OPUS selects data online by expected loss reduction against an evaluation proxy. That
proxy is English- and mathematics-heavy, which creates a systematic and predictable bias:
**it undervalues exactly the lanes we are trying to protect.** Agentic traces look like
noisy low-information terminal spew. Indic tokens raise average loss. Long packs are
throughput-expensive. A selector optimising a global objective will defund all three, and
it will look like it is doing a good job while it happens.

Floors are therefore implemented as a **hard two-stage sampler**, not as a soft prior:

1. With probability **43.5%** (the sum of all active floors), the lane is drawn from the
   floor distribution. OPUS does not see this decision and cannot override it.
2. With the remaining **56.5%**, OPUS allocates freely across all lanes by its own score.

A floor is a minimum, not a cap: OPUS is free to push a protected lane *above* its floor
out of discretionary mass, and during S4 it typically does for code.

<!-- BEGIN:floors -->
| Lane | Budget share | Protected floor | Floor active from | What the floor prevents |
|---|---:|---:|---|---|
| General web | 35.0% | none | S1 | No floor. OPUS over-selects web by default; it needs a ceiling, not a floor. |
| Specialist code | 32.0% | 20.0% | S1 | Stops a mid-run reasoning spike from cannibalising SWE-bench capability. |
| STEM / reasoning | 18.0% | 10.0% | S1 | Long-CoT tokens look high-loss and get down-weighted by loss-based selectors. |
| Native Indic | 8.0% | 8.0% | S1 | The binding constraint. Measured: cutting Indic to 0.5% costs 1.47 bits of Indic BPB to buy 0.04 bits of web BPB. |
| Agentic / tool-use | 5.0% | 4.0% | S1 | Measured: at 0.5% the model can no longer emit a well-formed tool call (structure score 0.03 vs 0.80). Terminal output looks like noise to a loss-based selector. |
| Long context | 2.0% | 1.5% | S3 | Prevents long packs being dropped for throughput reasons once sequence length rises. |
<!-- END:floors -->

Three deliberate choices here. **Web gets no floor** — it is what OPUS over-selects by
default; if anything it needs a ceiling. **The Indic floor equals its budget share (8%)**,
making it fully protected rather than partially, because the measured cost of starvation is
so lopsided (1.47 bits lost against 0.04 gained) that there is no version of this trade
worth letting a selector make. **The agentic floor (4%) sits below its share (5%)** so OPUS
retains some discretion to up-weight agentic data during S4 when it becomes genuinely
high-value.

---

## VI. Curriculum and stable transitions

Four pre-training stages plus an anneal. The stage mixes are not decorative: they
integrate, token-weighted, back to the Section II budget within 0.1pp, and `make check`
enforces that.

<!-- BEGIN:curriculum -->
| Stage | Tokens | Seq len | General web | Specialist code | STEM / reasoning | Native Indic | Agentic / tool-use | Long context |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| S1 Foundation | 1.08T | 4k | 57.0% | 21.0% | 10.0% | 8.0% | 4.0% | 0.0% |
| S2 Specialist | 1.26T | 4k | 32.0% | 40.0% | 16.0% | 8.0% | 4.0% | 0.0% |
| S3 Reasoning | 828.0B | 16k | 21.0% | 34.0% | 28.0% | 8.0% | 5.0% | 4.0% |
| S4 Agentic + Long Context | 432.0B | 64k | 15.0% | 33.0% | 24.5% | 8.0% | 10.5% | 9.0% |
| S5 Anneal (Tier A only) | 80.0B | 128k | 5.0% | 17.5% | 27.0% | 20.0% | 22.5% | 8.0% |
| **Realized S1-S4** | **3.60T** | | **34.93%** | **32.08%** | **17.98%** | **8.00%** | **5.01%** | **2.00%** |
| *Budget target* | | | *35.00%* | *32.00%* | *18.00%* | *8.00%* | *5.00%* | *2.00%* |
<!-- END:curriculum -->

**S1 Foundation (1.08T, 4k).** Web-heavy for language and world model. Note that Indic
sits at its full 8% and agentic at 4% *from the very first token*. Scarce capabilities are
never introduced late: a model that meets Devanagari for the first time at 60% of training
has already allocated its representational capacity elsewhere, and the tokenizer and
embedding table have already specialised.

**S2 Specialist (1.26T, 4k).** Code peaks at 40%. Sequence length stays at 4k because
throughput matters most here and almost nothing in this stage needs long range.

**S3 Reasoning (828B, 16k).** Long CoT and the reasoning-effort control tokens enter.
Context steps 4k → 16k.

**S4 Agentic and long context (432B, 64k).** Agentic triples to 10.5%, long context to 9%.
This is where Terminal-Bench and RULER are actually won. It is the most expensive stage
per token because of 64k attention, which is why it is the shortest.

**S5 Anneal (80B, 128k).** Tier A only, LR → ~0. Detailed below.

### Transition stability

V4 saw gradient-norm spikes of ~150× at stage boundaries. Three mitigations, applied at
every boundary:

1. **Mixture blending over a 10B-token band.** The old and new mixtures are linearly
   interpolated rather than switched. A discontinuous change in data distribution is a
   discontinuous change in the loss surface.
2. **LR re-warmup of 10% over 2B tokens** at each boundary.
3. **Sequence-length changes are the real culprit, and are handled separately.** The 4k →
   16k → 64k steps rescale the RoPE base *before* the boundary and hold the mixture fixed
   across the change, so length and distribution never move at the same time. Two variables
   changing simultaneously is how a spike becomes unattributable.

The proxy runs log per-step gradient norms (`max_grad_norm` in
[`results/ablation.json`](results/ablation.json)) so this claim stays measurable at small
scale.

### Annealing reserve: the Tier A holdout

80B of our best data is **withheld from S1–S4 by a hash-based holdout on the sampler, not
by a down-weight**. This matters: a soft down-weight is a suggestion that OPUS can
overrule, and a reserve OPUS can reach is not a reserve. Held-out shards are physically
absent from the pre-training index.

<!-- BEGIN:anneal -->
| Tier A component | Lane | Tokens | Share of anneal | Why it is held back |
|---|---|---:|---:|---|
| finemath_4plus | STEM / reasoning | 9.6B | 12.0% | Highest-density math on the open web. Held back so AIME/MATH gains land when LR is low and the model retains them. |
| verified_long_cot | STEM / reasoning | 12.0B | 15.0% | Execution- or answer-verified long CoT only. Unverified CoT goes in pretrain; verified CoT is the anneal payload. |
| execution_verified_agentic | Agentic / tool-use | 18.0B | 22.5% | Rollouts whose final state passes a real test/assertion. Scarcest, most expensive tokens we own. |
| stack_edu_top_decile | Specialist code | 14.0B | 17.5% | Top-decile educational code + docstring-complete repos + CommitPackFT diffs shaped like SWE-bench edits. |
| indic_verified_tier_a | Native Indic | 16.0B | 20.0% | 20% of the whole anneal for 8% of pretrain. Deliberate over-weight: Indic is the capability most likely to be under-consolidated at the end of a 3.6T English-dominated run. |
| longctx_packed_tier_a | Long context | 6.4B | 8.0% | Full-repo and full-paper packs at 128k. Long-context is a positional-extrapolation skill and consolidates best at low LR. |
| web_tier_a_reference | General web | 4.0B | 5.0% | Small anti-forgetting ballast so the anneal does not tank MMLU. |
| **Total** | | **80.0B** | **100%** | |
<!-- END:anneal -->

The allocation is deliberately *not* proportional to pre-training shares. Agentic takes
22.5% of the anneal against a 5% pre-training share, and Indic takes 20% against 8%. The
reasoning: at low LR the model consolidates rather than restructures, and the capabilities
most at risk of being under-consolidated after 3.6T English-dominated tokens are precisely
the scarce ones. Web gets 5% purely as anti-forgetting ballast.

---

## VII. Reasoning-effort control bands

Effort is a **control token** (`<effort=low|medium|high|ultra>`) prepended to
reasoning-bearing samples during S3 onward, so the user can trade latency for accuracy at
inference. Shares are of reasoning-tagged tokens, not of the whole budget.

<!-- BEGIN:bands -->
| Band | Thinking-token budget | Share of reasoning tokens |
|---|---|---:|
| `low` | 0-64 | 30% |
| `medium` | 64-512 | 40% |
| `high` | 512-4096 | 22% |
| `ultra` | 4096-32768 | 8% |
<!-- END:bands -->

The distribution is deliberately front-loaded toward short traces. The dominant failure
mode of effort-controlled models is **effort collapse**: trained mostly on long traces,
the model overthinks trivial inputs and ignores the tag. 70% of reasoning tokens sit in
`low` and `medium` to keep the short end genuinely short.

### One concrete example per band

**`low` — recall, no deliberation.** Budget 0–64 thinking tokens.

> **Prompt:** What is the capital of India?
> **Thinking:** *(none)*
> **Answer:** New Delhi.

A `low`-band trace that produces *any* reasoning is a training error. This band exists to
teach the model to stop.

**`medium` — a few bounded steps.** Budget 64–512.

> **Prompt:** A shop sells pens at ₹17 each. How many can be bought with ₹500, and what is left over?
> **Thinking:** 17 × 20 = 340. 17 × 29 = 493. 500 − 493 = 7, which is less than 17, so 29 is the maximum. Check: 17 × 30 = 510 > 500. Confirmed.
> **Answer:** 29 pens, ₹7 left over.

Arithmetic verified by a second method. Note the Indic-context framing: reasoning traces
are generated in Indic languages too, not only translated after the fact.

**`high` — structured multi-step with case analysis.** Budget 512–4096.

> **Prompt:** How many integers from 1 to 1000 inclusive are divisible by 3 or 5 but not by 15?
> **Thinking:** Inclusion–exclusion. |div by 3| = ⌊1000/3⌋ = 333. |div by 5| = ⌊1000/5⌋ = 200. |div by 15| = ⌊1000/15⌋ = 66. Divisible by 3 or 5 = 333 + 200 − 66 = 467. The question excludes multiples of 15, which are counted once in that 467, so subtract 66: 467 − 66 = 401. Sanity check: 333 − 66 = 267 (only 3), 200 − 66 = 134 (only 5), 267 + 134 = 401. Consistent.
> **Answer:** 401.

The band is defined by *structure* — a named method, a computation, an independent check —
not merely by length. Note that the naive answer (467) is wrong for the question asked;
`high`-band data is where we teach re-reading the constraint.

**`ultra` — hypothesis, real evidence, self-correction.** Budget 4096–32768. Almost always
agentic and interleaved with tool calls, because sustained reasoning without external
evidence mostly produces confident drift.

> **Prompt:** Our service fails to bind port 8080 on deploy, but only on the staging cluster. Fix it.
> **Thinking:** Three candidate causes: a stale process holding the port, a port conflict with a co-scheduled pod, or a permissions/capability difference between clusters. Cheapest discriminator first — is anything actually holding it?
> `lsof -i :8080` → *no output.* So nothing is holding it, which falsifies hypothesis one and, more importantly, means "port in use" is the wrong frame. Re-read the actual error rather than the assumed one.
> `kubectl logs deploy/api -n staging | tail -5` → `bind: permission denied`, not `address already in use`. Different failure entirely; my first two hypotheses were answering a question nobody asked.
> Permission denied on a high port is unusual — 8080 is unprivileged. Unless the container's `net.ipv4.ip_unprivileged_port_start` differs, or a securityContext drops `NET_BIND_SERVICE`.
> `kubectl get deploy/api -n staging -o jsonpath='{...securityContext}'` → staging sets `runAsNonRoot: true` with a hardened sysctl profile that production does not.
> Root cause found: the hardened profile raises the unprivileged-port floor above 8080. Two fixes — relax the sysctl, or bind a higher port and remap at the service. The second does not weaken the hardening, so prefer it.
> Applied service remap 8080 → 18080, redeployed, confirmed `Listening on :18080` and the service endpoint responds.
> **Answer:** Not a port conflict. Staging's hardened securityContext raises the unprivileged-port floor; the app now binds 18080 with the Service mapping 8080 → 18080. Hardening is unchanged.

The pedagogically load-bearing part is the model discovering its own framing was wrong and
saying so. Traces that proceed smoothly from correct hypothesis to correct fix teach
confidence, not debugging. This is why the anneal reserves 18B tokens of
**execution-verified** agentic data: the observations above must be real, or the model
learns to fabricate plausible `lsof` output.

---

## VIII. Proxy experiment: hypothesis and measured results

### The mixture as a falsifiable hypothesis

> **H1.** Protected floors at Indic 8% and agentic 4% preserve those capabilities at a
> general-lane cost of **< 0.10 bits per byte**.
>
> **H2.** Without floors, a loss-optimising selector starves both lanes, and the damage is
> **large and asymmetric** — it gives up far more capability than it gains.
>
> **H3.** Indic returns have a **knee at or below 8%**: 0% → 4% buys a large improvement,
> 8% → 16% buys little, so 8% is near-optimal rather than merely safe.

Each is falsifiable. H1 fails if the general-lane cost exceeds 0.10 BPB. H2 fails if the
starved arm loses little. H3 fails if 16% still improves Indic substantially at acceptable
cost — which would mean our Indic allocation is *too low*.

### The experiment we ran

| | |
|---|---|
| Model | 10.7M non-embedding params, 6 layers, d=384, byte-level, 384 ctx |
| Tokens per arm | 14.7M (1,200 steps × 32 × 384) |
| Arms completed | **2 of 11 planned** — `v5` (2 seeds) and `opus_greedy` (1 seed). The sweep arms were not run; see "status" below. |
| Corpora | **Real.** WikiText-2, local Python source, GSM8K, UD treebanks (Hindi/Marathi/Tamil/Telugu/Urdu), and 2,600 execution-grounded agentic traces |
| Controls | Identical seed, architecture, optimizer, step count and token count across arms. Only sampling proportions differ. |

**Status, stated up front so nothing below is oversold.** The full design is 11 arms
(13 runs). Three runs completed before the sweep was stopped for time. The two arms that
did complete are the headline comparison, so **H2 is tested and H1 is partially tested;
H3 is not tested.** `results/ablation.json` is resumable — `make ablation` skips completed
arms and runs the remaining ten in ~95 minutes.

Byte-level tokenization is a deliberate control, not a shortcut: a BPE vocabulary trained
on an English-dominated mixture makes Devanagari look artificially expensive, which would
contaminate the exact measurement this experiment exists to make. Bits-per-byte is
comparable across scripts.

The `opus_greedy` arm models what an unconstrained selector does when its scoring proxy is
English- and maths-heavy: it cuts Indic and agentic to 0.5% each and spends the tokens on
web and code instead. Sweep arms (not run) change **one lane at a time**, always taking the
difference from the web lane, so any movement is attributable.

**Metrics.** `BPB` is held-out bits per byte per lane. Two capability metrics go beyond
loss:

- **Indic script fidelity** — feed real Devanagari prompts, generate 192 bytes, measure the
  fraction of generated letters still in an Indic script. Intended to catch a model
  code-switching back to Latin under starvation. The proxy-scale analogue of script
  integrity on IndicGenBench. (In the event this metric proved *less* sensitive than
  expected — see the findings below.)
- **Agentic structure score** — given an agentic goal, does the continuation reproduce the
  think/action/observation protocol? The proxy-scale analogue of BFCL schema adherence.

### Results

<!-- BEGIN:ablation -->
| Arm | Indic % | Agentic % | web BPB | code BPB | stem BPB | indic BPB | agentic BPB | Indic script fidelity | Agentic structure |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `v5` | 8.0 | 5.0 | 2.406 | 2.411 | 2.093 | 2.329 | 0.757 | 0.988 | 0.797 |
| `opus_greedy` | 0.5 | 0.5 | 2.365 | 2.331 | 2.048 | 3.803 | 2.653 | 0.958 | 0.031 |
<!-- END:ablation -->

<!-- BEGIN:findings -->
| Question | Comparison | Measured |
|---|---|---|
| **H2.** What does an unprotected selector cost? | `opus_greedy` vs `v5`, Indic BPB | **+1.474 bits** |
| | `opus_greedy` vs `v5`, agentic BPB | **+1.896 bits** |
| | `opus_greedy` vs `v5`, web BPB (what it gains) | **-0.041 bits** |
| | `opus_greedy` vs `v5`, script fidelity | **0.958 vs 0.988** |
| **Is the general-lane cost even measurable?** | seed-to-seed spread of `v5` web BPB over 2 seeds | **0.037 bits** - the same size as the effect above |
| Seed noise ceiling | worst-lane BPB spread across 2 seeds of `v5` | **0.108 bits** (agentic) |
<!-- END:findings -->

**H2 is confirmed, and the asymmetry is extreme.** Removing the floors buys the greedy arm
**0.041 bits** on the general lane. It costs **1.474 bits** on Indic and **1.896 bits** on
agentic — a 36× and 46× worse trade than the gain. The starved model also loses the
agentic protocol almost entirely: its structure score falls from **0.797 to 0.031**,
meaning it can no longer emit a well-formed tool call, having seen tool traces in 0.5% of
its batches instead of 5%. This is the concrete justification for implementing floors as a
hard sampler stage that OPUS cannot override. A selector optimising a global loss proxy
would make exactly this trade, and on the metrics it was optimising it would look like an
improvement.

**H1 is supported, with an important caveat.** The predicted general-lane cost was
< 0.10 bits; measured, it is 0.041. But the seed-to-seed spread of `v5`'s own web BPB
across two seeds is **0.037 bits** — essentially the same magnitude. So the honest claim is
not "the floors cost 0.041 bits" but **"the floors cost an amount we cannot cleanly
distinguish from seed noise at this scale."** That is a better outcome for the plan than a
precise small number, but it is a weaker *measurement*, and separating the two needs more
seeds rather than more arms.

**One result modifies the specification.** We expected Indic starvation to show up first as
script collapse. It does not. At 0.5% Indic the model keeps script fidelity at **0.958** —
nearly intact — while its Indic BPB is 1.47 bits worse. Script identity survives a token
allocation; *competence* does not. Two consequences: the Indic floor must be sized for
competence rather than for script survival, and any training-time monitor that watches only
for code-switching will report green while the lane is being hollowed out. Section IV's
per-language floor is written on this basis.

**H3 is untested.** The Indic sweep (0/2/4/16%) did not run, so the claim that returns have
a knee at or below 8% is currently an argument from supply and per-language budgeting, not
a measurement. Until the sweep completes, **8% should be read as "defensible and safe", not
as "shown to be optimal"**. This is the largest evidential gap in the document, and it is
one command away from being closed.

### What this experiment does *not* establish

Stated plainly, because the gap matters.

- **10.7M parameters is two rungs below the specified 1B proxy and ~5 orders below V5.**
  Mixture effects are known to shift with scale; small models are disproportionately hurt
  by multilinguality because capacity is the binding constraint. The Indic tax measured
  here is likely an **over**-estimate, which makes the floor argument conservative.
- **BPB is not a benchmark.** It correlates with capability within a lane but cannot tell
  us about SWE-bench pass@1. The claims in Section II about specific benchmark wins are
  supported by the literature, not by this run.
- **The corpora are analogues.** UD treebanks are not Sangraha; 24MB of local Python is not
  The Stack v2. Relative comparisons between arms are meaningful; absolute values are not.
- **Two seeds on `v5`, one on `opus_greedy`.** Enough to show the Indic and agentic effects
  dwarf seed noise by more than an order of magnitude; **not** enough to resolve the
  general-lane cost, which is the same size as the spread.
- **Nine of eleven arms did not run.** The sweeps that would locate the knee, and the
  specialist ceilings that would quantify crowding-out, are specified and implemented but
  unexecuted.

### The 1B proxy we would run next

| | |
|---|---|
| Model | 1.4B params, 24 layers, d=2048, 100k BPE vocab with Indic-aware fertility budget |
| Tokens | 30B per arm (~20 tokens/param, Chinchilla-ish) |
| Arms | 6: `v5`, `opus_greedy`, `indic_04`, `indic_16`, `agentic_00`, `agentic_10` |
| Cost | ~6 × 250 A100-hours ≈ 1,500 GPU-hours |
| Stages | Compressed S1–S4 at 1/120 scale plus a 600M-token anneal, so the curriculum itself is tested, not only the flat mixture |

**Go / no-go metrics, decided before the run:**

| Metric | Threshold to proceed |
|---|---|
| MMLU (`v5` vs `opus_greedy`) | Gap ≤ 1.5 points |
| MILU / IndicMMLU-Pro | `v5` ≥ `opus_greedy` + 8 points |
| Indic script fidelity (generation) | ≥ 0.95 for `v5` |
| BFCL v3 AST accuracy | `v5` ≥ 2× `agentic_00` |
| HumanEval+ | `v5` within 1 point of a code-maximal control |
| Max gradient norm at stage boundaries | < 5× the running median |
| Indic knee | `indic_16` improves Indic ≤ 25% as much as `indic_04→08` did |

Failing the Indic knee test means 8% is wrong and should rise to 12%. Failing the MMLU gap
means the web lane must go back up to 40% and the code lane down to 28%. Both outcomes are
cheaper to discover at 1B than at 3.6T.

**Before that, finish the cheap rung.** The remaining ten proxy runs cost ~95 minutes on a
laptop and would close the H3 gap and quantify crowding-out:

```bash
make ablation && make tables   # resumes; completed arms are skipped
```

---

## IX. Targeted data cleaning

Cleaning effort is pointed at the lanes the audit shows to be starved, because that is
where a rejected document is expensive. Discarding a duplicate web page costs nothing when
the web lane runs at 0.25 epochs. Discarding an Indic document directly raises the Indic
lane's epoch count.

The pipeline in [`proxy/clean.py`](proxy/clean.py) implements length gating, exact dedup,
5-gram MinHash near-dedup with banded LSH at Jaccard ≥ 0.8, an Indic script gate, path/PII
scrubbing, and repetition detection. Run over the real corpora:

<!-- BEGIN:cleaning -->
| Lane | Docs in | Docs kept | Retention | Short | Exact dup | Near dup | Romanised (reclassified) | Repetitive | Path-scrubbed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| General web | 629 | 610 | 97.0% | 19 | 0 | 0 | 0 | 0 | 0 |
| Specialist code | 1520 | 1516 | 99.7% | 0 | 0 | 1 | 0 | 3 | 0 |
| STEM / reasoning | 8798 | 8796 | 100.0% | 2 | 0 | 0 | 0 | 0 | 0 |
| Native Indic | 5416 | 2731 | 50.4% | 0 | 0 | 0 | 2685 | 0 | 0 |
| Agentic / tool-use | 2600 | 2597 | 99.9% | 0 | 0 | 3 | 0 | 0 | 743 |
<!-- END:cleaning -->

**The finding that changed the spec.** The Indic script gate reclassified **2,685 of 5,416
documents (49.6%)** out of the native-script tier. All of them came from one source — the
UD Sanskrit-Vedic treebank — and every one is IAST *romanization*, with a median
native-script character ratio of **0.000**:

> `pūrvasya medhājananāni śuka sāri kṛśānām jihvāḥ badhnāti āśayati...`

A language-ID filter labels this "Sanskrit" with high confidence and admits it straight
into a verified native tier. It is perfectly good data — but it is romanised data, and a
plan that counts it as native-script Indic has silently overstated its verified supply by
half. This is a miniature of exactly the risk Section IV's tier structure exists to
prevent, discovered by running the filter rather than by reasoning about it.

We therefore **reclassify rather than discard**: those documents were routed to the
romanised tier, where they are genuinely valuable (a large share of real Indian user input
is Latin-script). The lesson generalises to the full inventory — **LID is not a script
gate**, and any Indic supply figure not separated by script should be treated as an upper
bound. We recommend re-auditing Sangraha's verified tier on this basis before trusting the
74.6B figure.

Second finding: **743 of 2,600 agentic traces (28.6%) leaked absolute local paths**
(`/Users/...`) into their observations, because real command output contains real paths.
Left unscrubbed, this is both a privacy leak and a memorisation target. Any
execution-grounded synthesis pipeline needs path normalisation as a first-class step, not
an afterthought.

Near-duplicate rates were low here (≤ 0.1%) because the sources are already curated;
against raw crawl we expect 30–60% and the same code path handles it.

---

## X. Risk register and reproduction

| Risk | Trigger | Response |
|---|---|---|
| MT quality gate rejects > 25% | Round-trip chrF++ below threshold | Spend the **35.9B unverified-native contingency**; shift the translated tier from 40% → 30% and unverified 30% → 40%. Supply exists for this today. |
| Agentic synthesis accept rate < 40% | Verification pass rate in generation logs | Cut the agentic lane to 4% (its floor) and move the difference to code. Do **not** pad with unverified rollouts. |
| Permissive code supply shrinks (licence changes) | Re-audit of Stack v2 | Code drops to 28%, web absorbs the difference. We do not substitute non-permissive code. |
| MMLU regression > 2 points at 1B proxy | Go/no-go table above | Web 35% → 40%, code 32% → 28%. |
| Gradient spikes persist at stage boundaries | `max_grad_norm` > 5× median | Widen the blend band from 10B to 25B tokens and decouple sequence-length changes further. |
| Indic knee is above 8% | `indic_16` still improving strongly | Raise Indic to 12%, taken from web. |

### Reproduce everything

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python numpy torch

make check      # validate spec consistency, print the supply audit
make data       # fetch real corpora, synthesise agentic traces, clean, shard
make ablation   # 13 training runs (~100 min on an M-series laptop GPU)
make tables     # regenerate every table in this README
```

| Path | What it is |
|---|---|
| [`inventory/inventory.json`](inventory/inventory.json) | Token supply per dataset, with provenance and overlap discounts |
| [`inventory/mixture.json`](inventory/mixture.json) | The budget, floors, curriculum, anneal reserve, effort bands |
| [`scripts/budget.py`](scripts/budget.py) | Supply audit, repetition pricing, consistency validator |
| [`scripts/synthesis_cost.py`](scripts/synthesis_cost.py) | GPU-day cost of the synthetic lanes |
| [`scripts/build_tables.py`](scripts/build_tables.py) | Generates and injects every table here |
| [`proxy/agentic_synth.py`](proxy/agentic_synth.py) | Execution-grounded agentic trace generator |
| [`proxy/clean.py`](proxy/clean.py) | Cleaning pipeline and report |
| [`proxy/train.py`](proxy/train.py) | Mixture sampler, trainer, per-lane evaluation |
| [`proxy/run_ablation.py`](proxy/run_ablation.py) | Arm definitions and sweep driver |
| [`results/`](results/) | Ablation results, cleaning report, run logs |

### Sources for inventory figures

Token counts are drawn from the dataset cards and papers for DCLM-Baseline, FineWeb-Edu,
Nemotron-CC, The Stack v2 / StarCoder2, Stack-Edu, MegaMath, FineMath, OpenWebMath,
Proof-Pile-2, peS2o, Sangraha / IndicLLMSuite, IndicCorp v2, Varta, MADLAD-400,
BPCC / Samanantar, ToolBench, xLAM / APIGen, and PG-19. Every row carries a
`provenance` field of `published`, `estimate` or `derived`; treat `estimate` rows as
±30% and `low` confidence rows as ±50%.
