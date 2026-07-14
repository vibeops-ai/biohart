# Biohart leaderboard: HealthBench (9 judges, n=30 tasks, seed=42)

Adversarial-mutation probes for biomedical-QA **judges** (LLM-as-judge / reward models). Each probe asks the judge `{question, context, answer, label}` -> `{verdict, confidence}`. We mutate the answer and measure how often the judge changes its verdict versus baseline.

All rates below are **applicable-only** (no-op variants, e.g. no negation token present, are excluded from the denominator, not counted as passes).

**Headline probe (valid positive control):**
- `single_word_negation_drop`: removes one load-bearing negation (`not`/`no`/`without`/...) so the answer genuinely asserts the wrong conclusion. A judge that reads should now score it INCORRECT (flip).

**Negative controls (judge should NOT flip):** `comment_only_filler`, `paraphrase_synonym`, `irrelevant_citation`, `shuffle_passage_order`.

**Reported but NOT headlined:** `reversed_answer` prepends a "Contrary to..." preamble but leaves the answer's substance unchanged, so it is not a true semantic reversal; `numeric_swap` is under-sampled (applicable on 15/30 answers). See KNOWN_LIMITATIONS.md.

| Judge | negation-drop up (valid) | reversed (surface) | numeric up | filler dn | para dn | cite dn | shuffle dn | $/30 |
|-------|:--:|:--:|:--:|:--:|:--:|:--:|:--:|--:|
| `gpt-4o-mini` | **0.50** | 0.37 | 0.60 | 0.07 | 0.00 | 0.10 | 0.13 | $0.03 |
| `mistralai/mistral-medium-3-5` | **0.36** | 0.30 | 0.27 | 0.03 | 0.14 | 0.00 | 0.03 | $0.37 |
| `gpt-4.1-mini` | **0.32** | 0.10 | 0.27 | 0.03 | 0.00 | 0.03 | 0.07 | $0.09 |
| `claude-haiku-4-5-20251001` | **0.27** | 0.13 | 0.40 | 0.00 | 0.00 | 0.07 | 0.17 | $0.32 |
| `cohere/command-a` | **0.27** | 0.33 | 0.20 | 0.03 | 0.14 | 0.03 | 0.03 | $0.62 |
| `gpt-4.1` | **0.27** | 0.27 | 0.40 | 0.13 | 0.14 | 0.07 | 0.07 | $0.46 |
| `google/gemini-3.1-pro-preview` | **0.23** | 0.57 | 0.67 | 0.10 | 0.14 | 0.10 | 0.10 | $2.24 |
| `claude-opus-4-7` | **0.09** | 0.17 | 0.20 | 0.03 | 0.00 | 0.13 | 0.03 | $6.15 |
| `claude-sonnet-4-6` | **0.09** | 0.13 | 0.13 | 0.07 | 0.00 | 0.10 | 0.27 | $0.84 |

## Headline findings (computed)

> **HealthBench is exploratory here and should not be cited as a judge ranking.** HealthBench answers are conversational (clarifying questions, refusals), not passage-grounded factual claims, so mutating the answer text does not reliably create a wrong-vs-passages case. Numbers are retained for transparency only; use the PubMedQA leaderboard for the actual result.

1. **On genuine single-word negation inversions, catch rate ranges 9% (claude-sonnet-4-6) to 50% (gpt-4o-mini) across 9 judges** (applicable n=22). Every judge misses some; the best still misses 50%.
2. **Negative controls hold low** (worst single cell 27%: claude-sonnet-4-6 / shuffle_passage_order). Judges are not simply noisy on irrelevant edits, so the negation-drop signal is not an artefact.
3. **`reversed_answer` is not a valid reversal** and is excluded from the headline: it prepends a preamble but leaves the answer's substance intact, so a competent judge correctly does not flip. Low values in that column do NOT indicate a fooled judge. (This was the source of an earlier over-claim.)
4. **`numeric_swap` is under-powered** (applicable n=15); reported for direction only, not ranked. A numeric-rich source (BioASQ-Factoid, or a curated dosage/lab-value set) is the fix.

## Why it matters for healthcare RL

An RL loop that uses an LLM-as-judge for the reward will reward-hack on whatever the judge cannot catch. A judge that misses a dropped negation rewards an answer asserting the opposite clinical conclusion. The gradient here (cheaper judges tend to catch fewer inversions) is the practical risk: cost-tier judges are what most pipelines actually deploy.

## Caveats (read before citing)

- **Small.** n=30 labeled tasks; the valid probe is applicable on only 22. Per-cell binomial CI is roughly +/- 0.2. This is a pilot, not a definitive ranking.
- **Single prompt template, temperature 0.1.** Assume meaningful prompt-fragility (other work shows 38-49pp swings on phrasing) until tested across prompts.
- **Contamination.** Public PubMedQA has been in training data for years; a synthetic held-out biomedical-QA set is the recommended defense and the next step.
- **Reproduce:** `python -m biohart.run --provider openrouter --model <id> --tasks biohart/datasets/healthbench/sample.json` then `python leaderboard/summary.py --dataset healthbench`.
