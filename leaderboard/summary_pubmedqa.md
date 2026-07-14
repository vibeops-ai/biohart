# Biohart leaderboard: PubMedQA (12 judges, n=30 tasks, seed=42)

Adversarial-mutation probes for biomedical-QA **judges** (LLM-as-judge / reward models). Each probe asks the judge `{question, context, answer, label}` -> `{verdict, confidence}`. We mutate the answer and measure how often the judge changes its verdict versus baseline.

All rates below are **applicable-only** (no-op variants, e.g. no negation token present, are excluded from the denominator, not counted as passes).

**Headline probe (valid positive control):**
- `single_word_negation_drop`: removes one load-bearing negation (`not`/`no`/`without`/...) so the answer genuinely asserts the wrong conclusion. A judge that reads should now score it INCORRECT (flip).

**Negative controls (judge should NOT flip):** `comment_only_filler`, `paraphrase_synonym`, `irrelevant_citation`, `shuffle_passage_order`.

**Reported but NOT headlined:** `reversed_answer` prepends a "Contrary to..." preamble but leaves the answer's substance unchanged, so it is not a true semantic reversal; `numeric_swap` is under-sampled (applicable on 1/30 answers). See KNOWN_LIMITATIONS.md.

| Judge | negation-drop up (valid) | reversed (surface) | numeric up | filler dn | para dn | cite dn | shuffle dn | $/30 |
|-------|:--:|:--:|:--:|:--:|:--:|:--:|:--:|--:|
| `claude-sonnet-4-6` | **0.92** | 0.63 | 1.00 | 0.07 | 0.00 | 0.00 | 0.03 | $0.60 |
| `google/gemini-3.1-pro-preview` | **0.92** | 0.77 | 1.00 | 0.07 | 0.12 | 0.03 | 0.07 | $1.49 |
| `mistralai/mistral-medium-3-5` | **0.83** | 0.80 | 1.00 | 0.07 | 0.00 | 0.00 | 0.00 | $0.27 |
| `claude-opus-4-7` | **0.75** | 0.87 | 1.00 | 0.00 | 0.00 | 0.07 | 0.03 | $4.44 |
| `cohere/command-a` | **0.75** | 0.83 | 0.00 | 0.03 | 0.00 | 0.00 | 0.03 | $0.46 |
| `gpt-4.1` | **0.75** | 0.53 | 1.00 | 0.00 | 0.00 | 0.03 | 0.00 | $0.33 |
| `gpt-4o-mini` | **0.75** | 0.33 | 1.00 | 0.03 | 0.00 | 0.00 | 0.03 | $0.02 |
| `x-ai/grok-4.3` | **0.75** | 0.70 | 1.00 | 0.10 | 0.12 | 0.10 | 0.10 | $0.50 |
| `claude-haiku-4-5-20251001` | **0.67** | 0.47 | 0.00 | 0.07 | 0.00 | 0.03 | 0.13 | $0.24 |
| `gpt-4.1-mini` | **0.67** | 0.40 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | $0.07 |
| `gpt-4o` | **0.58** | 0.50 | 1.00 | 0.07 | 0.00 | 0.03 | 0.07 | $0.42 |
| `meta-llama/llama-4-maverick` | **0.58** | 0.20 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | $0.55 |

## Headline findings (computed)

1. **On genuine single-word negation inversions, catch rate ranges 58% (meta-llama/llama-4-maverick) to 92% (claude-sonnet-4-6) across 12 judges** (applicable n=12). Every judge misses some; the best still misses 8%.
2. **Negative controls hold low** (worst single cell 13%: claude-haiku-4-5-20251001 / shuffle_passage_order). Judges are not simply noisy on irrelevant edits, so the negation-drop signal is not an artefact.
3. **`reversed_answer` is not a valid reversal** and is excluded from the headline: it prepends a preamble but leaves the answer's substance intact, so a competent judge correctly does not flip. Low values in that column do NOT indicate a fooled judge. (This was the source of an earlier over-claim.)
4. **`numeric_swap` is under-powered** (applicable n=1); reported for direction only, not ranked. A numeric-rich source (BioASQ-Factoid, or a curated dosage/lab-value set) is the fix.

## Why it matters for healthcare RL

An RL loop that uses an LLM-as-judge for the reward will reward-hack on whatever the judge cannot catch. A judge that misses a dropped negation rewards an answer asserting the opposite clinical conclusion. The gradient here (cheaper judges tend to catch fewer inversions) is the practical risk: cost-tier judges are what most pipelines actually deploy.

## Caveats (read before citing)

- **Small.** n=30 labeled tasks; the valid probe is applicable on only 12. Per-cell binomial CI is roughly +/- 0.2. This is a pilot, not a definitive ranking.
- **Single prompt template, temperature 0.1.** Assume meaningful prompt-fragility (other work shows 38-49pp swings on phrasing) until tested across prompts.
- **Contamination.** Public PubMedQA has been in training data for years; a synthetic held-out biomedical-QA set is the recommended defense and the next step.
- **Reproduce:** `python -m biohart.run --provider openrouter --model <id> --tasks biohart/datasets/pubmedqa/sample.json` then `python leaderboard/summary.py --dataset pubmedqa`.
