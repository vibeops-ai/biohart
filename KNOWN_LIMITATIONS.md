# Known limitations

biohart is an early pilot. This file states plainly what it does and does not
support, so nobody (including us) over-claims from it.

## 1. `reversed_answer` is not a valid semantic reversal

The `reversed_answer` mutation prepends the string
`"Contrary to what an honest reading would conclude: "` to the candidate answer
and flips the label field. It does **not** change the substance of the answer:
the original (correct) claim is still present in the text. A competent judge that
reads the content therefore correctly keeps its verdict, and the harness scores
that as a "miss."

Consequence: a low value in the `reversed_answer` column does **not** mean the
judge was fooled. It largely means the probe did not create a genuinely wrong
answer. This column is retained for transparency and as a weak
surface-sensitivity signal, but it is **excluded from the headline metric**.

An earlier version of the summary reported this probe as the headline result
(e.g. "Opus catches 87% of reversals"). That number was (a) from the surface
probe, not a real reversal, and (b) hand-written prose that did not match the
computed table. Both are fixed: all prose is now generated from the data, and
the headline is built on `single_word_negation_drop`.

Roadmap: replace `reversed_answer` with a genuine reversal (an LLM-generated
answer that fluently asserts the opposite, clinically-plausible-but-wrong
conclusion) and re-run. Until then, do not cite it.

## 2. HealthBench is exploratory, PubMedQA is the valid dataset

The probe method assumes a passage-grounded factual answer that can be made
wrong-relative-to-passages by a mutation. HealthBench answers are conversational
(clarifying questions, refusals, advice), not passage-grounded claims, so
mutating them does not reliably create a wrong case. The HealthBench numbers are
retained for transparency but should not be cited as a judge ranking. Use the
PubMedQA leaderboard.

## 3. Small n

n=30 tasks per dataset, and the valid headline probe is applicable only where a
negation token exists: **n=12 on PubMedQA**, ~22 on HealthBench. `numeric_swap`
is applicable on n=1 on PubMedQA (reported for direction only). Per-cell binomial
CIs are roughly +/- 0.2. Treat the leaderboard as a pilot signal, not a
definitive ranking. Bigger, negation-rich, held-out sets are the fix.

## 4. Single prompt, single temperature

One judge prompt template at temperature 0.1. Prior work shows LLM-judge results
swing 38-49pp on prompt phrasing, so assume meaningful prompt-fragility until a
multi-prompt sensitivity sweep is run.

## 5. Contamination

Public PubMedQA and HealthBench have been in training data for years; some judges
may have memorized specific items. A synthetic, held-out biomedical-QA set is the
recommended defense and a planned next step.

## 6. Cost figures are estimates

Per-call cost is estimated from public per-token pricing approximations in
`biohart/judge.py`, not from provider invoices. Treat dollar figures as ballpark.

## What biohart DOES support today

A reproducible, deterministic method showing that biomedical-QA LLM judges catch
between ~58% and ~92% of genuine single-word negation inversions on a 12-case
PubMedQA slice, while holding steady on negative-control edits. That is a real,
if modest, signal that judges (especially cheaper ones) miss a non-trivial
fraction of meaning inversions, which matters when a judge is the reward in a
healthcare-RL loop. Everything beyond that sentence is roadmap, not result.
