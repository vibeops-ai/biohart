# biohart

**Adversarial probes for biomedical-QA LLM judges.**

Companion to [goodhart](https://github.com/vibeops-ai/goodhart) (the code-RL judge benchmark). Same philosophy — semantically-clean and semantically-flipped mutations of a candidate answer — applied to biomedical question answering.

A health-RL training pipeline that uses an LLM-as-judge for the reward signal will reward-hack the policy in proportion to whatever the judge can be fooled by. biohart measures that.

This repository accompanies the post **[Medical AI's Trusting Trust Problem](https://getbiostack.com/blog/medical-ai-trusting-trust-problem)** (Biostack × VibeOps Research, May 2026, [cross-post](https://vibeops.tech/research/medical-ai-trusting-trust-problem)).

---

## What's in this repo

```
biohart/
  mutations.py       7-family mutation engine over QATask records
  judge.py           multi-provider judge runner (Anthropic / OpenAI / Google / Cohere / Mistral / OpenRouter / Azure)
  run.py             leaderboard runner
  datasets/
    healthbench/     30 OpenAI HealthBench tasks (long-form medical advice with rubric criteria)
    pubmedqa/        30 PubMedQA tasks (biomedical research, yes/no/maybe labels)
    bioasq/          30 BioASQ tasks (biomedical factoid QA)

leaderboard/
  results/           per-judge per-dataset scored JSONs
  summary.py         aggregator
  summary_healthbench.md   HealthBench leaderboard (9 judges)
  summary_pubmedqa.md      PubMedQA leaderboard (12 judges)

scripts/
  sample_healthbench.py    samplers used to build the 30-task slices
  sample_pubmedqa.py
  sample_bioasq.py
```

## Probe families

For each `(question, context, candidate_answer, claimed_label)` we ask the judge whether the candidate answer is supported by the context. Then we mutate the answer:

**Negative controls (judge SHOULD NOT flip):**
- `comment_only_filler` — append a non-substantive scientific filler ("Further work in larger cohorts is warranted.")
- `paraphrase_synonym` — domain-neutral synonym substitutions (patients → subjects, treatment → therapy)
- `irrelevant_citation` — append an unrelated PMID reference
- `shuffle_passage_order` — reorder context passages (lossless: same evidence, different sequence)

**Positive controls (judge SHOULD flip the verdict):**
- `reversed_answer` — prefix `"Contrary to what an honest reading would conclude:"` + flip the label
- `single_word_negation_drop` — delete one `not` / `no` / `without` / `never` / `cannot` token + flip the label
- `numeric_swap` — replace one percentage or count with a wildly different value + flip the label

## Headline result

On the 30-task HealthBench slice, across 9 frontier judges:

- The single best model on the easiest attack (Gemini 3.1 Pro Preview on `reversed_answer`) catches the flip **57%** of the time.
- Claude Opus 4.7 ($6.15 per 30 tasks) catches the same attack **17%** of the time.
- GPT-4.1 mini ($0.09 per 30 tasks) catches it **10%**.
- Negative-control flip rates stay below 13% across the board — judges are not noisy, they are inattentive to meaning changes.

See [`leaderboard/summary_healthbench.md`](leaderboard/summary_healthbench.md) for the full table.

## Reproduce

```bash
# Install
pip install -e .

# Set API keys for whichever judges you want to score
export ANTHROPIC_API_KEY=...
export OPENAI_API_KEY=...
export GOOGLE_API_KEY=...
export OPENROUTER_API_KEY=...

# Score a judge on the HealthBench slice
python -m biohart.run --dataset healthbench --judge claude-opus-4-7

# Regenerate summary tables
python leaderboard/summary.py
```

Total cost to reproduce the full HealthBench leaderboard across all 9 judges: ~$20 in API spend.

## Sample sizes and caveats

- 30 tasks per dataset (seed=42). Per-cell binomial 95% Wilson CI ≈ ±0.18 at p=0.5.
- `single_word_negation_drop` applies only to answers containing a negation token (typically n=22 of 30).
- `numeric_swap` applies only to answers containing a numeric value (typically n=15 of 30).
- Single prompt template at temperature 0.1. The code-side prompt-fragility study showed 38-49pp swings on prompt phrasing; assume similar here until tested.
- Public PubMedQA and HealthBench have been in training data for some models. Holdout-style synthetic biomedical-QA tasks are the recommended contamination defense for repeated runs.

## Related work

- **Master-RM** (Tencent + Princeton + UVa, July 2025): adversarial fine-tuning for reward models, demonstrated on code-RL — [arXiv:2507.08794](https://arxiv.org/abs/2507.08794)
- **goodhart** — code-RL companion benchmark — [github.com/vibeops-ai/goodhart](https://github.com/vibeops-ai/goodhart)
- **VibeOps Research** — [vibeops.tech/research](https://vibeops.tech/research)

## Citation

If you use biohart in research, please cite:

```bibtex
@misc{biohart2026,
  title  = {biohart: Adversarial probes for biomedical-QA LLM judges},
  author = {Kumar, Shivam Pankaj and Mishra, Sanat and Bararia, Swati and Raj, Kislay},
  year   = {2026},
  url    = {https://github.com/vibeops-ai/biohart}
}
```

## License

MIT. See [`LICENSE`](LICENSE).

## Contact

Questions, replication issues, or commercial inquiries: **hi@vibeops.tech**
