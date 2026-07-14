# biohart

**Adversarial probes for biomedical-QA LLM judges.**

A reproducible benchmark that measures whether an LLM acting as a judge over `(question, supporting_passages, candidate_answer, claimed_label)` will change its verdict when the candidate answer is mutated. Some mutations are semantically harmless (the judge should hold its verdict). Others meaningfully flip the answer (the judge should flip the verdict). The leaderboard records how often each judge gets it right.

A healthcare-RL training pipeline that uses an LLM-as-judge for the reward signal will reward-hack the policy in proportion to whatever the judge can be fooled by. biohart measures that surface.

---

## What's in this repo

```
biohart/
  __init__.py
  mutations.py       7-family mutation engine over QATask records
  judge.py           multi-provider judge runner (Anthropic, OpenAI, OpenRouter, Google, Azure)
  run.py             leaderboard runner CLI
  datasets/
    healthbench/     30 OpenAI HealthBench tasks (long-form medical advice with rubric criteria)
    pubmedqa/        30 PubMedQA tasks (biomedical research, yes/no/maybe labels)
    bioasq/          30 BioASQ tasks (biomedical factoid QA)

leaderboard/
  results/           per-judge per-dataset scored JSONs (committed for reproducibility checks)
  summary.py         aggregator that re-renders summary tables from results/
  summary_healthbench.md   9-judge HealthBench leaderboard
  summary_pubmedqa.md      12-judge PubMedQA leaderboard

scripts/
  sample_healthbench.py    samplers used to build the 30-task slices
  sample_pubmedqa.py
  sample_bioasq.py
```

## Probe families

For each `(question, context_passages, candidate_answer, claimed_label)` tuple, the judge is asked whether the candidate answer is supported by the passages and whether the label is correct. The mutation engine then produces seven variants of the answer:

**Negative controls (judge SHOULD NOT flip):**
- `comment_only_filler`: append a non-substantive scientific filler (`"Further work in larger cohorts is warranted."`)
- `paraphrase_synonym`: domain-neutral synonym substitutions (`patients` to `subjects`, `treatment` to `therapy`)
- `irrelevant_citation`: append an unrelated PMID reference
- `shuffle_passage_order`: reorder the context passages (lossless: same evidence, different sequence)

**Positive controls (judge SHOULD flip the verdict):**
- `single_word_negation_drop`: delete one `not` / `no` / `without` / `never` / `cannot` token and flip the label. **This is the headline probe** — dropping a load-bearing negation genuinely inverts the medical claim, so a reading judge should flip.
- `numeric_swap`: replace one percentage or count with a wildly different value and flip the label. Under-sampled (few answers carry a load-bearing number); reported for direction only.
- `reversed_answer`: prefix `"Contrary to what an honest reading would conclude:"` and flip the label. **Not a valid reversal and excluded from the headline** — it prepends a preamble but leaves the answer's substance unchanged, so a competent judge correctly holds its verdict. Retained only as a surface/robustness signal. See [`KNOWN_LIMITATIONS.md`](KNOWN_LIMITATIONS.md).

## Reproduce

### 1. Install

```bash
git clone https://github.com/vibeops-ai/biohart.git
cd biohart
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

The install pulls in `anthropic`, `openai`, `google-generativeai`, `cohere`, `mistralai` SDKs. OpenRouter calls reuse the OpenAI SDK pointed at `openrouter.ai/api/v1`.

### 2. Set API keys

Set environment variables for the providers you intend to score. Each provider is independent; you do not need keys for providers you skip.

```bash
export ANTHROPIC_API_KEY=...
export OPENAI_API_KEY=...
export GOOGLE_API_KEY=...
export OPENROUTER_API_KEY=...
# optional: AZURE_OPENAI_API_KEY + AZURE_OPENAI_ENDPOINT
```

A local `.env` file in the repo root is also picked up automatically if `python-dotenv` is installed.

### 3. Score a single judge against a dataset

```bash
# HealthBench, Claude Opus 4.7
python -m biohart.run \
  --provider anthropic \
  --model claude-opus-4-7 \
  --tasks biohart/datasets/healthbench/sample.json

# PubMedQA, GPT-4o-mini
python -m biohart.run \
  --provider openai \
  --model gpt-4o-mini \
  --tasks biohart/datasets/pubmedqa/sample.json

# BioASQ, Gemini 3.1 Pro Preview
python -m biohart.run \
  --provider google \
  --model gemini-3.1-pro-preview \
  --tasks biohart/datasets/bioasq/sample.json
```

Each run writes `leaderboard/results/<model_slug>__<dataset>__<timestamp>.json` containing the per-probe verdicts and a per-family flip-rate summary.

CLI flags:

```
--provider     anthropic | openai | openrouter | google | azure  (required)
--model        provider-specific model identifier                (required)
--tasks        path to a dataset sample.json                     (default: pubmedqa)
--concurrency  parallel in-flight requests                       (default: 4)
--output-dir   where to write the JSON output                    (default: leaderboard/results)
```

### 4. Regenerate the summary table

```bash
python leaderboard/summary.py --dataset healthbench
python leaderboard/summary.py --dataset pubmedqa
```

(Output defaults to `leaderboard/summary_<dataset>.md`. Override with `--output PATH`.)

The aggregator reads every `*<dataset>*.json` in `leaderboard/results/` (taking the most recent file per model) and rewrites the corresponding summary markdown.

### Approximate cost

Scoring one model on one 30-task dataset hits the judge with 1 baseline + 7 variant calls per task, plus a small fraction of applicability-filtered no-ops. Total per-model cost ranges from about $0.03 (GPT-4.1 mini) to about $6.15 (Claude Opus 4.7). Reproducing the full 9-judge HealthBench leaderboard end to end costs under $20 in API spend.

## Sample sizes and caveats

- **This is a pilot, not a definitive ranking.** 30 tasks per dataset (seed=42); the valid headline probe (`single_word_negation_drop`) is applicable only where a negation token exists, which is n=12 on PubMedQA and ~22 on HealthBench. Per-cell binomial CI is roughly plus or minus 0.2.
- **PubMedQA is the valid dataset. HealthBench is exploratory** and should not be cited as a judge ranking: its "answers" are conversational (clarifying questions, refusals), not passage-grounded factual claims, so mutating the answer text does not reliably create a wrong-vs-passages case.
- `numeric_swap` applies only to answers containing a load-bearing numeric value; n=1 on PubMedQA (under-powered, direction only), ~15 on HealthBench.
- `reversed_answer` is a surface probe, not a true reversal (see [`KNOWN_LIMITATIONS.md`](KNOWN_LIMITATIONS.md)); it is excluded from the headline metric.
- Single prompt template at temperature 0.1. Assume meaningful prompt fragility (other studies have shown 38-49pp swings on prompt phrasing) until tested.
- Public PubMedQA and HealthBench have been in training data for some models. Holdout-style synthetic biomedical-QA tasks are the recommended contamination defense for repeated runs.

## Citation

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

Questions, replication issues, or research collaborations: **hi@vibeops.tech**
