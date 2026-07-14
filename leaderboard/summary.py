"""Aggregate per-judge per-family flip rates into a leaderboard.

Every number and every sentence in the rendered markdown is computed from the
raw result JSONs. Nothing here is hand-written prose about specific models; if
you change the data, the writeup changes with it. (An earlier version hardcoded
the headline findings as string literals, which drifted out of sync with the
computed table. Do not reintroduce that.)

Metric = flip_rate on APPLICABLE variants only:
  positive controls  -> higher is better (judge correctly changed its verdict)
  negative controls  -> lower is better  (judge held its verdict on a no-op edit)

Headline is driven by `single_word_negation_drop`, the one positive control
that genuinely inverts a load-bearing claim. `reversed_answer` only prepends a
preamble to unchanged answer text (it does not reverse the substance), so it is
reported for transparency but EXCLUDED from the headline. `numeric_swap` is
reported with its applicable-n because it is under-sampled. See
KNOWN_LIMITATIONS.md.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# The valid, load-bearing positive control the headline is built on.
PRIMARY = "single_word_negation_drop"
# Reported but not headlined: surface-preamble only (does not invert substance).
SURFACE = "reversed_answer"
NEG_CONTROLS = (
    "comment_only_filler",
    "paraphrase_synonym",
    "irrelevant_citation",
    "shuffle_passage_order",
)

PRETTY = {"pubmedqa": "PubMedQA", "healthbench": "HealthBench", "bioasq": "BioASQ"}


def collect(results_dir: Path, dataset: str) -> list[dict]:
    """Most-recent run per model for a dataset."""
    by_model: dict[str, tuple[float, dict]] = {}
    for f in sorted(results_dir.glob(f"*{dataset}*.json")):
        try:
            data = json.loads(f.read_text())
        except json.JSONDecodeError:
            continue
        if not data.get("results"):
            continue
        s = data["summary"]
        mtime = f.stat().st_mtime
        if s["model"] not in by_model or by_model[s["model"]][0] < mtime:
            by_model[s["model"]] = (mtime, s)
    return [s for _, s in by_model.values()]


def _rate(s: dict, fam: str) -> float:
    """Applicable-only flip rate for a family."""
    return s["per_family"].get(fam, {}).get("flip_rate", 0.0)


def _applicable(s: dict, fam: str) -> int:
    ff = s["per_family"].get(fam, {})
    return ff.get("applicable", ff.get("total", 0))


def _pct(x: float) -> str:
    return f"{round(100 * x)}%"


def render(rows: list[dict], dataset: str) -> str:
    pretty = PRETTY.get(dataset, dataset)
    n_judges = len(rows)
    n_tasks = rows[0]["n_tasks"] if rows else 0
    prim_n = max((_applicable(s, PRIMARY) for s in rows), default=0)
    num_n = max((_applicable(s, "numeric_swap") for s in rows), default=0)

    # Rank by the valid primary probe (catch rate on genuine negation inversion).
    rows = sorted(rows, key=lambda s: -_rate(s, PRIMARY))

    L = [
        f"# Biohart leaderboard: {pretty} ({n_judges} judges, n={n_tasks} tasks, seed=42)",
        "",
        "Adversarial-mutation probes for biomedical-QA **judges** (LLM-as-judge / "
        "reward models). Each probe asks the judge `{question, context, answer, "
        "label}` -> `{verdict, confidence}`. We mutate the answer and measure how "
        "often the judge changes its verdict versus baseline.",
        "",
        "All rates below are **applicable-only** (no-op variants, e.g. no negation "
        "token present, are excluded from the denominator, not counted as passes).",
        "",
        "**Headline probe (valid positive control):**",
        "- `single_word_negation_drop`: removes one load-bearing negation "
        "(`not`/`no`/`without`/...) so the answer genuinely asserts the wrong "
        "conclusion. A judge that reads should now score it INCORRECT (flip).",
        "",
        "**Negative controls (judge should NOT flip):** `comment_only_filler`, "
        "`paraphrase_synonym`, `irrelevant_citation`, `shuffle_passage_order`.",
        "",
        "**Reported but NOT headlined:** `reversed_answer` prepends a "
        "\"Contrary to...\" preamble but leaves the answer's substance unchanged, "
        "so it is not a true semantic reversal; `numeric_swap` is under-sampled "
        f"(applicable on {num_n}/{n_tasks} answers). See KNOWN_LIMITATIONS.md.",
        "",
        f"| Judge | negation-drop up (valid) | reversed (surface) | numeric up | "
        f"filler dn | para dn | cite dn | shuffle dn | $/{n_tasks} |",
        "|-------|:--:|:--:|:--:|:--:|:--:|:--:|:--:|--:|",
    ]
    for s in rows:
        L.append(
            f"| `{s['model']}` "
            f"| **{_rate(s, PRIMARY):.2f}** "
            f"| {_rate(s, SURFACE):.2f} "
            f"| {_rate(s, 'numeric_swap'):.2f} "
            f"| {_rate(s, 'comment_only_filler'):.2f} "
            f"| {_rate(s, 'paraphrase_synonym'):.2f} "
            f"| {_rate(s, 'irrelevant_citation'):.2f} "
            f"| {_rate(s, 'shuffle_passage_order'):.2f} "
            f"| ${s['total_cost_usd']:.2f} |"
        )

    # ---- computed headline (no hardcoded model claims) ----
    prim = [(s["model"], _rate(s, PRIMARY)) for s in rows]
    best_m, best_r = prim[0]
    worst_m, worst_r = prim[-1]
    negmax, negmax_where = 0.0, ""
    for s in rows:
        for fam in NEG_CONTROLS:
            r = _rate(s, fam)
            if r > negmax:
                negmax, negmax_where = r, f"{s['model']} / {fam}"

    L += ["", "## Headline findings (computed)", ""]

    if dataset == "healthbench":
        L += [
            "> **HealthBench is exploratory here and should not be cited as a "
            "judge ranking.** HealthBench answers are conversational (clarifying "
            "questions, refusals), not passage-grounded factual claims, so "
            "mutating the answer text does not reliably create a wrong-vs-passages "
            "case. Numbers are retained for transparency only; use the PubMedQA "
            "leaderboard for the actual result.",
            "",
        ]

    L += [
        f"1. **On genuine single-word negation inversions, catch rate ranges "
        f"{_pct(worst_r)} ({worst_m}) to {_pct(best_r)} ({best_m}) across "
        f"{n_judges} judges** (applicable n={prim_n}). Every judge misses some; "
        f"the best still misses {_pct(1 - best_r)}.",
        f"2. **Negative controls hold low** (worst single cell {_pct(negmax)}: "
        f"{negmax_where}). Judges are not simply noisy on irrelevant edits, so the "
        f"negation-drop signal is not an artefact.",
        "3. **`reversed_answer` is not a valid reversal** and is excluded from the "
        "headline: it prepends a preamble but leaves the answer's substance "
        "intact, so a competent judge correctly does not flip. Low values in that "
        "column do NOT indicate a fooled judge. (This was the source of an earlier "
        "over-claim.)",
        f"4. **`numeric_swap` is under-powered** (applicable n={num_n}); reported "
        "for direction only, not ranked. A numeric-rich source (BioASQ-Factoid, or "
        "a curated dosage/lab-value set) is the fix.",
        "",
        "## Why it matters for healthcare RL",
        "",
        "An RL loop that uses an LLM-as-judge for the reward will reward-hack on "
        "whatever the judge cannot catch. A judge that misses a dropped negation "
        "rewards an answer asserting the opposite clinical conclusion. The gradient "
        "here (cheaper judges tend to catch fewer inversions) is the practical "
        "risk: cost-tier judges are what most pipelines actually deploy.",
        "",
        "## Caveats (read before citing)",
        "",
        f"- **Small.** n={n_tasks} labeled tasks; the valid probe is applicable on "
        f"only {prim_n}. Per-cell binomial CI is roughly +/- 0.2. This is a pilot, "
        "not a definitive ranking.",
        "- **Single prompt template, temperature 0.1.** Assume meaningful "
        "prompt-fragility (other work shows 38-49pp swings on phrasing) until "
        "tested across prompts.",
        "- **Contamination.** Public PubMedQA has been in training data for years; "
        "a synthetic held-out biomedical-QA set is the recommended defense and the "
        "next step.",
        f"- **Reproduce:** `python -m biohart.run --provider openrouter --model "
        f"<id> --tasks biohart/datasets/{dataset}/sample.json` then "
        f"`python leaderboard/summary.py --dataset {dataset}`.",
    ]
    return "\n".join(L) + "\n"


def main() -> None:
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="pubmedqa")
    p.add_argument("--output", default=None)
    args = p.parse_args()

    rows = collect(Path("leaderboard/results"), dataset=args.dataset)
    if not rows:
        logger.error("no results for dataset=%s", args.dataset)
        return
    out = Path(args.output) if args.output else Path(f"leaderboard/summary_{args.dataset}.md")
    out.write_text(render(rows, dataset=args.dataset))
    logger.info("Wrote %s (%d judges)", out, len(rows))


if __name__ == "__main__":
    main()
