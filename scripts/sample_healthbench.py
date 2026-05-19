"""Pull a HealthBench sample mapped onto the biohart QATask schema.

Schema mapping:
  question           <- prompt[0].content (the user's clinical query)
  context_passages   <- rubric criteria (each criterion = one "passage")
  long_answer        <- ideal_completions_data.ideal_completion
  final_decision     <- "yes" (we treat the physician-written gold as
                        the correct answer; the judge's job is to score
                        whether this answer matches the rubric)
  pubid              <- prompt_id

This lets the existing 7 mutation families and the existing run.py +
summary.py work without modification. HealthBench ideal completions
are long-form clinical advice and almost always contain numeric
claims (doses, frequencies, percentages), so the `numeric_swap`
positive-control probe finally fires at non-trivial rates here -
which was the structural gap on PubMedQA.

Output: biohart/datasets/healthbench/{sample.json, index.json}
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datasets import load_dataset

from biohart.mutations import QATask


def main() -> None:
    n = 30
    seed = 42
    out_dir = Path("biohart/datasets/healthbench")
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(
        "openai/healthbench",
        data_files="2025-05-07-06-14-12_oss_eval.jsonl",
        split="train",
    )

    rng = random.Random(seed)
    indices = list(range(len(ds)))
    rng.shuffle(indices)

    tasks: list[dict] = []
    for i in indices:
        if len(tasks) >= n:
            break
        ex = ds[i]
        icd = ex.get("ideal_completions_data") or {}
        ideal = icd.get("ideal_completion")
        if not ideal:
            continue
        rubrics = ex.get("rubrics") or []
        if not rubrics:
            continue
        # Skip multi-turn prompts to keep the question single-shot.
        if len(ex["prompt"]) != 1 or ex["prompt"][0]["role"] != "user":
            continue
        question = ex["prompt"][0]["content"]
        # Cap question length so the judge prompt stays inside reasonable
        # token budgets (judges with small context windows otherwise truncate).
        if len(question) > 2500:
            continue
        if len(ideal) > 4000:
            # HealthBench ideals can be 5K+ chars; cap to keep judge calls cheap.
            ideal = ideal[:4000].rstrip() + " [...]"
        # Each rubric criterion becomes a "passage" so the judge sees it
        # as part of the evidence the answer must meet.
        passages = [
            f"[{r['points']:+d} pts] {r['criterion']}"
            for r in rubrics
            if r.get("criterion")
        ]
        if not passages:
            continue
        task = QATask(
            task_id=f"hb-{ex['prompt_id'][:12]}",
            pubid=ex["prompt_id"],
            question=question,
            context_passages=passages[:8],  # cap rubric count for token budget
            long_answer=ideal,
            final_decision="yes",
        )
        tasks.append(task.__dict__)

    out_path = out_dir / "sample.json"
    out_path.write_text(json.dumps(tasks, indent=2))
    index = {
        "version": "0.1.0",
        "split": "healthbench-eval",
        "task_ids": [t["task_id"] for t in tasks],
        "license": "OpenAI HealthBench (per upstream)",
        "release_date": "2026-05-09",
        "description": (
            f"Random sample of {len(tasks)} HealthBench OSS-eval examples "
            f"(seed={seed}). Each task carries the user's clinical query, "
            "the physician-written ideal completion, and the rubric criteria "
            "as supporting passages. Long-form medical advice answers with "
            "frequent dosing / numeric claims - the right home for "
            "numeric_swap and drug-related probes."
        ),
    }
    (out_dir / "index.json").write_text(json.dumps(index, indent=2))
    print(f"Wrote {len(tasks)} tasks to {out_path}")
    # Diagnostics: probe-applicability counts
    import re
    numeric = sum(1 for t in tasks if re.search(r"\b\d{1,3}(\.\d+)?\s?(%|mg|mcg|g|mL|ml|µg|kg|hours?|days?|weeks?)\b", t["long_answer"], re.IGNORECASE))
    negation = sum(
        1 for t in tasks
        if re.search(r"\b(not|no|none|never|without|cannot|don't|do not)\b", t["long_answer"], re.IGNORECASE)
    )
    print(f"  numeric/dose-bearing answers: {numeric}/{len(tasks)}")
    print(f"  negation-bearing answers: {negation}/{len(tasks)}")


if __name__ == "__main__":
    main()
