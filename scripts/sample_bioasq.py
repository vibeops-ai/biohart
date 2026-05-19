"""Pull a random BioASQ sample.

kroshan/BioASQ has 3266 examples. Each row is `question` plus a `text`
field of the form `<answer> X <context> Y`. We split on those tags and
build the same QATask schema as PubMedQA so the same mutations apply.

Output: biohart/datasets/bioasq/sample.json
"""

from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datasets import load_dataset

from biohart.mutations import QATask


def _split(text: str) -> tuple[str, list[str]] | None:
    """Return (answer, context_passages) or None if format unexpected."""
    m = re.match(r"<answer>\s*(.+?)\s*<context>\s*(.*)$", text, flags=re.DOTALL)
    if not m:
        return None
    answer = m.group(1).strip()
    ctx = m.group(2).strip()
    if not answer or not ctx:
        return None
    # Split context on blank lines or sentence-ish boundaries; cap passage count.
    parts = [p.strip() for p in re.split(r"\n\s*\n|(?<=\.)\s{2,}", ctx) if p.strip()]
    if not parts:
        parts = [ctx]
    return answer, parts[:6]


def _decision_for_factoid(answer: str) -> str:
    """BioASQ factoid answers are short noun phrases; we wrap them in the
    yes/no/maybe schema PubMedQA uses by treating them as 'yes' (i.e.
    the candidate answer is the correct factoid). The judge's job is to
    decide whether the candidate matches what the passages support."""
    return "yes"


def main() -> None:
    n = 30
    seed = 42
    out_dir = Path("biohart/datasets/bioasq")
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset("kroshan/BioASQ", split="train")
    rng = random.Random(seed)

    candidates = list(range(len(ds)))
    rng.shuffle(candidates)

    tasks: list[dict] = []
    for i in candidates:
        if len(tasks) >= n:
            break
        ex = ds[i]
        parsed = _split(ex.get("text", ""))
        if parsed is None:
            continue
        answer, passages = parsed
        # Skip very long or empty answers.
        if not (4 <= len(answer) <= 400):
            continue
        # Numeric / short answers preferred, biases us towards factoid items
        # which is exactly what numeric_swap and single_word_negation_drop
        # target.
        task = QATask(
            task_id=f"bioasq-{i}",
            pubid=str(i),
            question=ex["question"],
            context_passages=passages,
            long_answer=answer,
            final_decision=_decision_for_factoid(answer),
        )
        tasks.append(task.__dict__)

    out_path = out_dir / "sample.json"
    out_path.write_text(json.dumps(tasks, indent=2))
    index = {
        "version": "0.1.0",
        "split": "bioasq-factoid",
        "task_ids": [t["task_id"] for t in tasks],
        "license": "BioASQ task-B (per upstream)",
        "release_date": "2026-05-09",
        "description": (
            f"Random sample of {len(tasks)} BioASQ-task-B QA examples (seed={seed}, "
            "source kroshan/BioASQ on HuggingFace). Short-form factoid answers, "
            "many with numeric values - the natural home for the numeric_swap "
            "and single_word_negation_drop probe families."
        ),
    }
    (out_dir / "index.json").write_text(json.dumps(index, indent=2))
    print(f"Wrote {len(tasks)} tasks to {out_path}")
    # Diagnostic: how many answers carry a number?
    numeric_count = sum(1 for t in tasks if re.search(r"\d", t["long_answer"]))
    negation_count = sum(
        1 for t in tasks
        if re.search(r"\b(not|no|none|never|without|cannot)\b", t["long_answer"], re.IGNORECASE)
    )
    print(f"  numeric-bearing answers: {numeric_count}/{len(tasks)}")
    print(f"  negation-bearing answers: {negation_count}/{len(tasks)}")


if __name__ == "__main__":
    main()
