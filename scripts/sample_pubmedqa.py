"""Pull a random sample of n PubMedQA-labeled tasks and write them to disk.

Output: biohart/datasets/pubmedqa/sample.json
        biohart/datasets/pubmedqa/index.json
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
    out_dir = Path("biohart/datasets/pubmedqa")
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
    rng = random.Random(seed)
    indices = rng.sample(range(len(ds)), n)

    tasks: list[dict] = []
    for i in indices:
        ex = ds[i]
        ctx = ex["context"]
        passages = ctx.get("contexts", [])
        if not passages:
            continue
        task = QATask(
            task_id=f"pmqa-{ex['pubid']}",
            pubid=str(ex["pubid"]),
            question=ex["question"],
            context_passages=passages,
            long_answer=ex["long_answer"],
            final_decision=ex["final_decision"],
        )
        tasks.append(task.__dict__)

    out_path = out_dir / "sample.json"
    out_path.write_text(json.dumps(tasks, indent=2))
    index = {
        "version": "0.1.0",
        "split": "pubmedqa-labeled",
        "task_ids": [t["task_id"] for t in tasks],
        "license": "MIT (Jin et al., 2019)",
        "release_date": "2026-05-09",
        "description": (
            f"Random sample of {len(tasks)} PubMedQA pqa_labeled examples (seed={seed}). "
            "Each task carries question / context_passages / long_answer / "
            "final_decision / pubid. Used as the source distribution for biohart adversarial probes."
        ),
    }
    (out_dir / "index.json").write_text(json.dumps(index, indent=2))
    print(f"Wrote {len(tasks)} tasks to {out_path}")


if __name__ == "__main__":
    main()
