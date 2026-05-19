"""Run one judge across all PubMedQA tasks (original + 5 mutation variants each).

Output: leaderboard/results/<model_slug>__<timestamp>.json with per-probe
verdicts and per-family hack rates.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import time
from collections import defaultdict
from pathlib import Path

# Load credentials from a local .env if present (optional convenience).
try:
    from dotenv import load_dotenv as _ld
    _candidate = Path.cwd() / ".env"
    if _candidate.exists():
        _ld(_candidate)
except Exception:
    pass

from biohart.judge import judge_one
from biohart.mutations import QATask, make_variants

logger = logging.getLogger(__name__)


def load_tasks(path: Path) -> list[QATask]:
    raw = json.loads(path.read_text())
    return [QATask(**r) for r in raw]


async def run_one_task(provider: str, model: str, task: QATask, sem: asyncio.Semaphore) -> dict:
    """Probe original + each variant; record verdicts and flips."""
    async def _one(question: str, passages: list[str], answer: str, label: str) -> dict:
        async with sem:
            return await judge_one(provider, model, question, passages, answer, label)

    # Baseline: judge the unmodified (question, context, answer, decision).
    baseline = await _one(
        task.question, task.context_passages, task.long_answer, task.final_decision
    )

    variants = make_variants(task)
    coros = [
        _one(v.question, v.context_passages, v.long_answer, v.final_decision)
        for v in variants
    ]
    variant_results = await asyncio.gather(*coros)

    # Per-variant flip = baseline.verdict differs from variant.verdict.
    per_variant = []
    for v, r in zip(variants, variant_results):
        flipped = (
            baseline["verdict"] is not None
            and r["verdict"] is not None
            and baseline["verdict"] != r["verdict"]
        )
        # Applicability: a variant whose answer + context are identical to
        # the baseline carries no signal (we asked the model the same thing
        # twice). Mark + exclude these from per-family denominators.
        applicable = (
            v.long_answer != task.long_answer
            or v.context_passages != task.context_passages
        )
        per_variant.append({
            "variant_id": v.variant_id,
            "family": v.family,
            "expected_judge_flip": v.expected_judge_flip,
            "applicable": applicable,
            "baseline_verdict": baseline["verdict"],
            "variant_verdict": r["verdict"],
            "flipped": flipped,
            "matches_expectation": flipped == v.expected_judge_flip,
            "cost_usd": r["cost_usd"],
            "latency_ms": r["latency_ms"],
        })

    return {
        "task_id": task.task_id,
        "baseline_verdict": baseline["verdict"],
        "baseline_confidence": baseline["confidence"],
        "baseline_cost_usd": baseline["cost_usd"],
        "variants": per_variant,
    }


async def run_all(provider: str, model: str, tasks: list[QATask], concurrency: int) -> dict:
    sem = asyncio.Semaphore(concurrency)
    coros = [run_one_task(provider, model, t, sem) for t in tasks]
    per_task = await asyncio.gather(*coros)

    # Aggregate per family. We track total emitted vs applicable separately
    # because single_word_negation_drop / numeric_swap silently fall back to
    # no-op variants when the source answer lacks a negation token / numeric
    # value. A no-op variant has expected_judge_flip=False AND identical
    # text, not informative for hack-rate.
    per_family: dict[str, dict[str, int]] = defaultdict(
        lambda: {"total": 0, "applicable": 0, "flipped": 0,
                 "applicable_flipped": 0, "matches_expectation": 0}
    )
    # Build a set of (task_id, family) that are no-ops to filter them out.
    # We detect by re-running the mutation function and comparing diff_text.
    # Simpler: use the applicability flag we already attach via the family
    # function (expected_judge_flip is True for positive-controls only when
    # the mutation actually fires, False otherwise).
    total_cost = 0.0
    for r in per_task:
        total_cost += r["baseline_cost_usd"]
        for v in r["variants"]:
            total_cost += v["cost_usd"]
            f = per_family[v["family"]]
            f["total"] += 1
            if v["applicable"]:
                f["applicable"] += 1
                if v["flipped"]:
                    f["applicable_flipped"] += 1
            if v["flipped"]:
                f["flipped"] += 1
            if v["matches_expectation"]:
                f["matches_expectation"] += 1

    summary = {
        "provider": provider,
        "model": model,
        "n_tasks": len(tasks),
        "total_cost_usd": total_cost,
        "per_family": {
            fam: {
                "total": c["total"],
                "applicable": c["applicable"],
                "flip_rate_total": c["flipped"] / c["total"] if c["total"] else 0.0,
                "flip_rate": (
                    c["applicable_flipped"] / c["applicable"] if c["applicable"] else 0.0
                ),
                "expectation_match_rate": (
                    c["matches_expectation"] / c["total"] if c["total"] else 0.0
                ),
            }
            for fam, c in per_family.items()
        },
    }

    return {"summary": summary, "results": per_task}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--provider", required=True, choices=["anthropic", "openai", "openrouter", "google", "azure"])
    p.add_argument("--model", required=True)
    p.add_argument("--tasks", default="biohart/datasets/pubmedqa/sample.json")
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--output-dir", default="leaderboard/results")
    args = p.parse_args()

    tasks = load_tasks(Path(args.tasks))
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = asyncio.run(run_all(args.provider, args.model, tasks, args.concurrency))

    slug = args.model.replace("/", "__")
    dataset = Path(args.tasks).parent.name or "tasks"
    out_path = out_dir / f"{slug}__{dataset}__{int(time.time())}.json"
    out_path.write_text(json.dumps(payload, indent=2))
    s = payload["summary"]
    logger.info("model=%s n=%d cost=$%.2f", s["model"], s["n_tasks"], s["total_cost_usd"])
    for fam, stats in s["per_family"].items():
        logger.info("  %-30s flip=%.2f match-expectation=%.2f",
                    fam, stats["flip_rate"], stats["expectation_match_rate"])


if __name__ == "__main__":
    main()
