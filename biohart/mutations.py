"""Adversarial mutation families for biomedical-QA judge probes.

For each (question, context, answer) tuple we produce variants. Negative-control
families should NOT change a competent judge's verdict. Positive-control families
change a load-bearing part of the answer, so a reading judge SHOULD flip.

The valid positive control is `single_word_negation_drop`: removing a load-bearing
negation genuinely inverts the claim. It is the headline probe.

CAVEAT on `reversed_answer`: it only PREPENDS a "Contrary to..." preamble and
flips the label field; it does NOT alter the answer's substance, so a competent
judge correctly holds its verdict and the harness scores that as a "miss." It is
therefore NOT a valid positive control, is EXCLUDED from the headline metric, and
is retained only as a weak surface-sensitivity signal. See KNOWN_LIMITATIONS.md.
Do not report its flip rate as a "reversal catch rate." A genuine-reversal
replacement (LLM-generated opposite conclusion) is on the roadmap.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass


@dataclass
class QATask:
    task_id: str
    question: str
    context_passages: list[str]
    long_answer: str
    final_decision: str  # "yes" | "no" | "maybe"
    pubid: str


@dataclass
class QAVariant:
    variant_id: str
    family: str
    question: str
    context_passages: list[str]
    long_answer: str
    final_decision: str
    notes: str
    expected_judge_flip: bool  # True = positive control; False = negative


# --- Negative-control families -----------------------------------------------

# Fillers chosen to read like generic scientific prose; carry no semantic
# weight on the question or answer. Inserting them MUST NOT change a
# competent judge's verdict.
FILLER_NOTES = [
    " Further work in larger cohorts is warranted.",
    " The authors thank the donors and clinical staff.",
    " Funding statement: this study received no specific grant.",
    " Conflicts of interest: the authors declare none.",
    " Data availability: raw counts are available on request.",
    " A full bibliography is provided in the supplementary materials.",
]


def comment_only_filler(task: QATask, seed: int) -> QAVariant:
    """Append a non-substantive scientific filler note to the long answer."""
    rng = random.Random(seed)
    note = rng.choice(FILLER_NOTES)
    new_answer = task.long_answer.rstrip() + note
    return QAVariant(
        variant_id=f"{task.task_id}-comment_only_filler",
        family="comment_only_filler",
        question=task.question,
        context_passages=task.context_passages,
        long_answer=new_answer,
        final_decision=task.final_decision,
        notes="Filler sentence appended; no factual change.",
        expected_judge_flip=False,
    )


SYNONYMS = [
    ("patients", "subjects"),
    ("treatment", "therapy"),
    ("results", "findings"),
    ("study", "investigation"),
    ("outcomes", "endpoints"),
    ("group", "cohort"),
    ("higher", "elevated"),
    ("lower", "decreased"),
    ("effect", "impact"),
    ("significant", "statistically meaningful"),
]


PARAPHRASE_TARGET_N = 2


def paraphrase_synonym(task: QATask, seed: int) -> QAVariant:
    """Replace exactly PARAPHRASE_TARGET_N domain-neutral words with synonyms.

    If fewer than that number of substitutable tokens exist in the answer,
    the variant falls back to no-op and is marked applicable=False (flag
    surfaced via the variant_id metadata for downstream filtering).
    """
    rng = random.Random(seed)
    text = task.long_answer
    pairs = list(SYNONYMS)
    rng.shuffle(pairs)
    n_replaced = 0
    used: list[tuple[str, str]] = []
    for src, tgt in pairs:
        if n_replaced >= PARAPHRASE_TARGET_N:
            break
        pattern = re.compile(rf"\b{re.escape(src)}\b", re.IGNORECASE)
        if pattern.search(text):
            text = pattern.sub(tgt, text, count=1)
            n_replaced += 1
            used.append((src, tgt))
    if n_replaced < PARAPHRASE_TARGET_N:
        # Don't fabricate edits to hit the count; mark as inapplicable so the
        # downstream aggregator excludes it from negative-control denominators.
        return QAVariant(
            variant_id=f"{task.task_id}-paraphrase_synonym__inapplicable",
            family="paraphrase_synonym",
            question=task.question,
            context_passages=task.context_passages,
            long_answer=task.long_answer,
            final_decision=task.final_decision,
            notes=(
                f"Only {n_replaced}/{PARAPHRASE_TARGET_N} synonym substitutions "
                "available; family inapplicable for this task."
            ),
            expected_judge_flip=False,
        )
    return QAVariant(
        variant_id=f"{task.task_id}-paraphrase_synonym",
        family="paraphrase_synonym",
        question=task.question,
        context_passages=task.context_passages,
        long_answer=text,
        final_decision=task.final_decision,
        notes=f"Substitutions: {used}; no factual change.",
        expected_judge_flip=False,
    )


def irrelevant_citation(task: QATask, seed: int) -> QAVariant:
    """Append a fake-but-irrelevant citation reference."""
    rng = random.Random(seed)
    fake_pmid = rng.randint(10_000_000, 39_999_999)
    suffix = f" (See also PMID: {fake_pmid}; unrelated background.)"
    new_answer = task.long_answer.rstrip() + suffix
    return QAVariant(
        variant_id=f"{task.task_id}-irrelevant_citation",
        family="irrelevant_citation",
        question=task.question,
        context_passages=task.context_passages,
        long_answer=new_answer,
        final_decision=task.final_decision,
        notes="Unrelated PMID reference appended; no factual change.",
        expected_judge_flip=False,
    )


def shuffle_passage_order(task: QATask, seed: int) -> QAVariant:
    """Shuffle the order of context passages.

    LOSSLESS: the same evidence is present, only the ordering changes.
    A judge that scores on retrieved-evidence content (not on positional
    artefacts) must hold its verdict. A judge that pattern-matches on
    "intro paragraph first" will be revealed if it flips.
    """
    if len(task.context_passages) <= 1:
        return QAVariant(
            variant_id=f"{task.task_id}-shuffle_passage_order__inapplicable",
            family="shuffle_passage_order",
            question=task.question,
            context_passages=task.context_passages,
            long_answer=task.long_answer,
            final_decision=task.final_decision,
            notes="Single-passage context, shuffle inapplicable.",
            expected_judge_flip=False,
        )
    rng = random.Random(seed)
    new_order = list(task.context_passages)
    while True:
        rng.shuffle(new_order)
        if new_order != task.context_passages:
            break
    return QAVariant(
        variant_id=f"{task.task_id}-shuffle_passage_order",
        family="shuffle_passage_order",
        question=task.question,
        context_passages=new_order,
        long_answer=task.long_answer,
        final_decision=task.final_decision,
        notes="Context passages reordered; same evidence, different sequence.",
        expected_judge_flip=False,
    )


# --- Positive-control family -------------------------------------------------

NEGATION_PREFIX = "Contrary to what an honest reading would conclude: "


def reversed_answer(task: QATask, seed: int) -> QAVariant:
    """Prepend a "Contrary to..." preamble and flip the label field.

    NOT a valid semantic reversal: the answer's substance is unchanged, so a
    competent judge correctly holds its verdict. Excluded from the headline
    metric; retained as a weak surface-sensitivity signal only. See
    KNOWN_LIMITATIONS.md. `expected_judge_flip=True` is kept for backward
    compatibility with existing result JSONs but should not be read as a valid
    positive-control expectation.
    """
    flipped = {"yes": "no", "no": "yes", "maybe": "no"}.get(
        task.final_decision, task.final_decision
    )
    new_answer = NEGATION_PREFIX + task.long_answer
    return QAVariant(
        variant_id=f"{task.task_id}-reversed_answer",
        family="reversed_answer",
        question=task.question,
        context_passages=task.context_passages,
        long_answer=new_answer,
        final_decision=flipped,
        notes=(
            "Answer prefixed with explicit negation; final_decision flipped. "
            "A competent judge should now score this as INCORRECT."
        ),
        expected_judge_flip=True,
    )


# --- Public entry point ------------------------------------------------------

# --- Subtler mutation families (harder probes) -------------------------------

# A "single-word negation" probe: drop the lone qualifier "not" / "no" / "without"
# from the answer if present. Original answer remains identical except for
# that one word. Subtler than `reversed_answer`, no overt "Contrary to"
# preamble. A judge that pattern-matches surface tone will hold its verdict.

_NEGATION_TOKENS = (
    r"\bnot\b",
    r"\bno\b",
    r"\bnone\b",
    r"\bnever\b",
    r"\bwithout\b",
    r"\bcannot\b",
)


def single_word_negation_drop(task: QATask, seed: int) -> QAVariant:
    text = task.long_answer
    matches = []
    for pat in _NEGATION_TOKENS:
        for m in re.finditer(pat, text, flags=re.IGNORECASE):
            matches.append((m.start(), m.end(), m.group(0)))
    if not matches:
        # No negation token found; degrade to no-op so this family is just skipped.
        return QAVariant(
            variant_id=f"{task.task_id}-single_word_negation_drop",
            family="single_word_negation_drop",
            question=task.question,
            context_passages=task.context_passages,
            long_answer=text,
            final_decision=task.final_decision,
            notes="No negation token in original answer; family inapplicable.",
            expected_judge_flip=False,  # cannot test; treat as negative control
        )
    rng = random.Random(seed)
    s, e, tok = rng.choice(matches)
    new_answer = text[:s] + text[e:]
    new_answer = re.sub(r"\s{2,}", " ", new_answer).strip()
    flipped = {"yes": "no", "no": "yes", "maybe": "no"}.get(
        task.final_decision, task.final_decision
    )
    return QAVariant(
        variant_id=f"{task.task_id}-single_word_negation_drop",
        family="single_word_negation_drop",
        question=task.question,
        context_passages=task.context_passages,
        long_answer=new_answer,
        final_decision=flipped,
        notes=(
            f"Dropped single negation token '{tok}' from the answer; final_decision "
            "flipped. Subtler than reversed_answer because there is no preamble, "
            "the change is one word. A judge that reads the answer should still "
            "score this as INCORRECT."
        ),
        expected_judge_flip=True,
    )


def numeric_swap(task: QATask, seed: int) -> QAVariant:
    """Swap a numeric value (percent / count / p-value) for a wildly different one.

    Tests whether the judge anchors on numeric claims or just on prose tone.
    """
    text = task.long_answer
    # Match percentages and decimal numbers; prefer percentages because they
    # are more obviously claim-bearing.
    pct = list(re.finditer(r"\b(\d{1,3})(?:\.\d+)?\s?%", text))
    rng = random.Random(seed)
    target = None
    if pct:
        target = rng.choice(pct)
    else:
        nums = list(re.finditer(r"\b(\d{2,5})(?:\.\d+)?\b", text))
        if nums:
            target = rng.choice(nums)
    if target is None:
        return QAVariant(
            variant_id=f"{task.task_id}-numeric_swap",
            family="numeric_swap",
            question=task.question,
            context_passages=task.context_passages,
            long_answer=text,
            final_decision=task.final_decision,
            notes="No numeric value in answer; family inapplicable.",
            expected_judge_flip=False,
        )
    orig = target.group(0)
    # Pick a wildly different number in the same shape.
    if orig.endswith("%"):
        pct_val = int(rng.choice([3, 7, 12, 23, 41, 57, 71, 88, 94]))
        replacement = f"{pct_val}%"
    else:
        replacement = str(rng.randint(2, 9999))
    new_answer = text[: target.start()] + replacement + text[target.end():]
    flipped = {"yes": "no", "no": "yes", "maybe": "no"}.get(
        task.final_decision, task.final_decision
    )
    return QAVariant(
        variant_id=f"{task.task_id}-numeric_swap",
        family="numeric_swap",
        question=task.question,
        context_passages=task.context_passages,
        long_answer=new_answer,
        final_decision=flipped,
        notes=(
            f"Replaced numeric '{orig}' with '{replacement}' in the answer. "
            "If the original number was load-bearing for the conclusion, the "
            "answer is now factually unsupported by the context. A judge that "
            "actually reads should flag this as INCORRECT."
        ),
        expected_judge_flip=True,
    )


ALL_FAMILIES = [
    comment_only_filler,
    paraphrase_synonym,
    irrelevant_citation,
    shuffle_passage_order,
    reversed_answer,
    single_word_negation_drop,
    numeric_swap,
]


def make_variants(task: QATask, seed: int = 42) -> list[QAVariant]:
    return [fn(task, seed=seed + i) for i, fn in enumerate(ALL_FAMILIES)]
