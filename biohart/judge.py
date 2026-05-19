"""Run a judge over biohart probes.

A probe is `(question, context_passages, candidate_answer, claimed_label)`. The
judge model is asked to return a JSON verdict `{"verdict": "correct" | "incorrect",
"confidence": 0.0-1.0}`. Providers supported: anthropic, openai, openrouter, google,
azure.

Each `call_model` returns `(raw_text, cost_usd, latency_ms)`. Costs are estimated
from public per-token pricing approximations; treat them as ballpark, not invoice.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Optional

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = (
    "You are a careful biomedical-evidence reviewer. Given a question, "
    "supporting passages, and a candidate long-form answer with a yes/no/maybe "
    "label, decide whether the candidate answer is supported and correct based "
    "on the passages. Return ONLY a JSON object, no preamble, no fences."
)

VERDICT_PROMPT = """\
QUESTION:
{question}

SUPPORTING PASSAGES:
{passages}

CANDIDATE ANSWER (label = {label}):
{answer}

Return strictly:
{{"verdict": "correct" | "incorrect", "confidence": 0.0-1.0, "reasoning": "<= 200 chars"}}
"""


# Per-million-token pricing approximations. (input_per_M, output_per_M) in USD.
# Update as published vendor pricing changes; treat as ballpark.
_PRICING: dict[str, tuple[float, float]] = {
    # Anthropic
    "claude-opus-4-7": (15.0, 75.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-haiku-4-5-20251001": (0.80, 4.0),
    # OpenAI
    "gpt-4.1": (2.0, 8.0),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4o": (2.50, 10.0),
    "gpt-4o-mini": (0.15, 0.60),
    # Google
    "gemini-3.1-pro-preview": (6.0, 18.0),
    "google/gemini-3.1-pro-preview": (6.0, 18.0),
    # Cohere
    "cohere/command-a": (2.50, 10.0),
    # Mistral
    "mistralai/mistral-medium-3-5": (1.0, 3.0),
    # OpenRouter pass-throughs (best-effort; OpenRouter returns actual cost in their API too)
    "meta-llama/llama-4-maverick": (0.30, 0.90),
    "x-ai/grok-4.3": (4.0, 12.0),
}


def _estimate_cost(model: str, in_tokens: int, out_tokens: int) -> float:
    in_p, out_p = _PRICING.get(model, (1.0, 4.0))
    return (in_tokens * in_p + out_tokens * out_p) / 1_000_000


# --- Provider implementations ----------------------------------------------

_ANTHROPIC_CLIENT = None
_OPENAI_CLIENT = None
_OPENROUTER_CLIENT = None
_AZURE_CLIENT = None


async def _anthropic_call(model: str, prompt: str) -> tuple[str, float, int]:
    global _ANTHROPIC_CLIENT
    if _ANTHROPIC_CLIENT is None:
        from anthropic import AsyncAnthropic
        _ANTHROPIC_CLIENT = AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    start = time.monotonic()
    try:
        resp = await _ANTHROPIC_CLIENT.messages.create(
            model=model,
            max_tokens=2048,
            temperature=0.1,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        latency = int((time.monotonic() - start) * 1000)
        in_t = resp.usage.input_tokens
        out_t = resp.usage.output_tokens
        text = "".join(b.text for b in resp.content if hasattr(b, "text"))
        return text, _estimate_cost(model, in_t, out_t), latency
    except Exception as e:
        logger.error("anthropic call failed (%s): %s", model, e)
        return "", 0.0, int((time.monotonic() - start) * 1000)


async def _openai_call(model: str, prompt: str) -> tuple[str, float, int]:
    global _OPENAI_CLIENT
    if _OPENAI_CLIENT is None:
        from openai import AsyncOpenAI
        _OPENAI_CLIENT = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
    start = time.monotonic()
    try:
        resp = await _OPENAI_CLIENT.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_tokens=2048,
        )
        latency = int((time.monotonic() - start) * 1000)
        in_t = resp.usage.prompt_tokens if resp.usage else 0
        out_t = resp.usage.completion_tokens if resp.usage else 0
        text = resp.choices[0].message.content or ""
        return text, _estimate_cost(model, in_t, out_t), latency
    except Exception as e:
        logger.error("openai call failed (%s): %s", model, e)
        return "", 0.0, int((time.monotonic() - start) * 1000)


async def _openrouter_call(model: str, prompt: str) -> tuple[str, float, int]:
    """OpenRouter exposes an OpenAI-compatible API at openrouter.ai/api/v1."""
    global _OPENROUTER_CLIENT
    if _OPENROUTER_CLIENT is None:
        from openai import AsyncOpenAI
        _OPENROUTER_CLIENT = AsyncOpenAI(
            api_key=os.environ["OPENROUTER_API_KEY"],
            base_url="https://openrouter.ai/api/v1",
        )
    start = time.monotonic()
    try:
        resp = await _OPENROUTER_CLIENT.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_tokens=2048,
        )
        latency = int((time.monotonic() - start) * 1000)
        in_t = resp.usage.prompt_tokens if resp.usage else 0
        out_t = resp.usage.completion_tokens if resp.usage else 0
        text = resp.choices[0].message.content or ""
        return text, _estimate_cost(model, in_t, out_t), latency
    except Exception as e:
        logger.error("openrouter call failed (%s): %s", model, e)
        return "", 0.0, int((time.monotonic() - start) * 1000)


async def _google_call(model: str, prompt: str) -> tuple[str, float, int]:
    import google.generativeai as genai
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
    start = time.monotonic()
    try:
        client_model = genai.GenerativeModel(
            model_name=model.replace("google/", ""),
            system_instruction=SYSTEM_PROMPT,
        )
        resp = await client_model.generate_content_async(
            prompt,
            generation_config={"temperature": 0.1, "max_output_tokens": 2048},
        )
        latency = int((time.monotonic() - start) * 1000)
        in_t = getattr(resp.usage_metadata, "prompt_token_count", 0)
        out_t = getattr(resp.usage_metadata, "candidates_token_count", 0)
        text = resp.text if hasattr(resp, "text") else ""
        return text, _estimate_cost(model, in_t, out_t), latency
    except Exception as e:
        logger.error("google call failed (%s): %s", model, e)
        return "", 0.0, int((time.monotonic() - start) * 1000)


async def _azure_call(deployment: str, prompt: str) -> tuple[str, float, int]:
    global _AZURE_CLIENT
    if _AZURE_CLIENT is None:
        from openai import AsyncAzureOpenAI
        _AZURE_CLIENT = AsyncAzureOpenAI(
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2025-01-01-preview"),
        )
    start = time.monotonic()
    try:
        resp = await _AZURE_CLIENT.chat.completions.create(
            model=deployment,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_tokens=2048,
        )
        latency = int((time.monotonic() - start) * 1000)
        in_t = resp.usage.prompt_tokens if resp.usage else 0
        out_t = resp.usage.completion_tokens if resp.usage else 0
        text = resp.choices[0].message.content or ""
        return text, _estimate_cost(deployment, in_t, out_t), latency
    except Exception as e:
        logger.error("azure call failed (%s): %s", deployment, e)
        return "", 0.0, int((time.monotonic() - start) * 1000)


async def call_model(provider: str, model: str, prompt: str) -> tuple[str, float, int]:
    """Dispatch to a provider-specific async call. Returns (text, cost_usd, latency_ms)."""
    if provider == "anthropic":
        return await _anthropic_call(model, prompt)
    if provider == "openai":
        return await _openai_call(model, prompt)
    if provider == "openrouter":
        return await _openrouter_call(model, prompt)
    if provider == "google":
        return await _google_call(model, prompt)
    if provider == "azure":
        return await _azure_call(model, prompt)
    raise ValueError(f"unknown provider: {provider!r}")


# --- Verdict parsing -------------------------------------------------------

def _format_passages(passages: list[str]) -> str:
    return "\n\n".join(f"[{i + 1}] {p}" for i, p in enumerate(passages))


def _parse_verdict(raw: str) -> Optional[dict]:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```", 2)[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.rsplit("```", 1)[0].strip()
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{.*?\}", raw, re.DOTALL)
        if not m:
            return None
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
    if not isinstance(obj, dict) or "verdict" not in obj:
        return None
    v = str(obj.get("verdict", "")).strip().lower()
    if v not in ("correct", "incorrect"):
        return None
    obj["verdict"] = v
    obj["confidence"] = float(obj.get("confidence", 0.0))
    return obj


async def judge_one(
    provider: str,
    model: str,
    question: str,
    passages: list[str],
    answer: str,
    label: str,
) -> dict:
    prompt = VERDICT_PROMPT.format(
        question=question.strip(),
        passages=_format_passages(passages),
        answer=answer.strip(),
        label=label,
    )
    raw, cost, latency_ms = await call_model(provider, model, prompt)
    parsed = _parse_verdict(raw)
    return {
        "raw": raw,
        "verdict": parsed["verdict"] if parsed else None,
        "confidence": parsed["confidence"] if parsed else None,
        "cost_usd": cost,
        "latency_ms": latency_ms,
    }
