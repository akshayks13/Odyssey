"""LLM factory and ReAct helper (Gemini, then Groq)."""
from __future__ import annotations

import functools
import json
import logging
import re
from typing import Any

from config import (
    GEMINI_API_KEY,
    GEMINI_MODEL,
    GROQ_API_KEY,
    GROQ_MODEL,
    LLM_DISABLED,
)

logger = logging.getLogger("odyssey.llm")

_gemini_quota_hit = False


def _quota_error(exc: BaseException) -> bool:
    blob = f"{type(exc).__name__} {exc}".lower()
    return any(
        token in blob
        for token in ("429", "resourceexhausted", "quota", "rate-limit", "rate_limit", "ratelimit")
    )


def _skip_gemini() -> bool:
    return _gemini_quota_hit


def _mark_gemini_quota(exc: BaseException | None = None) -> None:
    global _gemini_quota_hit
    if _gemini_quota_hit:
        return
    _gemini_quota_hit = True
    extra = f" ({exc})" if exc else ""
    logger.warning("Gemini quota/rate-limit hit%s — switching to Groq for this process.", extra)


def llm_provider_name() -> str | None:
    if LLM_DISABLED:
        return None
    if GEMINI_API_KEY and not _skip_gemini():
        return f"gemini:{GEMINI_MODEL}"
    if GROQ_API_KEY:
        return f"groq:{GROQ_MODEL}"
    return None


def _install_gemini_fail_fast() -> None:
    """Stop langchain_google_genai from sleeping on 429 before we can use Groq."""
    try:
        from tenacity import retry, stop_after_attempt

        import langchain_google_genai.chat_models as gemini_chat

        def _no_retry_decorator():
            return retry(reraise=True, stop=stop_after_attempt(1))

        gemini_chat._create_retry_decorator = _no_retry_decorator  # type: ignore[assignment]
    except Exception:
        pass


@functools.lru_cache(maxsize=1)
def _gemini_llm():
    if LLM_DISABLED or not GEMINI_API_KEY:
        return None
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI

        _install_gemini_fail_fast()
        return ChatGoogleGenerativeAI(
            model=GEMINI_MODEL,
            google_api_key=GEMINI_API_KEY,
            temperature=0.2,
            max_retries=0,
        )
    except Exception:
        return None


@functools.lru_cache(maxsize=1)
def _groq_llm():
    if LLM_DISABLED or not GROQ_API_KEY:
        return None
    try:
        from langchain_groq import ChatGroq

        return ChatGroq(model=GROQ_MODEL, api_key=GROQ_API_KEY, temperature=0.2)
    except Exception:
        return None


def get_llm():
    if LLM_DISABLED:
        return None
    if not _skip_gemini():
        gemini = _gemini_llm()
        if gemini is not None:
            return gemini
    return _groq_llm()


def parse_json_blob(text: str) -> dict:
    """Extract a JSON object from a model reply (raw or fenced)."""
    if not text:
        return {}
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1)
    else:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            text = text[start : end + 1]
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        return {}


def _react_loop(llm, tools: list, system: str, user: str, max_rounds: int) -> dict:
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

    by_name = {t.name: t for t in tools}
    model = llm.bind_tools(tools)
    messages: list = [SystemMessage(content=system), HumanMessage(content=user)]
    recorded: list[dict] = []

    for _round_idx in range(max_rounds):
        try:
            ai: AIMessage = model.invoke(messages)
        except Exception as exc:
            if _quota_error(exc):
                _mark_gemini_quota(exc)
            raise
        messages.append(ai)
        tool_calls = getattr(ai, "tool_calls", None) or []
        if not tool_calls:
            text = (ai.content or "") if isinstance(ai.content, str) else str(ai.content or "")
            decision = parse_json_blob(text)
            decision["_tool_calls"] = [c["name"] for c in recorded]
            decision["_reasoning_text"] = text
            return decision
        for call in tool_calls:
            name = call.get("name") if isinstance(call, dict) else getattr(call, "name", "")
            args = call.get("args") if isinstance(call, dict) else getattr(call, "args", {}) or {}
            call_id = call.get("id") if isinstance(call, dict) else getattr(call, "id", name)
            tool = by_name.get(name)
            if tool is None:
                result: Any = {"error": f"unknown tool {name}"}
            else:
                try:
                    result = tool.invoke(args)
                except Exception as exc:  # noqa: BLE001
                    result = {"error": str(exc)}
            recorded.append({"name": name, "args": args, "result": result})
            messages.append(
                ToolMessage(
                    content=json.dumps(result, default=str)[:8000],
                    tool_call_id=call_id or name,
                    name=name,
                )
            )
    return {}


def llm_json(system: str, user: str) -> dict:
    """JSON-only call: Gemini first, Groq if Gemini is skipped or fails."""
    return llm_decide(get_llm(), tools=[], system=system, user=user, max_rounds=1)


def llm_decide(llm, tools: list, system: str, user: str, max_rounds: int = 6) -> dict:
    """bind_tools ReAct loop, then parse a JSON decision. Empty dict on failure.

    Gemini first; on 429/quota, Groq for this call and the rest of the process.
    """
    groq = _groq_llm()
    candidates = []
    if not _skip_gemini() and llm is not None:
        candidates.append(llm)
    elif not _skip_gemini() and _gemini_llm() is not None:
        candidates.append(_gemini_llm())
    if groq is not None and groq not in candidates:
        candidates.append(groq)

    for model in candidates:
        try:
            decision = _react_loop(model, tools, system, user, max_rounds)
            if decision:
                return decision
        except Exception as exc:
            if _quota_error(exc):
                _mark_gemini_quota(exc)
            continue
    return {}
