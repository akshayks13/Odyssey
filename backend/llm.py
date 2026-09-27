"""LLM factory and ReAct helper (each Gemini key in turn, then Groq)."""
from __future__ import annotations

import functools
import json
import logging
import re
import threading
import time
from typing import Any

from config import (
    GEMINI_API_KEYS,
    GEMINI_MODEL,
    GROQ_API_KEY,
    GROQ_MODEL,
    LLM_DISABLED,
)

logger = logging.getLogger("odyssey.llm")

MODEL_UNAVAILABLE = (
    "The language model is unavailable right now (it may be rate-limited), so nothing was planned. "
    "Please try again in a few minutes, or open the sample trip."
)
_cooldown_until: dict[str, float] = {}
_last_error: dict[str, str] = {}
_answered: set[str] = set()
_last_used: dict[str, str] = {}
_lock = threading.Lock()


def agent_trace(decision: dict | None = None, *, algorithms: list[str] | None = None, note: str = "") -> dict:
    """How an agent reached its answer, for the timeline: the tools the model actually called, the
    provider that answered, and the algorithms the code ran.

    The tool names come from the decision itself, so this works for every provider — Gemini runs on
    Google's SDK and never produces the LangChain messages the SSE stream watches for tool calls.
    """
    return {
        "tools": list((decision or {}).get("_tool_calls") or []),
        "engine": (_last_used.get("name") if decision else None),
        "algorithms": list(algorithms or []),
        "note": note,
    }


def _quota_error(exc: BaseException) -> bool:
    blob = f"{type(exc).__name__} {exc}".lower()
    return any(
        token in blob
        for token in ("429", "resourceexhausted", "quota", "rate-limit", "rate_limit", "ratelimit", "rate limit")
    )


def _retry_after_seconds(exc: BaseException) -> float:
    """The wait the provider asks for ("try again in 6.5s", "retry_delay { seconds: 21 }"), else 20s."""
    text = str(exc)
    m = re.search(r"try again in\s+(?:(\d+)m)?\s*([\d.]+)s", text, re.IGNORECASE)
    if m:
        return float(m.group(1) or 0) * 60 + float(m.group(2))
    m = re.search(r"retry in\s+([\d.]+)s|retry_delay\s*\{\s*seconds:\s*(\d+)|retryDelay['\"]?\s*:\s*['\"]?([\d.]+)s", text, re.IGNORECASE)
    if m:
        return float(m.group(1) or m.group(2) or m.group(3))
    return 20.0


def _cool_down(provider: str, exc: BaseException) -> None:
    wait = min(900.0, _retry_after_seconds(exc) + 1.0)  # a daily limit can mean many minutes
    with _lock:
        _cooldown_until[provider] = time.monotonic() + wait
        _last_error[provider] = f"rate-limited, retrying in {wait:.0f}s"
    logger.warning("%s rate-limited; using the other provider for %.0fs", provider, wait)


def _cooldown_left(provider: str) -> float:
    with _lock:
        return max(0.0, _cooldown_until.get(provider, 0.0) - time.monotonic())


@functools.lru_cache(maxsize=1)
def _groq_llm():
    if LLM_DISABLED or not GROQ_API_KEY:
        return None
    try:
        from langchain_groq import ChatGroq

        return ChatGroq(model=GROQ_MODEL, api_key=GROQ_API_KEY, temperature=0.2, max_retries=0, timeout=60)
    except Exception:
        return None


class _Gemini:
    """Gemini through Google's own SDK. The LangChain client drops the thought signatures that Gemini 3
    needs to continue a tool call, so it cannot run a multi-step tool loop; this can."""

    def __init__(self, api_key: str, model: str):
        from google import genai
        from google.genai import types

        # A request that never answers must not freeze the plan: give up after a minute and try the next key.
        self.client = genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=60_000))
        self.model = model


@functools.lru_cache(maxsize=1)
def _gemini_llms() -> list:
    """One client per key. Each key has its own quota, so when one is rate-limited the next takes over."""
    if LLM_DISABLED:
        return []
    out = []
    for key in GEMINI_API_KEYS:
        try:
            out.append(_Gemini(key, GEMINI_MODEL))
        except Exception:
            continue
    return out


def _providers() -> list[tuple[str, Any]]:
    """Each Gemini key ("gemini", "gemini-2", ...), then Groq. Gemini follows tool calls reliably and has the
    larger quota; Groq's free daily allowance is used up within a couple of plans."""
    groq = _groq_llm()
    found = [("gemini" if i == 0 else f"gemini-{i + 1}", m) for i, m in enumerate(_gemini_llms())]
    return found + ([("groq", groq)] if groq is not None else [])


def llm_provider_name() -> str | None:
    if LLM_DISABLED:
        return None
    for name, _ in _providers():
        if _cooldown_left(name) == 0.0:
            return f"{name}:{GROQ_MODEL if name == 'groq' else GEMINI_MODEL}"
    return None


def providers_status() -> dict:
    """What the models are actually doing, for /api/health.

    A configured key is not a working key, and a cooldown is only known once a call has failed, so
    each provider reports whether it has ever answered here rather than only whether it is configured.
    """
    if LLM_DISABLED:
        return {"state": "disabled", "ready": [], "cooling_down": {}, "providers": []}
    rows = []
    for name, _ in _providers():
        left = _cooldown_left(name)
        rows.append(
            {
                "provider": name,
                "model": GROQ_MODEL if name == "groq" else GEMINI_MODEL,
                "state": "cooling_down" if left > 0 else ("answered" if name in _answered else "untried"),
                "retry_in_seconds": round(left) if left > 0 else None,
                "last_error": _last_error.get(name),
            }
        )
    ready = [r["provider"] for r in rows if r["state"] != "cooling_down"]
    if not rows:
        state = "no_key_configured"
    elif not ready:
        state = "all_rate_limited"
    elif any(r["state"] == "answered" for r in rows):
        state = "ok"
    else:
        state = "configured_but_untried"  # nothing has been asked yet: a key may still be dead
    return {
        "state": state,
        "ready": ready,
        "cooling_down": {r["provider"]: r["retry_in_seconds"] for r in rows if r["state"] == "cooling_down"},
        "providers": rows,
    }


def get_llm():
    if LLM_DISABLED:
        return None
    providers = _providers()
    ready = [m for name, m in providers if _cooldown_left(name) == 0.0]
    return ready[0] if ready else (providers[0][1] if providers else None)


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


def _run_tool(by_name: dict, name: str, args: dict) -> Any:
    tool = by_name.get(name)
    if tool is None:
        return {"error": f"unknown tool {name}"}
    try:
        return tool.invoke(args)
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}


def _react_loop(llm, tools: list, system: str, user: str, max_rounds: int, json_mode: bool = True) -> dict:
    """Groq (LangChain) tool loop. The last round has no tools, so the model has to give its answer."""
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

    by_name = {t.name: t for t in tools}
    final = llm.bind(response_format={"type": "json_object"}) if json_mode else llm
    with_tools = llm.bind_tools(tools) if tools else final
    messages: list = [SystemMessage(content=system), HumanMessage(content=user)]
    called: list[str] = []

    for round_ in range(max_rounds):
        model = with_tools if (tools and round_ < max_rounds - 1) or not tools else final
        ai: AIMessage = model.invoke(messages)
        messages.append(ai)
        tool_calls = getattr(ai, "tool_calls", None) or []
        if not tool_calls:
            text = (ai.content or "") if isinstance(ai.content, str) else str(ai.content or "")
            decision = parse_json_blob(text)
            if not decision:  # an empty/unparseable reply is a failure, not an answer: let the caller fail over
                return {}
            decision["_tool_calls"] = called
            return decision
        for call in tool_calls:
            name = call.get("name") if isinstance(call, dict) else getattr(call, "name", "")
            args = call.get("args") if isinstance(call, dict) else getattr(call, "args", {}) or {}
            call_id = call.get("id") if isinstance(call, dict) else getattr(call, "id", name)
            result = _run_tool(by_name, name, args)
            called.append(name)
            messages.append(
                ToolMessage(content=json.dumps(result, default=str, separators=(",", ":"))[:4000], tool_call_id=call_id or name, name=name)
            )
    return {}


def _gemini_loop(gem: "_Gemini", tools: list, system: str, user: str, max_rounds: int, json_mode: bool = True) -> dict:
    """Gemini tool loop on Google's SDK. Each model turn is sent back unchanged (thought signatures included)."""
    from google.genai import types
    from langchain_core.utils.function_calling import convert_to_openai_tool

    by_name = {t.name: t for t in tools}
    declarations = [
        types.FunctionDeclaration(
            name=t.name,
            description=t.description,
            parameters_json_schema=convert_to_openai_tool(t)["function"]["parameters"],
        )
        for t in tools
    ]

    def config(with_tools: bool):
        return types.GenerateContentConfig(
            system_instruction=system,
            temperature=0.2,
            tools=[types.Tool(function_declarations=declarations)] if with_tools and declarations else None,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            response_mime_type=None if with_tools and declarations else "application/json",
        )

    contents = [types.Content(role="user", parts=[types.Part(text=user)])]
    called: list[str] = []
    for round_ in range(max_rounds):
        resp = gem.client.models.generate_content(
            model=gem.model, contents=contents, config=config(bool(tools) and round_ < max_rounds - 1)
        )
        turn = resp.candidates[0].content if resp.candidates else None
        if turn is None:
            return {}
        calls = [p.function_call for p in (turn.parts or []) if p.function_call]
        if not calls:
            decision = parse_json_blob(resp.text or "")
            if not decision:  # an empty/unparseable reply is a failure, not an answer: let the caller fail over
                return {}
            decision["_tool_calls"] = called
            return decision
        contents.append(turn)
        replies = []
        for call in calls:
            result = _run_tool(by_name, call.name, dict(call.args or {}))
            called.append(call.name)
            payload = result if isinstance(result, dict) else {"result": result}
            if len(json.dumps(payload, default=str)) > 6000:
                payload = {"truncated": json.dumps(payload, default=str)[:4000]}
            replies.append(types.Part.from_function_response(name=call.name, response=json.loads(json.dumps(payload, default=str))))
        contents.append(types.Content(role="user", parts=replies))
    return {}


_RUNNERS = {"groq": _react_loop, "gemini": _gemini_loop}


def llm_json(system: str, user: str) -> dict:
    """JSON-only call: the first provider that answers (see `_providers`)."""
    return llm_decide(get_llm(), tools=[], system=system, user=user, max_rounds=1)


def llm_decide(llm, tools: list, system: str, user: str, max_rounds: int = 3) -> dict:
    """bind_tools ReAct loop, then parse a JSON decision. Empty dict on failure.

    The first provider in `_providers` answers; on a rate limit or error the next one does. All can call tools. If every provider is
    rate-limited for up to a minute, wait for the first one to come back instead of giving up.
    """
    if LLM_DISABLED or llm is None or not _providers():
        return {}

    for attempt in range(3):
        for name, model in _providers():
            if _cooldown_left(name) > 0.0:
                continue
            runner = _RUNNERS.get(name.split("-")[0])
            if runner is None:  # a misconfigured provider name: skip it, don't mistake this for an LLM failure
                logger.warning("no runner for provider %r", name)
                continue
            try:
                decision = runner(model, tools, system, user, max_rounds)
            except Exception as exc:  # noqa: BLE001
                if _quota_error(exc):
                    _cool_down(name, exc)
                    continue
                if not tools:
                    if "json" in str(exc).lower():  # strict JSON mode rejected the reply: ask again in plain mode
                        try:
                            decision = runner(model, [], system, user, 1, json_mode=False)
                            if decision:
                                return decision
                        except Exception:  # noqa: BLE001
                            pass
                    with _lock:
                        _last_error[name] = str(exc)[:160]
                    logger.warning("%s call failed: %s", name, str(exc)[:160])
                    continue
                # A malformed tool call: let the model still decide, without tools.
                try:
                    decision = runner(model, [], system, user, 1)
                except Exception as exc2:  # noqa: BLE001
                    if _quota_error(exc2):
                        _cool_down(name, exc2)
                    continue
            if decision:
                with _lock:
                    _answered.add(name)
                    _last_error.pop(name, None)
                    _last_used["name"] = f"{name}:{GROQ_MODEL if name == 'groq' else GEMINI_MODEL}"
                return decision
            with _lock:
                _last_error[name] = "unparseable reply"
            logger.warning("%s returned no usable decision (empty or unparseable reply)", name)
        waits = [_cooldown_left(name) for name, _ in _providers()]
        if waits and 0 < min(waits) <= 70.0:
            wait_s = min(waits) + 0.2
            logger.info("every provider is cooling down; waiting %.1fs for the first one to come back", wait_s)
            time.sleep(wait_s)
            continue
        break
    return {}
