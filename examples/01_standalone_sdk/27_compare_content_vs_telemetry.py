"""
Helper: Compare assistant content between telemetry logs and persisted events.

- Runs a short conversation using the configured LLM (LLM_MODEL/LLM_API_KEY)
- Ensures telemetry logging is enabled and a persistence_dir is used
- After the run, scans:
  - Telemetry logs under LOG_DIR/completions for entries written during this run
  - Conversation events under <persistence_dir>/<conv_id_hex>/events
- For each case where assistant content is empty either in telemetry or in events,
  appends a section into a Markdown report with the JSON for:
  - The telemetry LLM response (response object)
  - The matching (or nearest) assistant MessageEvent (if any)

Usage:
  LLM_MODEL=gemini/gemini-2.5-pro LLM_API_KEY=... uv run python \
    examples/01_standalone_sdk/27_compare_content_vs_telemetry.py

Optional env:
  LOG_DIR defaults to ./logs; the report is stored under LOG_DIR/comparisons
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from pydantic import SecretStr

from openhands.sdk import LLM, Agent, Conversation


def ensure_dir(p: str) -> None:
    Path(p).mkdir(parents=True, exist_ok=True)


def read_json(p: Path) -> Any:
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def content_text_from_message_content(payload: Any) -> str:
    """Heuristic extraction of text from message.content payloads.

    Supports several shapes:
    - None -> ""
    - str -> itself
    - list[dict or str] -> concatenate any 'text' fields from dict items or str items
    """
    if payload is None:
        return ""
    if isinstance(payload, str):
        return payload
    if isinstance(payload, list):
        texts: list[str] = []
        for item in payload:
            if isinstance(item, str):
                texts.append(item)
            elif isinstance(item, dict):
                # Common shapes: {"type": "text", "text": "..."}
                t = item.get("text")
                if isinstance(t, str):
                    texts.append(t)
        return "\n".join(texts)
    return ""


def content_text_from_event_llm_message_content(payload: Any) -> str:
    """Extract text from persisted event llm_message.content list.

    Events typically store a list of TextContent dicts with 'text' keys.
    """
    if not payload:
        return ""
    if isinstance(payload, list):
        texts: list[str] = []
        for item in payload:
            if isinstance(item, dict):
                t = item.get("text")
                if isinstance(t, str):
                    texts.append(t)
        return "\n".join(texts)
    return ""


def main() -> None:
    api_key = os.getenv("LLM_API_KEY")
    assert api_key, "LLM_API_KEY is required"
    model = os.getenv("LLM_MODEL", "gemini/gemini-2.5-pro")

    log_dir = os.getenv("LOG_DIR", "logs")
    completions_dir = os.path.join(log_dir, "completions")
    comparisons_dir = os.path.join(log_dir, "comparisons")
    ensure_dir(completions_dir)
    ensure_dir(comparisons_dir)

    # Build LLM with telemetry logging enabled
    llm = LLM(
        model=model,
        api_key=SecretStr(api_key),
        log_completions=True,
        log_completions_folder=completions_dir,
        usage_id="agent",
    )

    agent = Agent(llm=llm)

    # Use .conversations as base; conversation id folder appended automatically
    persistence_base = ".conversations"
    ensure_dir(persistence_base)

    # Mark start time to filter telemetry written during this run
    t0 = time.time() - 1.0  # small slack

    conversation = Conversation(
        agent=agent,
        workspace=os.getcwd(),
        persistence_dir=persistence_base,
    )

    # Generate only text turns (no tool calls) to force MessageEvent
    conversation.send_message("Reply with only OK (no tools)")
    conversation.run()
    conversation.send_message("Now reply with: This is a second text-only message.")
    conversation.run()

    # Locate events dir
    # persistence_dir is non-None since we passed a base dir
    assert conversation.state.persistence_dir is not None
    conv_dir = Path(conversation.state.persistence_dir)
    events_dir = conv_dir / "events"

    # Collect telemetry records written after t0 and map by response id
    telemetry_files = sorted(Path(completions_dir).glob("*.json"))
    telemetry_by_id: dict[str, dict[str, Any]] = {}
    for tf in telemetry_files:
        try:
            data = read_json(tf)
        except Exception:
            continue
        ts = float(data.get("timestamp", 0.0))
        if ts < t0:
            continue
        resp = data.get("response", {})
        rid = resp.get("id")
        if isinstance(rid, str) and rid:
            telemetry_by_id[rid] = {"path": str(tf), "data": data}

    # Collect LLM-convertible events grouped by llm_response_id
    events_by_id: dict[str, dict[str, list[dict[str, Any]]]] = {}
    if events_dir.is_dir():
        for ef in sorted(events_dir.glob("*.json")):
            try:
                ev = read_json(ef)
            except Exception:
                continue
            kind = ev.get("kind")
            # MessageEvent (assistant) – use llm_response_id when present
            if kind == "MessageEvent":
                if ev.get("source") == "agent":
                    rid = ev.get("llm_response_id")
                    if isinstance(rid, str) and rid:
                        events_by_id.setdefault(rid, {"message": [], "action": []})
                        events_by_id[rid]["message"].append(
                            {"path": str(ef), "data": ev}
                        )
            # ActionEvent – always has llm_response_id
            elif kind == "ActionEvent":
                rid = ev.get("llm_response_id")
                if isinstance(rid, str) and rid:
                    events_by_id.setdefault(rid, {"message": [], "action": []})
                    events_by_id[rid]["action"].append({"path": str(ef), "data": ev})

    # Prepare markdown report
    out_path = Path(comparisons_dir) / (f"content_vs_telemetry_{int(time.time())}.md")

    def dump_json(obj: Any) -> str:
        try:
            return json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=True)
        except Exception:
            return str(obj)

    lines: list[str] = []
    lines.append("# Content vs Telemetry Comparison\n")
    lines.append(f"- Model: **{model}**\n")
    lines.append(f"- Conversation dir: `{conv_dir}`\n")
    lines.append(f"- Events dir: `{events_dir}`\n")
    lines.append(f"- Telemetry scan start: `{t0}`\n")
    lines.append("")

    # Helper to check emptiness for telemetry message content
    def telemetry_content_text(rec: dict[str, Any]) -> str:
        resp = rec["data"].get("response", {})
        # Chat completions shape
        choices = resp.get("choices") or []
        if choices:
            msg = (choices[0] or {}).get("message", {})
            content = msg.get("content")
            return content_text_from_message_content(content).strip()
        # Responses API shape
        out_text = resp.get("output_text")
        if isinstance(out_text, str):
            return out_text.strip()
        return ""

    # Helpers to extract MessageEvent-side content and reasoning
    def message_event_text(ev: dict[str, Any]) -> str:
        lm = ev.get("llm_message", {})
        return content_text_from_event_llm_message_content(lm.get("content")).strip()

    def message_event_reasoning(ev: dict[str, Any]) -> str:
        lm = ev.get("llm_message", {})
        rc = lm.get("reasoning_content")
        if isinstance(rc, str):
            return rc.strip()
        return ""

    # Telemetry-side reasoning (chat-completions shape)
    def telemetry_reasoning_text(rec: dict[str, Any]) -> str:
        resp = rec["data"].get("response", {})
        choices = resp.get("choices") or []
        if choices:
            msg = (choices[0] or {}).get("message", {})
            rc = msg.get("reasoning_content")
            if isinstance(rc, str):
                return rc.strip()
        return ""

    lines.append(
        "## MessageEvent-focused cases (empty content and/or reasoning present)\n"
    )

    # Compare telemetry entries with events by response id (MessageEvents only)
    for rid, tele in telemetry_by_id.items():
        ev_bucket = events_by_id.get(rid, {"message": [], "action": []})
        if not ev_bucket.get("message"):
            continue  # focus only on responses that resulted in MessageEvent

        tele_txt = telemetry_content_text(tele)
        tele_empty = len(tele_txt) == 0
        tele_reason = telemetry_reasoning_text(tele)

        # Consider the first MessageEvent for this response id
        msg_event = ev_bucket["message"][0]
        etxt = message_event_text(msg_event["data"]) or ""
        e_reason = message_event_reasoning(msg_event["data"]) or ""
        evt_empty = len(etxt) == 0

        # Report only when we see the phenomenon of interest
        if not ((tele_empty and tele_reason) or (evt_empty and e_reason)):
            continue

        lines.append(f"### Response `{rid}`\n")
        # Telemetry block
        lines.append(f"- Telemetry file: `{tele['path']}`\n")
        lines.append(f"- Telemetry content empty: **{tele_empty}**\n")
        lines.append(f"- Telemetry reasoning present: **{bool(tele_reason)}**\n")
        if tele_reason:
            preview = tele_reason[:200].replace("\n", " ")
            lines.append(f"  - Reasoning preview: {preview}...\n")
        lines.append("- Telemetry response JSON:\n")
        lines.append("```json")
        lines.append(dump_json(tele["data"].get("response", {})))
        lines.append("```")

        # Event block (MessageEvent)
        lines.append(f"- Event (MessageEvent) file: `{msg_event['path']}`\n")
        lines.append(f"  - Event content empty: **{evt_empty}**\n")
        lines.append(f"  - Event reasoning present: **{bool(e_reason)}**\n")
        if e_reason:
            eprev = e_reason[:200].replace("\n", " ")
            lines.append(f"    - Reasoning preview: {eprev}...\n")
        lines.append("  - Event JSON:\n")
        lines.append("```json")
        lines.append(dump_json(msg_event["data"]))
        lines.append("```")
        lines.append("")

    with out_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Wrote report to: {out_path}")


if __name__ == "__main__":
    main()
