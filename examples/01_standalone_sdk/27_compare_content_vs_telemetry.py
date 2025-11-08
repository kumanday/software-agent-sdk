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
from openhands.sdk.tool import Tool
from openhands.tools.terminal import TerminalTool


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

    agent = Agent(llm=llm, tools=[Tool(name=TerminalTool.name)])

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

    # Generate both a tool-call turn and a pure text turn
    conversation.send_message("Please echo 'HELLO' then say Done.")
    conversation.run()
    conversation.send_message("Reply with only OK")
    conversation.run()

    # Locate events dir
    # persistence_dir is non-None since we passed a base dir
    assert conversation.state.persistence_dir is not None
    conv_dir = Path(conversation.state.persistence_dir)
    events_dir = conv_dir / "events"

    # Collect telemetry records written after t0
    telemetry_files = sorted(Path(completions_dir).glob("*.json"))
    telemetry_records: list[dict[str, Any]] = []
    for tf in telemetry_files:
        try:
            data = read_json(tf)
        except Exception:
            continue
        ts = float(data.get("timestamp", 0.0))
        if ts >= t0:
            telemetry_records.append({"path": str(tf), "data": data})

    # Collect assistant MessageEvents
    assistant_events: list[dict[str, Any]] = []
    if events_dir.is_dir():
        for ef in sorted(events_dir.glob("*.json")):
            try:
                ev = read_json(ef)
            except Exception:
                continue
            if ev.get("kind") == "MessageEvent":
                lm = ev.get("llm_message") or {}
                if lm.get("role") == "assistant":
                    assistant_events.append({"path": str(ef), "data": ev})

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

    # Helper to check emptiness for event message content
    def event_content_text(ev: dict[str, Any]) -> str:
        lm = ev["data"].get("llm_message", {})
        return content_text_from_event_llm_message_content(lm.get("content")).strip()

    # Pair up by order for display purposes
    max_len = max(len(telemetry_records), len(assistant_events))

    lines.append("## Cases with empty content in telemetry or events\n")

    for i in range(max_len):
        tele = telemetry_records[i] if i < len(telemetry_records) else None
        evt = assistant_events[i] if i < len(assistant_events) else None

        tele_txt = telemetry_content_text(tele) if tele else ""
        evt_txt = event_content_text(evt) if evt else ""

        tele_empty = (tele is not None) and (len(tele_txt) == 0)
        evt_empty = (evt is not None) and (len(evt_txt) == 0)

        if not (tele_empty or evt_empty):
            continue

        lines.append(f"### Pair #{i + 1}\n")
        if tele is not None:
            lines.append(f"- Telemetry file: `{tele['path']}`\n")
            lines.append(f"- Telemetry content empty: **{tele_empty}**\n")
            lines.append("- Telemetry response JSON:\n")
            lines.append("```json")
            lines.append(dump_json(tele["data"].get("response", {})))
            lines.append("```")
        else:
            lines.append("- Telemetry: none\n")

        if evt is not None:
            lines.append(f"- Event file: `{evt['path']}`\n")
            lines.append(f"- Event content empty: **{evt_empty}**\n")
            lines.append("- Event JSON:\n")
            lines.append("```json")
            lines.append(dump_json(evt["data"]))
            lines.append("```")
        else:
            lines.append("- Event: none\n")

        lines.append("")

    with out_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Wrote report to: {out_path}")


if __name__ == "__main__":
    main()
