"""Streamlit system interface for Veridiction (text/audio -> legal pipeline -> structured output -> TTS).

This mirrors the Gradio UI functionality without the Mermaid graph. A richer graph
renderer will be added in the next step. Current scope: end-to-end run, safety,
structured sections, retrieval table, audio playback, and health checks.
"""

from __future__ import annotations

import json
import importlib
import os
import re
import tempfile
import time
import traceback
import urllib.error
import urllib.request
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import streamlit as st

from agents.langgraph_flow import VeridictionGraph
from audio.transcriber import AudioTranscriber, TranscriberConfig
from tts.speak import TTSConfig, TTSGenerator

# --------------------------------------------------------------------------------------
# Service layer with lazy caches
# --------------------------------------------------------------------------------------


@dataclass
class ServiceCaches:
    flow: Optional[VeridictionGraph] = None
    flow_provider: str = "auto"
    transcribers: Dict[Tuple[str, str, bool], AudioTranscriber] = None
    tts_engines: Dict[Tuple[str, str], TTSGenerator] = None


CACHES = ServiceCaches(transcribers={}, tts_engines={})


def get_flow(top_k: int, provider: str) -> VeridictionGraph:
    if CACHES.flow is None or CACHES.flow_provider != provider:
        CACHES.flow = VeridictionGraph(top_k=top_k, advisor_provider=provider)
        CACHES.flow_provider = provider
    CACHES.flow.top_k = top_k
    return CACHES.flow


def get_transcriber(model_name: str, model_dir: str, local_only: bool) -> AudioTranscriber:
    key = (model_name, model_dir, local_only)
    if key not in CACHES.transcribers:
        CACHES.transcribers[key] = AudioTranscriber(
            config=TranscriberConfig(
                model_name=model_name,
                model_dir=model_dir,
                local_files_only=local_only,
                language="en",
                beam_size=5,
                vad_filter=True,
            )
        )
    return CACHES.transcribers[key]


def get_tts(engine: str, fallback_engine: str) -> TTSGenerator:
    key = (engine, fallback_engine)
    if key not in CACHES.tts_engines:
        CACHES.tts_engines[key] = TTSGenerator(
            config=TTSConfig(
                preferred_engine=engine,
                fallback_engine=fallback_engine,
                output_dir="data/tts",
            )
        )
    return CACHES.tts_engines[key]


# --------------------------------------------------------------------------------------
# Utility helpers (mirrors app_gradio formatting logic)
# --------------------------------------------------------------------------------------


def _format_passages_for_table(passages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for idx, item in enumerate(passages, start=1):
        score = float(item.get("score") or 0.0)
        text = str(item.get("passage", "")).strip().replace("\n", " ")
        preview = text[:260].rstrip() + (" ..." if len(text) > 260 else "")
        source = str(item.get("metadata", {}).get("dataset", ""))
        rows.append({"Rank": idx, "Score": round(score, 4), "Source": source, "Passage Preview": preview})
    return rows


def _build_structured_json(output: Dict[str, Any], elapsed_ms: float) -> Dict[str, Any]:
    safety = output.get("safety", {}) or {}
    passages = output.get("retrieved_passages", []) or []
    structured_response = output.get("structured_response", {}) or {}

    structured = {
        "meta": {
            "input_mode": output.get("input_mode", "text"),
            "latency_ms": round(elapsed_ms, 2),
            "top_k_retrieved": len(passages),
        },
        "input": {
            "transcript_or_query": output.get("transcript", ""),
            "audio_metadata": output.get("audio_metadata", {}),
        },
        "classification": {
            "claim_type": output.get("claim_type", "other"),
            "urgency": output.get("urgency", "low"),
            "confidence": output.get("confidence", 0.0),
        },
        "retrieval": {
            "passages": passages,
        },
        "structured_response": structured_response,
        "safety": {
            "risk_flags": safety.get("risk_flags", []),
            "safe_next_steps": safety.get("safe_next_steps", []),
            "disclaimer": safety.get("disclaimer", ""),
        },
        "tts": output.get("tts", {}),
        "tts_summary": output.get("tts_summary", ""),
        "final_text": output.get("final_text", ""),
    }
    return structured


def _classify_flow_node_type(title: str, details: str) -> str:
    text = f"{title} {details}".lower()
    if any(token in text for token in ("emergency", "danger", "safety", "112", "helpline")):
        return "emergency"
    if any(token in text for token in ("document", "evidence", "proof", "records", "paper")):
        return "documentation"
    if any(token in text for token in ("file", "complaint", "fir", "petition", "application", "notice")):
        return "filing"
    if any(token in text for token in ("hearing", "follow", "comply", "order", "appear")):
        return "follow_up"
    return "action"


def _sanitize_flow_title(title: str, details: str, idx: int) -> str:
    raw = (title or "").strip()
    if not raw or re.fullmatch(r"step\s*\d+", raw, flags=re.IGNORECASE):
        detail_words = [w for w in re.split(r"\s+", (details or "").strip()) if w]
        if detail_words:
            short = " ".join(detail_words[:4]).strip(".,;: ")
            return short.title()
        return f"Action {idx}"
    return raw


def _wrap_text_for_node(text: str, max_len: int = 32) -> str:
    words = text.split()
    if not words:
        return ""
    lines: List[str] = []
    current: List[str] = []
    current_len = 0
    for word in words:
        add_len = len(word) if not current else len(word) + 1
        if current_len + add_len > max_len:
            lines.append(" ".join(current))
            current = [word]
            current_len = len(word)
        else:
            current.append(word)
            current_len += add_len
    if current:
        lines.append(" ".join(current))
    return "<br>".join(lines[:3])


def _build_flow_explanation_lines(graph_data: Dict[str, Any]) -> List[str]:
    nodes = sorted(list(graph_data.get("nodes", [])), key=lambda x: int(x.get("order", 0)))
    lines: List[str] = []
    for node in nodes:
        node_type = str(node.get("type", "action"))
        if node_type in {"start", "end"}:
            continue
        order = int(node.get("order", 0))
        title = str(node.get("title", "")).strip()
        details = str(node.get("details", "")).strip()
        if title and details:
            lines.append(f"{order}. {title}: {details}")
        elif details:
            lines.append(f"{order}. {details}")
        elif title:
            lines.append(f"{order}. {title}")
    return lines


def _build_flow_explanation_items(graph_data: Dict[str, Any]) -> List[Dict[str, str]]:
    nodes = sorted(list(graph_data.get("nodes", [])), key=lambda x: int(x.get("order", 0)))
    items: List[Dict[str, str]] = []
    for node in nodes:
        node_type = str(node.get("type", "action"))
        if node_type in {"start", "end"}:
            continue
        order = int(node.get("order", 0))
        title = str(node.get("title", "")).strip() or "Action"
        details = str(node.get("details", "")).strip() or "Proceed with this step."
        items.append(
            {
                "step": str(order),
                "title": title,
                "details": details,
                "type": node_type,
            }
        )
    return items


def _node_fill_color(node_type: str) -> str:
    palette = {
        "start": "#D9F2FF",
        "end": "#DDF8E8",
        "emergency": "#FFD7D7",
        "documentation": "#FDECC8",
        "filing": "#E4E0FF",
        "follow_up": "#E0F0FF",
        "action": "#F0F4F8",
    }
    return palette.get(node_type, "#F0F4F8")


def _risk_border_color(severity_level: str, urgency: str) -> str:
    sev = (severity_level or "").lower()
    urg = (urgency or "").lower()
    if sev == "critical":
        return "#B00020"
    if sev == "high" or urg == "high":
        return "#D35400"
    if sev == "medium" or urg == "medium":
        return "#A15C00"
    return "#2E7D32"


def _build_flowchart_graph_data(
    structured_response: Dict[str, Any],
    urgency: str,
    risk_flags: List[str],
    safe_next_steps: List[str],
) -> Dict[str, Any]:
    flow_steps = structured_response.get("flowchart", []) or []
    severity = structured_response.get("severity_assessment", {}) or {}
    severity_level = str(severity.get("level", urgency or "medium"))
    border_color = _risk_border_color(severity_level=severity_level, urgency=urgency)

    risk_flag_set = {str(x).lower() for x in risk_flags}
    has_immediate_risk = any(x in risk_flag_set for x in ("immediate_danger", "domestic_violence_risk", "high_urgency"))
    emergency_detail = (
        str(safe_next_steps[0]).strip()
        if safe_next_steps
        else "Call 112 and move to a safe place before proceeding with legal filing."
    )

    nodes: List[Dict[str, Any]] = [
        {
            "id": "start",
            "label": "Case Intake",
            "title": "Case Intake",
            "details": "User issue captured and legal triage started.",
            "type": "start",
            "order": 0,
            "fill_color": _node_fill_color("start"),
            "border_color": border_color,
        }
    ]

    nodes.append(
        {
            "id": "decision_risk",
            "label": "Immediate Risk Check",
            "title": "Immediate danger or threat present?",
            "details": (
                "Risk indicators detected; emergency actions should be prioritized first."
                if has_immediate_risk
                else "No immediate physical danger detected; proceed through standard legal steps."
            ),
            "type": "decision",
            "order": 1,
            "fill_color": "#FFF3CD",
            "border_color": border_color,
        }
    )

    nodes.append(
        {
            "id": "emergency_action",
            "label": "Emergency Action",
            "title": "Emergency action",
            "details": emergency_detail,
            "type": "emergency",
            "order": 2,
            "fill_color": _node_fill_color("emergency"),
            "border_color": border_color,
        }
    )

    for idx, raw in enumerate(flow_steps, start=1):
        details = str(raw.get("details", "")).strip() or "Proceed with this action."
        title = _sanitize_flow_title(str(raw.get("title", f"Step {idx}")), details, idx)
        node_type = _classify_flow_node_type(title=title, details=details)
        nodes.append(
            {
                "id": f"step_{idx}",
                "label": title,
                "title": title,
                "details": details,
                "type": node_type,
                "order": idx + 2,
                "fill_color": _node_fill_color(node_type),
                "border_color": border_color,
            }
        )

    end_order = len(flow_steps) + 3
    nodes.append(
        {
            "id": "end",
            "label": "Next Legal Milestone",
            "title": "Next Legal Milestone",
            "details": "Submit, track, and act on authority/court response.",
            "type": "end",
            "order": end_order,
            "fill_color": _node_fill_color("end"),
            "border_color": border_color,
        }
    )

    edges: List[Dict[str, Any]] = []

    process_ids = [f"step_{idx}" for idx in range(1, len(flow_steps) + 1)]
    first_process = process_ids[0] if process_ids else "end"

    edges.append({"id": "edge_1", "source": "start", "target": "decision_risk", "label": "intake"})
    edges.append({"id": "edge_2", "source": "decision_risk", "target": "emergency_action", "label": "Yes"})
    edges.append({"id": "edge_3", "source": "decision_risk", "target": first_process, "label": "No"})
    edges.append({"id": "edge_4", "source": "emergency_action", "target": first_process, "label": "After safety"})

    for idx in range(len(process_ids) - 1):
        edges.append(
            {
                "id": f"edge_process_{idx + 1}",
                "source": process_ids[idx],
                "target": process_ids[idx + 1],
                "label": "next",
            }
        )

    edges.append(
        {
            "id": "edge_end",
            "source": process_ids[-1] if process_ids else "emergency_action",
            "target": "end",
            "label": "submit",
        }
    )

    return {
        "meta": {
            "severity_level": severity_level,
            "urgency": urgency,
            "risk_flags": risk_flags,
            "layout": "left_to_right_dag",
        },
        "nodes": nodes,
        "edges": edges,
    }


def _render_flowchart_graphviz(graph_data: Dict[str, Any]) -> bool:
    try:
        graphviz = importlib.import_module("graphviz")
    except Exception:
        return False

    nodes = sorted(list(graph_data.get("nodes", [])), key=lambda x: int(x.get("order", 0)))
    edges = list(graph_data.get("edges", []))
    meta = graph_data.get("meta", {}) or {}
    if not nodes:
        return False

    shape_map = {
        "start": "ellipse",
        "end": "ellipse",
        "decision": "diamond",
        "emergency": "box",
        "documentation": "box",
        "filing": "box",
        "follow_up": "box",
        "action": "box",
    }

    dot = graphviz.Digraph("legal_flow", graph_attr={"rankdir": "LR", "splines": "spline", "nodesep": "0.55", "ranksep": "0.8"})
    dot.attr("node", style="filled", fontname="Helvetica", fontsize="10", color="#A15C00", penwidth="2")
    dot.attr("edge", color="#6B7280", penwidth="1.6", fontname="Helvetica", fontsize="10")

    for node in nodes:
        node_id = str(node.get("id", ""))
        if not node_id:
            continue
        node_type = str(node.get("type", "action"))
        title = str(node.get("title", "")).strip() or str(node.get("label", "Action")).strip() or "Action"
        label = _wrap_text_for_node(title, max_len=18).replace("<br>", "\\n")
        dot.node(
            node_id,
            label=label,
            shape=shape_map.get(node_type, "box"),
            fillcolor=str(node.get("fill_color", "#F0F4F8")),
        )

    for edge in edges:
        src = str(edge.get("source", ""))
        dst = str(edge.get("target", ""))
        if not src or not dst:
            continue
        label = str(edge.get("label", "")).strip()
        dot.edge(src, dst, label=label)

    st.caption(
        "Flowchart: standard symbols are used (Oval = Start/End, Diamond = Decision, Rectangle = Process). "
        f"Urgency: {meta.get('urgency', 'medium')} | Severity: {meta.get('severity_level', 'medium')}"
    )
    st.graphviz_chart(dot)
    return True


def _render_flowchart_native(graph_data: Dict[str, Any]) -> bool:
    nodes = sorted(list(graph_data.get("nodes", [])), key=lambda x: int(x.get("order", 0)))
    if not nodes:
        return False

    by_id = {str(n.get("id", "")): n for n in nodes}

    def _card(title: str, details: str, bg: str = "#F8FAFC") -> str:
        safe_title = escape((title or "Action").strip())
        safe_details = escape((details or "Proceed with this step.").strip())
        return (
            f"<div style='min-width:200px;max-width:260px;border:2px solid #A15C00;"
            f"border-radius:10px;padding:10px 12px;background:{bg};margin:6px;'>"
            f"<div style='font-weight:700;color:#1F2937;margin-bottom:6px'>{safe_title}</div>"
            f"<div style='color:#374151;font-size:13px;line-height:1.25'>{safe_details}</div>"
            "</div>"
        )

    def _arrow(label: str = "") -> str:
        safe_label = escape(label)
        label_html = f"<div style='font-size:12px;color:#4B5563'>{safe_label}</div>" if safe_label else ""
        return (
            "<div style='display:flex;flex-direction:column;align-items:center;justify-content:center;"
            "min-width:42px;margin:0 4px;'>"
            f"{label_html}"
            "<div style='font-size:24px;color:#6B7280;line-height:1'>→</div>"
            "</div>"
        )

    start = by_id.get("start", {"title": "Case Intake", "details": "Issue captured."})
    decision = by_id.get("decision_risk", {"title": "Risk Check", "details": "Assess immediate danger."})
    emergency = by_id.get("emergency_action", {"title": "Emergency Action", "details": "Take urgent safety action."})
    end = by_id.get("end", {"title": "Next Legal Milestone", "details": "Submit and track response."})
    process_nodes = [n for n in nodes if str(n.get("id", "")).startswith("step_")]

    main_parts: List[str] = [
        _card(str(start.get("title", "Case Intake")), str(start.get("details", "Issue captured.")), "#D9F2FF"),
        _arrow(),
        _card(str(decision.get("title", "Immediate danger or threat present?")), str(decision.get("details", "")), "#FFF3CD"),
        _arrow("No"),
    ]
    for idx, node in enumerate(process_nodes):
        main_parts.append(_card(str(node.get("title", "Action")), str(node.get("details", "")), str(node.get("fill_color", "#F0F4F8"))))
        if idx < len(process_nodes) - 1:
            main_parts.append(_arrow())
    if process_nodes:
        main_parts.append(_arrow())
    main_parts.append(_card(str(end.get("title", "Next Legal Milestone")), str(end.get("details", "")), "#DDF8E8"))

    emergency_parts: List[str] = [
        _card(str(decision.get("title", "Immediate danger or threat present?")), str(decision.get("details", "")), "#FFF3CD"),
        _arrow("Yes"),
        _card(str(emergency.get("title", "Emergency Action")), str(emergency.get("details", "")), "#FFD7D7"),
    ]
    if process_nodes:
        emergency_parts.extend([_arrow("Then continue"), _card(str(process_nodes[0].get("title", "Action")), str(process_nodes[0].get("details", "")), str(process_nodes[0].get("fill_color", "#F0F4F8")))])

    st.caption("Standard flow view with explicit No/Yes decision branches.")
    st.markdown(
        "<div style='font-weight:700;color:#111827;margin:4px 0 6px;'>Main Legal Path</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div style='display:flex;flex-wrap:wrap;align-items:center;gap:4px;'>" + "".join(main_parts) + "</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div style='font-weight:700;color:#111827;margin:12px 0 6px;'>Emergency Branch</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div style='display:flex;flex-wrap:wrap;align-items:center;gap:4px;'>" + "".join(emergency_parts) + "</div>",
        unsafe_allow_html=True,
    )
    return True


def _section_to_text(title: str, lines: List[str]) -> str:
    if not lines:
        return f"{title}:\n- Not available"
    out = [f"{title}:"]
    for item in lines:
        out.append(f"- {item}")
    return "\n".join(out)


def _emit_progress(callback: Optional[Callable[[int, str], None]], pct: int, message: str) -> None:
    if callback is None:
        return
    callback(max(0, min(100, int(pct))), message)


def _build_dynamic_tts_summary(output: Dict[str, Any], transcript: str) -> str:
    structured = output.get("structured_response", {}) or {}
    steps = structured.get("possible_steps", {}) or {}
    docs = structured.get("required_documentation", {}) or {}
    filing = structured.get("courts_and_filing_process", {}) or {}
    helplines = structured.get("helplines_india", []) or []
    safety = output.get("safety", {}) or {}

    immediate_actions = [str(x).strip() for x in steps.get("immediate_actions", []) if str(x).strip()]
    legal_actions = [str(x).strip() for x in steps.get("legal_actions", []) if str(x).strip()]
    doc_items = (
        [str(x).strip() for x in docs.get("mandatory", []) if str(x).strip()]
        + [str(x).strip() for x in docs.get("supporting", []) if str(x).strip()]
    )
    process_items = [str(x).strip() for x in filing.get("application_process", []) if str(x).strip()]
    forum_items = [str(x).strip() for x in filing.get("courts_forum", []) if str(x).strip()]
    safety_steps = [str(x).strip() for x in safety.get("safe_next_steps", []) if str(x).strip()]

    next_steps_items = (immediate_actions + legal_actions)[:3]
    next_steps_text = "; ".join(next_steps_items) if next_steps_items else (
        "Collect evidence, prepare a dated timeline, and approach legal aid"
    )

    docs_text = ", ".join(doc_items[:4]) if doc_items else "identity proof and all incident-related records"

    if process_items:
        process_text = " then ".join(process_items[:2])
    elif forum_items:
        process_text = f"Approach {forum_items[0]} and submit a written complaint with your documents"
    else:
        process_text = "Submit a written complaint to the appropriate authority and track acknowledgement"

    emergency_text = ""
    emergency_lines: List[str] = []
    if safety_steps:
        emergency_lines.append(safety_steps[0])
    if helplines:
        for helpline in helplines[:2]:
            name = str((helpline or {}).get("name", "")).strip()
            number = str((helpline or {}).get("number", "")).strip()
            if name and number:
                emergency_lines.append(f"{name}: {number}")
    if emergency_lines:
        emergency_text = "Emergency support: " + " | ".join(emergency_lines[:2])
    else:
        emergency_text = "Emergency support: call 112 if there is immediate danger"

    claim = str(output.get("claim_type", "legal issue")).replace("_", " ")
    return (
        f"For your {claim} case, here is what to do now. "
        f"Next steps: {next_steps_text}. "
        f"Keep these documents ready: {docs_text}. "
        f"Process to follow: {process_text}. "
        f"{emergency_text}."
    )


def _extract_structured_sections(structured: Dict[str, Any]) -> Tuple[str, str, str, str, str, str, str, str]:
    scenario = structured.get("case_scenario", {}) or {}
    steps = structured.get("possible_steps", {}) or {}
    docs = structured.get("required_documentation", {}) or {}
    filing = structured.get("courts_and_filing_process", {}) or {}
    severity = structured.get("severity_assessment", {}) or {}
    helplines = structured.get("helplines_india", []) or []
    flowchart = structured.get("flowchart", []) or []

    scenario_lines = [scenario.get("summary", "")] + list(scenario.get("key_facts", []))
    if scenario.get("missing_details"):
        scenario_lines.append("Missing details: " + "; ".join(scenario.get("missing_details", [])))

    step_lines = (
        [f"Immediate: {x}" for x in steps.get("immediate_actions", [])]
        + [f"Legal: {x}" for x in steps.get("legal_actions", [])]
        + [f"Next 48 Hours: {x}" for x in steps.get("next_48_hours", [])]
    )

    doc_lines = (
        [f"Mandatory: {x}" for x in docs.get("mandatory", [])]
        + [f"Supporting: {x}" for x in docs.get("supporting", [])]
        + [f"Optional: {x}" for x in docs.get("optional", [])]
    )

    court_lines = (
        [f"State: {filing.get('state', '')}"]
        + [f"Forum: {x}" for x in filing.get("courts_forum", [])]
        + [f"Process: {x}" for x in filing.get("application_process", [])]
        + [f"Jurisdiction: {filing.get('jurisdiction_note', '')}"]
    )

    severity_lines = [
        f"Level: {severity.get('level', '')}",
        f"Rationale: {severity.get('rationale', '')}",
        f"Time Sensitivity: {severity.get('time_sensitivity', '')}",
    ]

    helpline_lines = [
        f"{h.get('name', '')}: {h.get('number', '')} | {h.get('availability', '')} | {h.get('applicability', '')}"
        for h in helplines
    ]

    flow_lines: List[str] = []
    for idx, step in enumerate(flowchart, start=1):
        step_no = step.get("step", idx)
        title = str(step.get("title", "")).strip()
        details = str(step.get("details", "")).strip()
        if title and re.fullmatch(r"step\s*\d+", title, flags=re.IGNORECASE):
            line = f"{step_no}. {details}" if details else f"{step_no}."
        elif title and details:
            line = f"{step_no}. {title}: {details}"
        elif details:
            line = f"{step_no}. {details}"
        else:
            line = f"{step_no}."
        flow_lines.append(line)

    return (
        _section_to_text("Case Scenario", [x for x in scenario_lines if x]),
        _section_to_text("Possible Steps", [x for x in step_lines if x]),
        _section_to_text("Required Documentation", [x for x in doc_lines if x]),
        _section_to_text("Courts and Filing Process", [x for x in court_lines if x]),
        _section_to_text("Severity", [x for x in severity_lines if x]),
        _section_to_text("India Helplines", [x for x in helpline_lines if x]),
        _section_to_text("Flowchart", [x for x in flow_lines if x]),
        str(structured.get("tts_summary", "")),
    )


# --------------------------------------------------------------------------------------
# Health checks
# --------------------------------------------------------------------------------------


def health_check_provider(timeout: int = 8) -> Tuple[bool, str]:
    api_key = os.getenv("GROK_API_KEY") or os.getenv("GROQ_API_KEY")
    base_url = os.getenv("GROK_BASE_URL") or os.getenv("GROQ_BASE_URL") or "https://api.groq.com/openai/v1"
    model = os.getenv("GROK_MODEL") or os.getenv("GROQ_MODEL") or "llama-3.3-70b-versatile"
    if not api_key:
        return False, "Missing GROQ/GROK API key"

    body = {
        "model": model,
        "messages": [{"role": "user", "content": "ping"}],
        "max_tokens": 4,
    }
    req = urllib.request.Request(
        url=f"{base_url.rstrip('/')}/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status == 200:
                return True, f"Provider reachable ({model})"
            return False, f"HTTP {resp.status}"
    except urllib.error.HTTPError as exc:
        return False, f"HTTP {exc.code}"
    except urllib.error.URLError as exc:
        return False, f"Conn error: {exc.reason}"


def health_check_retriever(flow: VeridictionGraph) -> Tuple[bool, str]:
    try:
        hits = flow.retriever.query("test retrieval", top_k=1)
        if hits:
            return True, "Retriever returned results"
        return False, "No results"
    except Exception as exc:  # pragma: no cover
        return False, f"Retriever error: {exc}"


# --------------------------------------------------------------------------------------
# Core pipeline runner
# --------------------------------------------------------------------------------------


def _normalize_mode(input_mode: str, query: str, has_audio: bool) -> bool:
    mode = (input_mode or "auto").strip().lower()
    if mode == "audio":
        return True
    if mode == "text":
        return False
    # auto
    use_audio = has_audio and not query
    if has_audio and query:
        use_audio = True
    return use_audio


def run_pipeline(
    input_mode: str,
    query_text: str,
    audio_file: Optional[bytes],
    audio_name: str,
    top_k: int,
    advisor_provider: str,
    stt_model_name: str,
    stt_model_dir: str,
    stt_local_files_only: bool,
    enable_tts: bool,
    tts_engine: str,
    tts_fallback_engine: str,
    progress_callback: Optional[Callable[[int, str], None]] = None,
) -> Dict[str, Any]:
    start = time.perf_counter()
    transcript = ""
    audio_meta: Dict[str, Any] = {}

    _emit_progress(progress_callback, 5, "Validating inputs")

    has_audio = bool(audio_file)
    use_audio = _normalize_mode(input_mode, query_text, has_audio)

    if use_audio:
        _emit_progress(progress_callback, 15, "Capturing and transcribing voice input")
        if not audio_file:
            raise ValueError("Audio mode selected but no audio provided.")
        suffix = os.path.splitext(audio_name or "upload.wav")[1] or ".wav"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(audio_file)
            tmp_path = tmp.name
        transcriber = get_transcriber(stt_model_name, stt_model_dir, stt_local_files_only)
        transcription = transcriber.transcribe_file(
            audio_path=tmp_path,
            language="en",
            beam_size=5,
            vad_filter=True,
        )
        transcript = str(transcription.get("text", "")).strip()
        audio_meta = {
            "audio_file": audio_name,
            "language": transcription.get("language", "en"),
            "language_probability": transcription.get("language_probability", 0.0),
            "duration": transcription.get("duration", 0.0),
            "segments": len(transcription.get("segments", [])),
        }
        if not transcript:
            raise ValueError("Transcription is empty. Try clearer speech and lower background noise.")
    else:
        _emit_progress(progress_callback, 15, "Preparing text query")
        transcript = (query_text or "").strip()

    if not transcript:
        raise ValueError("Provide either text query or audio input.")

    _emit_progress(progress_callback, 40, "Running retrieval and legal reasoning")
    flow = get_flow(top_k=top_k, provider=advisor_provider)
    output = flow.run(transcript)
    output["input_mode"] = "audio" if use_audio else "text"
    output["transcript"] = transcript
    if audio_meta:
        output["audio_metadata"] = audio_meta

    # Guarantee scenario-specific TTS summary every run, even if provider/fallback is generic.
    dynamic_tts_summary = _build_dynamic_tts_summary(output=output, transcript=transcript)
    output["tts_summary"] = dynamic_tts_summary
    structured_response = output.get("structured_response", {}) or {}
    if structured_response:
        structured_response["tts_summary"] = dynamic_tts_summary
        output["structured_response"] = structured_response

    risk_flags = [str(x) for x in (output.get("safety", {}) or {}).get("risk_flags", [])]
    safe_next_steps = [str(x) for x in (output.get("safety", {}) or {}).get("safe_next_steps", [])]
    flowchart_graph_data = _build_flowchart_graph_data(
        structured_response=structured_response,
        urgency=str(output.get("urgency", "medium")),
        risk_flags=risk_flags,
        safe_next_steps=safe_next_steps,
    )
    output["flowchart_graph"] = flowchart_graph_data

    tts_path: Optional[str] = None
    if enable_tts:
        _emit_progress(progress_callback, 80, "Generating speech output")
        tts = get_tts(tts_engine, tts_fallback_engine)
        tts_spoken_text = str(output.get("tts_summary", "")).strip() or str(output.get("final_text", ""))
        tts_result = tts.speak_to_file(
            text=tts_spoken_text,
            include_disclaimer=False,
        )
        tts_path = tts_result["audio_path"]
        output["tts"] = {
            "engine": tts_result["engine"],
            "audio_path": tts_result["audio_path"],
            "mime_type": tts_result["mime_type"],
            "size_bytes": tts_result["size_bytes"],
            "spoken_text": tts_spoken_text,
        }

    _emit_progress(progress_callback, 92, "Formatting final response")

    elapsed_ms = (time.perf_counter() - start) * 1000
    passages = output.get("retrieved_passages", []) or []
    passage_table = _format_passages_for_table(passages)
    structured = _build_structured_json(output, elapsed_ms)
    structured["flowchart_graph"] = flowchart_graph_data
    structured_response = output.get("structured_response", {}) or {}
    (
        case_scenario_text,
        possible_steps_text,
        required_docs_text,
        courts_process_text,
        severity_text,
        helplines_text,
        flowchart_text,
        tts_summary_text,
    ) = _extract_structured_sections(structured_response)
    safety = output.get("safety", {}) or {}
    flow_explanation_lines = _build_flow_explanation_lines(flowchart_graph_data)

    status = (
        "Run completed successfully. "
        f"Mode={output.get('input_mode', 'text')} | "
        f"Claim={output.get('claim_type', 'other')} | "
        f"Latency={elapsed_ms:.1f} ms"
    )

    _emit_progress(progress_callback, 100, "Completed")

    return {
        "transcript": transcript,
        "claim_type": str(output.get("claim_type", "other")),
        "urgency": str(output.get("urgency", "low")),
        "confidence": float(output.get("confidence", 0.0)),
        "final_text": str(output.get("final_text", "")),
        "case_scenario_text": case_scenario_text,
        "possible_steps_text": possible_steps_text,
        "required_docs_text": required_docs_text,
        "courts_process_text": courts_process_text,
        "severity_text": severity_text,
        "helplines_text": helplines_text,
        "flowchart_text": flowchart_text,
        "flowchart_explanation_lines": flow_explanation_lines,
        "tts_summary_text": tts_summary_text,
        "safety_data": safety,
        "safety_json": json.dumps(safety, ensure_ascii=True, indent=2),
        "passage_table": passage_table,
        "structured_json": json.dumps(structured, ensure_ascii=True, indent=2),
        "raw_json": json.dumps(output, ensure_ascii=True, indent=2),
        "flowchart_graph_json": json.dumps(flowchart_graph_data, ensure_ascii=True, indent=2),
        "tts_path": tts_path,
        "status": status,
    }


# --------------------------------------------------------------------------------------
# Streamlit UI
# --------------------------------------------------------------------------------------


def _render_status_panel(result: Dict[str, Any]) -> None:
    st.subheader("Run Summary")
    c1, c2, c3 = st.columns(3)
    c1.metric("Claim Type", result.get("claim_type", ""))
    c2.metric("Urgency", result.get("urgency", ""))
    c3.metric("Confidence", f"{result.get('confidence', 0.0):.3f}")
    st.write(result.get("status", ""))


def _render_tabs(result: Dict[str, Any]) -> None:
    tabs = st.tabs([
        "Overview",
        "Legal Sections",
        "Flowchart",
        "Safety",
        "Retrieval",
        "Structured JSON",
        "Audio Output",
    ])

    with tabs[0]:
        st.text_area("Transcript / Final Query", result.get("transcript", ""), height=120)
        st.text_area("Final Advisor Text", result.get("final_text", ""), height=200)

    with tabs[1]:
        st.text_area("Case Scenario", result.get("case_scenario_text", ""), height=160)
        st.text_area("Possible Steps", result.get("possible_steps_text", ""), height=160)
        st.text_area("Required Documentation", result.get("required_docs_text", ""), height=140)
        st.text_area("Courts and Filing Process", result.get("courts_process_text", ""), height=140)
        st.text_area("Severity Assessment", result.get("severity_text", ""), height=120)
        st.text_area("India Helplines", result.get("helplines_text", ""), height=120)

    with tabs[2]:
        graph_json_text = result.get("flowchart_graph_json", "{}")
        try:
            graph_data = json.loads(graph_json_text)
        except Exception:
            graph_data = {}

        rendered = _render_flowchart_graphviz(graph_data)
        if not rendered:
            rendered = _render_flowchart_native(graph_data)
        if not rendered:
            st.warning("Flowchart rendering unavailable for this response. Showing detailed step explanation.")

        st.subheader("Step-by-Step Explanation")
        explanation_items = _build_flow_explanation_items(graph_data)
        if explanation_items:
            for item in explanation_items:
                step = item.get("step", "")
                title = item.get("title", "Action")
                details = item.get("details", "Proceed with this step.")
                node_type = str(item.get("type", "action")).replace("_", " ").title()
                with st.expander(f"Step {step}: {title}", expanded=True):
                    st.markdown(f"Type: {node_type}")
                    st.markdown(details)
        else:
            explanation_lines = result.get("flowchart_explanation_lines", []) or []
            if explanation_lines:
                for line in explanation_lines:
                    st.markdown(f"- {line}")
            else:
                fallback_text = str(result.get("flowchart_text", "")).strip()
                if fallback_text:
                    st.markdown(fallback_text.replace("\n", "  \n"))
                else:
                    st.info("No flowchart explanation available for this query.")

    with tabs[3]:
        safety_data = result.get("safety_data", {}) or {}
        risk_flags = list(safety_data.get("risk_flags", []) or [])
        safe_next_steps = list(safety_data.get("safe_next_steps", []) or [])
        disclaimer = str(safety_data.get("disclaimer", "")).strip()

        st.subheader("Safety Guidance")
        if risk_flags:
            st.error("Potential safety risks detected. Please prioritize immediate safety actions.")
            st.markdown("Detected risk indicators:")
            for flag in risk_flags:
                st.markdown(f"- {str(flag).replace('_', ' ').title()}")
        else:
            st.success("No immediate safety danger detected from the current query.")

        st.markdown("Recommended safe actions:")
        if safe_next_steps:
            for step in safe_next_steps:
                st.markdown(f"- {step}")
        else:
            st.markdown("- Follow the legal steps in the flowchart and seek legal aid support.")

        if disclaimer:
            st.info(disclaimer)

    with tabs[4]:
        table = result.get("passage_table", [])
        if table:
            st.dataframe(table, use_container_width=True)
        else:
            st.write("No passages returned.")

    with tabs[5]:
        st.code(result.get("structured_json", "{}"), language="json")
        st.code(result.get("raw_json", "{}"), language="json")

    with tabs[6]:
        st.text_area("TTS Summary (Spoken)", result.get("tts_summary_text", ""), height=100)
        if result.get("tts_path"):
            audio_file = result["tts_path"]
            try:
                audio_bytes = Path(audio_file).read_bytes()
                st.audio(audio_bytes, format="audio/mpeg")
            except OSError:
                st.warning("TTS file not found on disk.")
        else:
            st.write("TTS not generated or disabled.")


def _sidebar_controls() -> Dict[str, Any]:
    st.sidebar.header("Controls")
    input_mode = st.sidebar.radio("Input Mode", ["Auto", "Text", "Audio"], index=0)
    top_k = st.sidebar.slider("Retriever Top-K", min_value=1, max_value=10, value=5, step=1)
    advisor_provider = st.sidebar.selectbox("Advisor Provider", ["auto", "grok", "fallback"], index=0)
    enable_tts = st.sidebar.checkbox("Enable TTS", value=True)
    tts_engine = st.sidebar.selectbox("TTS Engine", ["edge_tts", "pyttsx3"], index=0)
    tts_fallback = st.sidebar.selectbox("TTS Fallback Engine", ["pyttsx3", "edge_tts"], index=0)

    st.sidebar.markdown("---")
    st.sidebar.subheader("Speech-to-Text")
    stt_model_name = st.sidebar.text_input("STT Model", value="distil-large-v3")
    stt_model_dir = st.sidebar.text_input("STT Cache Dir", value="data/models/faster-whisper")
    stt_local_only = st.sidebar.checkbox("STT Local Files Only", value=True)

    st.sidebar.markdown("---")
    if st.sidebar.button("Run Health Checks"):
        with st.spinner("Checking provider and retriever..."):
            ok_provider, msg_provider = health_check_provider()
            ok_retriever, msg_retriever = health_check_retriever(get_flow(top_k, advisor_provider))
        st.sidebar.success(f"Provider: {'OK' if ok_provider else 'FAIL'} | {msg_provider}")
        st.sidebar.success(f"Retriever: {'OK' if ok_retriever else 'FAIL'} | {msg_retriever}")

    return {
        "input_mode": input_mode,
        "top_k": top_k,
        "advisor_provider": advisor_provider,
        "enable_tts": enable_tts,
        "tts_engine": tts_engine,
        "tts_fallback": tts_fallback,
        "stt_model_name": stt_model_name,
        "stt_model_dir": stt_model_dir,
        "stt_local_only": stt_local_only,
    }


def main() -> None:
    st.set_page_config(page_title="Veridiction Law Assistant (Streamlit)", layout="wide")
    Path("data/tts").mkdir(parents=True, exist_ok=True)

    st.title("Veridiction Law Assistant")
    st.caption("Voice/Text -> Legal Pipeline -> Structured Response -> TTS")

    controls = _sidebar_controls()

    st.markdown("### Input Console")
    query = st.text_area(
        "Text Query",
        value="",
        height=140,
        placeholder="Example: My employer has not paid my salary for 3 months.",
    )

    st.markdown("#### Voice Input")
    if hasattr(st, "audio_input"):
        st.caption("Record directly from your microphone for real-time voice queries.")
        audio_upload = st.audio_input("Tap to record voice")
    else:
        st.warning("This Streamlit version does not support direct microphone input. Upgrade Streamlit for real-time voice capture.")
        audio_upload = st.file_uploader("Voice Input (fallback upload)", type=["wav", "mp3", "m4a", "flac"])

    example = st.selectbox(
        "Quick Examples",
        [
            "-- select --",
            "My employer has not paid my salary for 3 months",
            "My landlord is illegally evicting me without proper notice",
            "Police arrested me without proper FIR or charges",
        ],
        index=0,
    )
    if example != "-- select --" and not query:
        query = example

    if st.button("Run End-to-End", type="primary"):
        try:
            progress_slot = st.empty()
            progress = progress_slot.progress(0, text="Starting...")

            def _progress_update(pct: int, message: str) -> None:
                progress.progress(pct, text=f"{pct}% - {message}")

            audio_bytes = audio_upload.read() if audio_upload else None
            audio_name = getattr(audio_upload, "name", "") if audio_upload else ""
            result = run_pipeline(
                input_mode=controls["input_mode"],
                query_text=query,
                audio_file=audio_bytes,
                audio_name=audio_name,
                top_k=controls["top_k"],
                advisor_provider=controls["advisor_provider"],
                stt_model_name=controls["stt_model_name"],
                stt_model_dir=controls["stt_model_dir"],
                stt_local_files_only=controls["stt_local_only"],
                enable_tts=controls["enable_tts"],
                tts_engine=controls["tts_engine"],
                tts_fallback_engine=controls["tts_fallback"],
                progress_callback=_progress_update,
            )

            progress.progress(100, text="100% - Completed")
            _render_status_panel(result)
            _render_tabs(result)
        except Exception as exc:
            st.error(f"Error: {exc}")
            st.code(traceback.format_exc())


if __name__ == "__main__":
    main()
