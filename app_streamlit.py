"""Streamlit system interface for Veridiction (text/audio -> legal pipeline -> structured output -> TTS).

This mirrors the Gradio UI functionality without the Mermaid graph. A richer graph
renderer will be added in the next step. Current scope: end-to-end run, safety,
structured sections, retrieval table, audio playback, and health checks.
"""

from __future__ import annotations

import json
import os
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

    flow_lines = [f"Step {s.get('step', '')}: {s.get('title', '')} - {s.get('details', '')}" for s in flowchart]

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
        "tts_summary_text": tts_summary_text,
        "safety_json": json.dumps(safety, ensure_ascii=True, indent=2),
        "passage_table": passage_table,
        "structured_json": json.dumps(structured, ensure_ascii=True, indent=2),
        "raw_json": json.dumps(output, ensure_ascii=True, indent=2),
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
        "Flowchart (text only)",
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
        st.text_area("Flowchart (Text Steps)", result.get("flowchart_text", ""), height=200)
        st.info("Graphical flowchart coming in Step 2 (pyvis/Plotly replacement for Mermaid).")

    with tabs[3]:
        st.code(result.get("safety_json", "{}"), language="json")

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
