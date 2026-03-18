"""Step 6: Gradio system interface for end-to-end testing.

Flow:
- Text or microphone/upload audio input
- Step 4 transcription (when audio provided)
- Steps 1-3 legal pipeline (classifier + retriever + advisor + safety)
- Step 5 TTS output generation
"""

from __future__ import annotations

import json
import time
import traceback
from pathlib import Path
from typing import Any

import gradio as gr

from agents.langgraph_flow import VeridictionGraph
from audio.transcriber import AudioTranscriber, TranscriberConfig
from tts.speak import TTSConfig, TTSGenerator


APP_CSS = """
:root {
  --bg-soft: #f5f7fb;
  --card-bg: #ffffff;
  --ink: #12203a;
  --muted: #5f6f8a;
  --brand: #0b66ff;
  --brand-2: #0c8f7a;
  --accent: #f3f8ff;
  --border: #d9e2f2;
}

body {
  background: radial-gradient(circle at 0% 0%, #eef5ff 0%, #f8fbff 40%, #f5f7fb 100%);
}

#app-shell {
  max-width: 1380px;
  margin: 0 auto;
}

.hero {
  border: 1px solid var(--border);
  background: linear-gradient(110deg, #0b66ff 0%, #0c8f7a 100%);
  color: #ffffff;
  border-radius: 16px;
  padding: 18px 20px;
  margin-bottom: 14px;
  box-shadow: 0 10px 26px rgba(8, 33, 74, 0.14);
}

.hero h1 {
  margin: 0;
  font-size: 28px;
  line-height: 1.2;
}

.hero p {
  margin: 10px 0 0 0;
  opacity: 0.95;
  font-size: 14px;
}

.card {
  border: 1px solid var(--border);
  background: var(--card-bg);
  border-radius: 14px;
  padding: 10px;
  box-shadow: 0 6px 16px rgba(17, 39, 83, 0.06);
}

.section-title {
  color: var(--ink);
  font-size: 16px;
  font-weight: 700;
  margin: 4px 2px 10px 2px;
}

.subtle {
  color: var(--muted);
  font-size: 13px;
}

.btn-primary {
  background: linear-gradient(90deg, #0b66ff 0%, #004ec9 100%) !important;
  color: #ffffff !important;
  border: none !important;
}

.btn-secondary {
  background: #ffffff !important;
  color: #0b66ff !important;
  border: 1px solid #0b66ff !important;
}

.badge-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(120px, 1fr));
  gap: 8px;
}

.badge-item {
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 8px 10px;
  background: var(--accent);
  color: var(--ink);
  font-size: 13px;
}
"""


class AppServices:
    """Lazy-loaded services for UI actions."""

    def __init__(self) -> None:
        self._flow: VeridictionGraph | None = None
        self._transcriber_cache: dict[tuple[str, str, bool], AudioTranscriber] = {}
        self._tts_cache: dict[tuple[str, str], TTSGenerator] = {}

    def get_flow(self, top_k: int) -> VeridictionGraph:
        if self._flow is None:
            self._flow = VeridictionGraph(top_k=top_k)
        self._flow.top_k = top_k
        return self._flow

    def get_transcriber(self, model_name: str, model_dir: str, local_only: bool) -> AudioTranscriber:
        key = (model_name, model_dir, local_only)
        if key not in self._transcriber_cache:
            self._transcriber_cache[key] = AudioTranscriber(
                config=TranscriberConfig(
                    model_name=model_name,
                    model_dir=model_dir,
                    local_files_only=local_only,
                    language="en",
                    beam_size=5,
                    vad_filter=True,
                )
            )
        return self._transcriber_cache[key]

    def get_tts(self, engine: str, fallback_engine: str) -> TTSGenerator:
        key = (engine, fallback_engine)
        if key not in self._tts_cache:
            self._tts_cache[key] = TTSGenerator(
                config=TTSConfig(
                    preferred_engine=engine,
                    fallback_engine=fallback_engine,
                    output_dir="data/tts",
                )
            )
        return self._tts_cache[key]


SERVICES = AppServices()


def _format_passages_for_table(passages: list[dict[str, Any]]) -> list[list[Any]]:
    rows: list[list[Any]] = []
    for idx, item in enumerate(passages, start=1):
        score = float(item.get("score") or 0.0)
        text = str(item.get("passage", "")).strip().replace("\n", " ")
        preview = text[:260].rstrip() + (" ..." if len(text) > 260 else "")
        source = str(item.get("metadata", {}).get("dataset", ""))
        rows.append([idx, round(score, 4), source, preview])
    return rows


def _build_structured_json(output: dict[str, Any], elapsed_ms: float) -> dict[str, Any]:
    advisor = output.get("advisor", {}) or {}
    safety = output.get("safety", {}) or {}
    passages = output.get("retrieved_passages", []) or []

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
        "advisor": {
            "issue_summary": advisor.get("issue_summary", ""),
            "action_steps": advisor.get("action_steps", []),
            "legal_basis": advisor.get("legal_basis", []),
            "documents_to_collect": advisor.get("documents_to_collect", []),
            "escalation_guidance": advisor.get("escalation_guidance", ""),
        },
        "safety": {
            "risk_flags": safety.get("risk_flags", []),
            "safe_next_steps": safety.get("safe_next_steps", []),
            "disclaimer": safety.get("disclaimer", ""),
        },
        "tts": output.get("tts", {}),
        "final_text": output.get("final_text", ""),
    }
    return structured


def _normalize_audio_mode(input_mode: str, query_text: str, audio_file: str | None) -> tuple[bool, str]:
    mode = (input_mode or "Auto").strip().lower()
    query = (query_text or "").strip()
    has_audio = bool(audio_file)

    if mode == "audio":
        return True, query
    if mode == "text":
        return False, query

    # Auto mode:
    use_audio = has_audio and not query
    if has_audio and query:
        use_audio = True
    return use_audio, query


def run_end_to_end(
    input_mode: str,
    query_text: str,
    audio_file: str | None,
    top_k: int,
    stt_model_name: str,
    stt_model_dir: str,
    stt_local_files_only: bool,
    enable_tts: bool,
    tts_engine: str,
    tts_fallback_engine: str,
) -> tuple[str, str, str, float, str, str, str, list[list[Any]], str, str | None, str]:
    """Main UI callback for complete Veridiction flow."""
    start = time.perf_counter()
    try:
        use_audio, normalized_query = _normalize_audio_mode(input_mode, query_text, audio_file)

        transcript = ""
        audio_meta: dict[str, Any] = {}

        if use_audio:
            if not audio_file:
                raise ValueError("Audio mode selected but no audio input received.")
            transcriber = SERVICES.get_transcriber(
                model_name=stt_model_name,
                model_dir=stt_model_dir,
                local_only=stt_local_files_only,
            )
            transcription = transcriber.transcribe_file(
                audio_path=audio_file,
                language="en",
                beam_size=5,
                vad_filter=True,
            )
            transcript = str(transcription.get("text", "")).strip()
            audio_meta = {
                "audio_file": audio_file,
                "language": transcription.get("language", "en"),
                "language_probability": transcription.get("language_probability", 0.0),
                "duration": transcription.get("duration", 0.0),
                "segments": len(transcription.get("segments", [])),
            }
            if not transcript:
                raise ValueError("Transcription is empty. Try clearer speech and lower background noise.")
        else:
            transcript = normalized_query

        if not transcript:
            raise ValueError("Provide either text query or audio input.")

        flow = SERVICES.get_flow(top_k=top_k)
        output = flow.run(transcript)
        output["input_mode"] = "audio" if use_audio else "text"
        output["transcript"] = transcript
        if audio_meta:
            output["audio_metadata"] = audio_meta

        tts_path: str | None = None
        if enable_tts:
            tts = SERVICES.get_tts(tts_engine, tts_fallback_engine)
            tts_result = tts.speak_to_file(
                text=str(output.get("final_text", "")),
                include_disclaimer=False,
            )
            tts_path = tts_result["audio_path"]
            output["tts"] = {
                "engine": tts_result["engine"],
                "audio_path": tts_result["audio_path"],
                "mime_type": tts_result["mime_type"],
                "size_bytes": tts_result["size_bytes"],
            }

        elapsed_ms = (time.perf_counter() - start) * 1000
        passages = output.get("retrieved_passages", []) or []
        passage_table = _format_passages_for_table(passages)
        structured = _build_structured_json(output, elapsed_ms)

        status = (
            "Run completed successfully. "
            f"Mode={output.get('input_mode', 'text')} | "
            f"Claim={output.get('claim_type', 'other')} | "
            f"Latency={elapsed_ms:.1f} ms"
        )

        return (
            transcript,
            str(output.get("claim_type", "other")),
            str(output.get("urgency", "low")),
            float(output.get("confidence", 0.0)),
            str(output.get("final_text", "")),
            json.dumps(output.get("advisor", {}), ensure_ascii=True, indent=2),
            json.dumps(output.get("safety", {}), ensure_ascii=True, indent=2),
            passage_table,
            json.dumps(structured, ensure_ascii=True, indent=2),
            json.dumps(output, ensure_ascii=True, indent=2),
            tts_path,
            status,
        )
    except Exception as exc:  # pragma: no cover
        err = f"Error: {exc}\n\n{traceback.format_exc()}"
        return (
            "",
            "error",
            "error",
            0.0,
            err,
            "{}",
            "{}",
            [],
            json.dumps({"error": str(exc)}, ensure_ascii=True, indent=2),
            err,
            None,
            "Run failed. See error details in output panes.",
        )


def clear_outputs() -> tuple[str, str, str, float, str, str, str, list[list[Any]], str, str, None, str]:
    return (
        "",
        "",
        "",
        0.0,
        "",
        "{}",
        "{}",
        [],
        "{}",
        "{}",
        None,
        "Cleared.",
    )


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Veridiction Law Assistant - End-to-End MVP", css=APP_CSS) as demo:
        gr.HTML(
            """
            <div id='app-shell'>
              <div class='hero'>
                <h1>Veridiction Law Assistant</h1>
                <p>System Interface: Voice/Text Input -> Legal Pipeline -> Structured Response -> TTS Playback</p>
              </div>
            </div>
            """
        )

        with gr.Row(equal_height=True):
            with gr.Column(scale=4):
                with gr.Group(elem_classes=["card"]):
                    gr.HTML("<div class='section-title'>[I/O] Input Console</div>")
                    input_mode = gr.Radio(
                        choices=["Auto", "Text", "Audio"],
                        value="Auto",
                        label="Input Mode",
                    )
                    query_input = gr.Textbox(
                        label="Text Query",
                        lines=4,
                        placeholder="Example: My employer has not paid my salary for 3 months.",
                    )
                    audio_input = gr.Audio(
                        label="Voice Input (Microphone or Upload)",
                        sources=["microphone", "upload"],
                        type="filepath",
                    )

                with gr.Accordion("[Settings] Advanced Controls", open=False):
                    with gr.Group(elem_classes=["card"]):
                        top_k = gr.Slider(minimum=1, maximum=10, value=5, step=1, label="Retriever Top-K")
                        stt_model_name = gr.Textbox(value="distil-large-v3", label="STT Model Name")
                        stt_model_dir = gr.Textbox(value="data/models/faster-whisper", label="STT Model Cache Directory")
                        stt_local_files_only = gr.Checkbox(value=True, label="STT Local Files Only")
                        enable_tts = gr.Checkbox(value=True, label="Enable TTS")
                        tts_engine = gr.Dropdown(
                            choices=["edge_tts", "pyttsx3"],
                            value="edge_tts",
                            label="TTS Engine",
                        )
                        tts_fallback_engine = gr.Dropdown(
                            choices=["pyttsx3", "edge_tts"],
                            value="pyttsx3",
                            label="TTS Fallback Engine",
                        )

                with gr.Row():
                    run_btn = gr.Button("Run End-to-End", variant="primary", elem_classes=["btn-primary"])
                    clear_btn = gr.Button("Clear Outputs", variant="secondary", elem_classes=["btn-secondary"])

                gr.Examples(
                    examples=[
                        ["Auto", "My employer has not paid my salary for 3 months", None, 5, "distil-large-v3", "data/models/faster-whisper", True, True, "edge_tts", "pyttsx3"],
                        ["Auto", "My landlord is illegally evicting me without proper notice", None, 5, "distil-large-v3", "data/models/faster-whisper", True, True, "edge_tts", "pyttsx3"],
                        ["Auto", "Police arrested me without proper FIR or charges", None, 5, "distil-large-v3", "data/models/faster-whisper", True, True, "edge_tts", "pyttsx3"],
                    ],
                    inputs=[
                        input_mode,
                        query_input,
                        audio_input,
                        top_k,
                        stt_model_name,
                        stt_model_dir,
                        stt_local_files_only,
                        enable_tts,
                        tts_engine,
                        tts_fallback_engine,
                    ],
                    label="Quick Examples",
                )

            with gr.Column(scale=8):
                with gr.Group(elem_classes=["card"]):
                    gr.HTML("<div class='section-title'>[Status] Run Summary</div>")
                    status_out = gr.Textbox(label="Execution Status", lines=2)
                    with gr.Row():
                        claim_type_out = gr.Textbox(label="Claim Type")
                        urgency_out = gr.Textbox(label="Urgency")
                        confidence_out = gr.Number(label="Confidence")

                with gr.Tab("Overview"):
                    with gr.Group(elem_classes=["card"]):
                        transcript_out = gr.Textbox(label="Transcript / Final Query", lines=4)
                        final_text_out = gr.Textbox(label="Final Advisor Text", lines=12)
                        with gr.Row():
                            advisor_json_out = gr.Code(label="Advisor JSON", language="json")
                            safety_json_out = gr.Code(label="Safety JSON", language="json")

                with gr.Tab("Retrieval"):
                    with gr.Group(elem_classes=["card"]):
                        passages_table_out = gr.Dataframe(
                            headers=["Rank", "Score", "Source", "Passage Preview"],
                            datatype=["number", "number", "str", "str"],
                            wrap=True,
                            label="Retrieved Passages",
                        )

                with gr.Tab("Structured JSON"):
                    with gr.Group(elem_classes=["card"]):
                        structured_json_out = gr.Code(label="Structured Response JSON", language="json")
                        raw_json_out = gr.Code(label="Raw Pipeline JSON", language="json")

                with gr.Tab("Audio Output"):
                    with gr.Group(elem_classes=["card"]):
                        tts_audio_out = gr.Audio(label="TTS Playback", type="filepath", interactive=False)
                        gr.HTML(
                            """
                            <div class='subtle'>
                              TTS is generated from <b>final_text</b> and includes legal disclaimer handling.
                            </div>
                            """
                        )

        run_btn.click(
            fn=run_end_to_end,
            inputs=[
                input_mode,
                query_input,
                audio_input,
                top_k,
                stt_model_name,
                stt_model_dir,
                stt_local_files_only,
                enable_tts,
                tts_engine,
                tts_fallback_engine,
            ],
            outputs=[
                transcript_out,
                claim_type_out,
                urgency_out,
                confidence_out,
                final_text_out,
                advisor_json_out,
                safety_json_out,
                passages_table_out,
                structured_json_out,
                raw_json_out,
                tts_audio_out,
                status_out,
            ],
        )

        clear_btn.click(
            fn=clear_outputs,
            outputs=[
                transcript_out,
                claim_type_out,
                urgency_out,
                confidence_out,
                final_text_out,
                advisor_json_out,
                safety_json_out,
                passages_table_out,
                structured_json_out,
                raw_json_out,
                tts_audio_out,
                status_out,
            ],
        )

    return demo


def main() -> None:
    Path("data/tts").mkdir(parents=True, exist_ok=True)
    app = build_app()
    app.queue()
    app.launch(server_name="127.0.0.1", server_port=7860, share=False)


if __name__ == "__main__":
    main()
