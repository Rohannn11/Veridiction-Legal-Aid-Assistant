"""Step 6: Gradio system interface for end-to-end testing.

Flow:
- Text or microphone/upload audio input
- Step 4 transcription (when audio provided)
- Steps 1-3 legal pipeline (classifier + retriever + advisor + safety)
- Step 5 TTS output generation
"""

from __future__ import annotations

import json
import traceback
from pathlib import Path
from typing import Any

import gradio as gr

from agents.langgraph_flow import VeridictionGraph
from audio.transcriber import AudioTranscriber, TranscriberConfig
from tts.speak import TTSConfig, TTSGenerator


class AppServices:
    """Lazy-loaded services for UI actions."""

    def __init__(self) -> None:
        self._flow: VeridictionGraph | None = None
        self._transcriber_cache: dict[tuple[str, str, bool], AudioTranscriber] = {}
        self._tts_cache: dict[str, TTSGenerator] = {}

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

    def get_tts(self, engine: str) -> TTSGenerator:
        if engine not in self._tts_cache:
            self._tts_cache[engine] = TTSGenerator(
                config=TTSConfig(
                    preferred_engine=engine,
                    fallback_engine="pyttsx3" if engine == "edge_tts" else "edge_tts",
                    output_dir="data/tts",
                )
            )
        return self._tts_cache[engine]


SERVICES = AppServices()


def _format_passages(passages: list[dict[str, Any]]) -> str:
    if not passages:
        return "No passages retrieved."

    blocks: list[str] = []
    for idx, item in enumerate(passages, start=1):
        score = item.get("score", 0.0)
        text = str(item.get("passage", "")).strip().replace("\n", " ")
        if len(text) > 600:
            text = text[:600].rstrip() + " ..."
        blocks.append(f"{idx}. Score: {score:.4f}\n{text}")
    return "\n\n".join(blocks)


def run_end_to_end(
    query_text: str,
    audio_file: str | None,
    prefer_audio: bool,
    top_k: int,
    model_name: str,
    model_dir: str,
    local_files_only: bool,
    enable_tts: bool,
    tts_engine: str,
) -> tuple[str, str, str, float, str, str, str, str | None]:
    """Main UI callback for complete Veridiction flow."""
    try:
        transcript = ""
        audio_meta: dict[str, Any] = {}

        use_audio = bool(audio_file) and (prefer_audio or not query_text.strip())
        if use_audio:
            transcriber = SERVICES.get_transcriber(
                model_name=model_name,
                model_dir=model_dir,
                local_only=local_files_only,
            )
            transcription = transcriber.transcribe_file(audio_file, language="en", beam_size=5, vad_filter=True)
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
            transcript = query_text.strip()

        if not transcript:
            raise ValueError("Please enter text query or provide microphone/upload audio.")

        flow = SERVICES.get_flow(top_k=top_k)
        output = flow.run(transcript)
        output["input_mode"] = "audio" if use_audio else "text"
        output["transcript"] = transcript
        if audio_meta:
            output["audio_metadata"] = audio_meta

        tts_path: str | None = None
        if enable_tts:
            tts = SERVICES.get_tts(tts_engine)
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

        claim_type = str(output.get("claim_type", "other"))
        urgency = str(output.get("urgency", "low"))
        confidence = float(output.get("confidence", 0.0))
        final_text = str(output.get("final_text", ""))
        passages_text = _format_passages(output.get("retrieved_passages", []))
        raw_json = json.dumps(output, ensure_ascii=True, indent=2)

        return (
            transcript,
            claim_type,
            urgency,
            confidence,
            final_text,
            passages_text,
            raw_json,
            tts_path,
        )
    except Exception as exc:  # pragma: no cover
        err = f"Error: {exc}\n\n{traceback.format_exc()}"
        return ("", "error", "error", 0.0, err, "", err, None)


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Veridiction Law Assistant - End-to-End MVP") as demo:
        gr.Markdown("# Veridiction Law Assistant - System Interface")
        gr.Markdown(
            "Text or voice input -> legal pipeline -> optional TTS output. "
            "Use this to test Step 4, Step 5, and end-to-end system behavior."
        )

        with gr.Row():
            query_input = gr.Textbox(
                label="Text Query (optional if using voice)",
                lines=4,
                placeholder="Example: My employer has not paid my salary for 3 months.",
            )

        with gr.Row():
            audio_input = gr.Audio(
                label="Voice Input (microphone/upload)",
                sources=["microphone", "upload"],
                type="filepath",
            )

        with gr.Row():
            prefer_audio = gr.Checkbox(value=True, label="Prefer audio when both text and audio are present")
            top_k = gr.Slider(minimum=1, maximum=10, value=5, step=1, label="Retriever Top-K")

        with gr.Accordion("Advanced", open=False):
            with gr.Row():
                model_name = gr.Textbox(value="distil-large-v3", label="STT Model Name")
                model_dir = gr.Textbox(value="data/models/faster-whisper", label="STT Model Cache Directory")
                local_files_only = gr.Checkbox(value=True, label="STT Local Files Only")
            with gr.Row():
                enable_tts = gr.Checkbox(value=True, label="Enable TTS Output")
                tts_engine = gr.Dropdown(choices=["edge_tts", "pyttsx3"], value="edge_tts", label="TTS Engine")

        run_btn = gr.Button("Run End-to-End")

        with gr.Row():
            transcript_out = gr.Textbox(label="Transcript/Final Query", lines=4)
            final_text_out = gr.Textbox(label="Advisor Final Response", lines=12)

        with gr.Row():
            claim_type_out = gr.Textbox(label="Claim Type")
            urgency_out = gr.Textbox(label="Urgency")
            confidence_out = gr.Number(label="Confidence")

        passages_out = gr.Textbox(label="Retrieved Passages", lines=12)
        raw_json_out = gr.Textbox(label="Raw JSON Output", lines=18)
        tts_audio_out = gr.Audio(label="TTS Audio Output", type="filepath", interactive=False)

        run_btn.click(
            fn=run_end_to_end,
            inputs=[
                query_input,
                audio_input,
                prefer_audio,
                top_k,
                model_name,
                model_dir,
                local_files_only,
                enable_tts,
                tts_engine,
            ],
            outputs=[
                transcript_out,
                claim_type_out,
                urgency_out,
                confidence_out,
                final_text_out,
                passages_out,
                raw_json_out,
                tts_audio_out,
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
