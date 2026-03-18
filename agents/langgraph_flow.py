"""Step 3: Veridiction LangGraph flow (Retriever -> Advisor -> Safety).

English-first implementation with structured outputs and local model fallback.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field, ValidationError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from nlp.classifier import ClaimClassifier
from rag.retriever import LegalRetriever


LOGGER = logging.getLogger(__name__)

MANDATORY_DISCLAIMER = (
    "⚠️ This is NOT legal advice. This is an AI research prototype. "
    "Please consult a qualified lawyer immediately."
)


class AdvisorOutput(BaseModel):
    """Structured payload from the advisor node."""

    issue_summary: str = Field(description="One short summary of the user's legal issue")
    action_steps: list[str] = Field(description="Step-by-step practical next actions")
    legal_basis: list[str] = Field(description="Grounding points from retrieved passages")
    documents_to_collect: list[str] = Field(description="Evidence/documents user should gather")
    escalation_guidance: str = Field(description="When to escalate to lawyer/police/authority")


class SafetyOutput(BaseModel):
    """Structured payload from the safety node."""

    risk_flags: list[str] = Field(default_factory=list)
    safe_next_steps: list[str] = Field(default_factory=list)
    disclaimer: str = Field(default=MANDATORY_DISCLAIMER)


class VeridictionState(TypedDict, total=False):
    """State object passed across graph nodes."""

    user_query: str
    claim: dict[str, Any]
    retrieved_passages: list[dict[str, Any]]
    advisor_output: dict[str, Any]
    safety_output: dict[str, Any]
    final_response: dict[str, Any]
    error: str


class LocalAdvisor:
    """Local LLM advisor with robust fallback when model runtime is unavailable."""

    def __init__(self, model_id: str = "meta-llama/Llama-3.2-3B-Instruct") -> None:
        self.model_id = model_id
        self._generator = None

    def _ensure_generator(self) -> None:
        if self._generator is not None:
            return
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline

            quant_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )
            tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                device_map="auto",
                quantization_config=quant_cfg,
            )
            self._generator = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
            )
            LOGGER.info("Loaded advisor LLM in 4-bit mode: %s", self.model_id)
        except Exception as exc:
            LOGGER.warning("Advisor LLM unavailable; using deterministic fallback: %s", exc)
            self._generator = False

    def generate(self, query: str, claim: dict[str, Any], passages: list[dict[str, Any]]) -> AdvisorOutput:
        """Generate structured advisor output from query and retrieved passages."""
        self._ensure_generator()
        if self._generator is False:
            return self._fallback_advice(query=query, claim=claim, passages=passages)

        prompt = self._build_prompt(query=query, claim=claim, passages=passages)
        try:
            raw = self._generator(
                prompt,
                max_new_tokens=450,
                do_sample=False,
                temperature=0.1,
                repetition_penalty=1.05,
            )[0]["generated_text"]
            parsed = self._extract_json(raw)
            return AdvisorOutput.model_validate(parsed)
        except (ValidationError, ValueError, KeyError, IndexError) as exc:
            LOGGER.warning("LLM output parse failed; using fallback advice: %s", exc)
            return self._fallback_advice(query=query, claim=claim, passages=passages)

    def _build_prompt(self, query: str, claim: dict[str, Any], passages: list[dict[str, Any]]) -> str:
        snippets: list[str] = []
        for idx, item in enumerate(passages[:5], start=1):
            passage = item.get("passage", "")
            score = item.get("score")
            snippets.append(f"[{idx}] score={score}: {passage[:380]}")
        context_block = "\n".join(snippets)

        return (
            "You are an English-only legal research assistant for India. "
            "Use retrieved context conservatively and avoid hallucination.\n\n"
            f"User query: {query}\n"
            f"Claim JSON: {json.dumps(claim, ensure_ascii=True)}\n\n"
            "Retrieved passages:\n"
            f"{context_block}\n\n"
            "Return STRICT JSON only with keys: "
            "issue_summary, action_steps, legal_basis, documents_to_collect, escalation_guidance."
        )

    def _extract_json(self, text: str) -> dict[str, Any]:
        block = re.search(r"\{[\s\S]*\}", text)
        if not block:
            raise ValueError("No JSON object found in advisor output")
        return json.loads(block.group(0))

    def _fallback_advice(self, query: str, claim: dict[str, Any], passages: list[dict[str, Any]]) -> AdvisorOutput:
        claim_type = claim.get("claim_type", "other")
        urgency = claim.get("urgency", "medium")
        legal_basis = [item.get("passage", "")[:220] for item in passages[:3] if item.get("passage")]

        action_steps = [
            "Write a dated incident summary in simple English with exact timeline.",
            "Collect primary evidence: IDs, contracts, salary slips, bills, messages, photos, or FIR details.",
            "Visit nearest legal aid clinic or District Legal Services Authority with documents.",
            "File a formal written complaint with acknowledgement receipt.",
        ]
        if claim_type == "unpaid_wages":
            action_steps.insert(2, "Prepare wage calculation sheet month-wise and attach proof of work.")
        if claim_type == "police_harassment":
            action_steps.insert(0, "Document officer name, station, date/time, and any witnesses immediately.")

        documents = [
            "Government ID proof",
            "Address proof",
            "Any agreements or appointment letter",
            "Payment or bank transaction proof",
            "Messages/recordings/photos relevant to the incident",
        ]

        escalation = "If there is immediate danger, contact emergency services and a qualified lawyer without delay."
        if urgency == "medium":
            escalation = "Escalate to a qualified lawyer within 24-48 hours if no response from authority."

        return AdvisorOutput(
            issue_summary=f"Likely category: {claim_type}. User reports: {query[:120]}",
            action_steps=action_steps,
            legal_basis=legal_basis or ["Limited retrieved support. Re-run retrieval with refined query."],
            documents_to_collect=documents,
            escalation_guidance=escalation,
        )


class VeridictionGraph:
    """End-to-end Step 3 graph orchestration."""

    def __init__(self, top_k: int = 5) -> None:
        self.top_k = top_k
        self.classifier = ClaimClassifier()
        self.retriever = LegalRetriever()
        self.advisor = LocalAdvisor()
        self.graph = self._build_graph()

    def _build_graph(self):
        builder = StateGraph(VeridictionState)
        builder.add_node("retriever", self.retriever_node)
        builder.add_node("advisor", self.advisor_node)
        builder.add_node("safety", self.safety_node)
        builder.add_edge(START, "retriever")
        builder.add_edge("retriever", "advisor")
        builder.add_edge("advisor", "safety")
        builder.add_edge("safety", END)
        return builder.compile()

    def retriever_node(self, state: VeridictionState) -> VeridictionState:
        query = state.get("user_query", "")
        claim = self.classifier.classify(query)
        passages = self.retriever.query(query, top_k=self.top_k)
        return {"claim": claim, "retrieved_passages": passages}

    def advisor_node(self, state: VeridictionState) -> VeridictionState:
        query = state.get("user_query", "")
        claim = state.get("claim", {})
        passages = state.get("retrieved_passages", [])
        advisor_output = self.advisor.generate(query=query, claim=claim, passages=passages)
        return {"advisor_output": advisor_output.model_dump()}

    def safety_node(self, state: VeridictionState) -> VeridictionState:
        query = state.get("user_query", "").lower()
        claim = state.get("claim", {})
        advisor_output = state.get("advisor_output", {})
        risk_flags = self._risk_flags(query=query, claim=claim)

        safe_steps: list[str] = []
        if "immediate_danger" in risk_flags:
            safe_steps.append("Seek immediate physical safety and contact emergency services.")
        if "police_misconduct" in risk_flags:
            safe_steps.append("Approach higher police authority or legal aid body with written complaint.")
        if not safe_steps:
            safe_steps.append("Follow documented steps and seek legal aid for case-specific guidance.")

        safety_output = SafetyOutput(
            risk_flags=risk_flags,
            safe_next_steps=safe_steps,
            disclaimer=MANDATORY_DISCLAIMER,
        )

        final_response = {
            "claim_type": claim.get("claim_type", "other"),
            "urgency": claim.get("urgency", "low"),
            "confidence": claim.get("confidence", 0.0),
            "retrieved_passages": state.get("retrieved_passages", []),
            "advisor": advisor_output,
            "safety": safety_output.model_dump(),
            "final_text": self._compose_final_text(advisor_output=advisor_output, safety=safety_output),
        }
        return {"safety_output": safety_output.model_dump(), "final_response": final_response}

    def _risk_flags(self, query: str, claim: dict[str, Any]) -> list[str]:
        flags: list[str] = []
        if any(token in query for token in ("danger", "assault", "violence", "threat", "urgent", "tonight")):
            flags.append("immediate_danger")
        if claim.get("claim_type") == "police_harassment":
            flags.append("police_misconduct")
        if claim.get("claim_type") == "domestic_violence":
            flags.append("domestic_violence_risk")
        if claim.get("urgency") == "high" and "high_urgency" not in flags:
            flags.append("high_urgency")
        return flags

    def _compose_final_text(self, advisor_output: dict[str, Any], safety: SafetyOutput) -> str:
        lines: list[str] = []
        summary = advisor_output.get("issue_summary", "")
        if summary:
            lines.append(f"Issue summary: {summary}")

        steps = advisor_output.get("action_steps", [])
        if steps:
            lines.append("Recommended steps:")
            for idx, step in enumerate(steps, start=1):
                lines.append(f"{idx}. {step}")

        escalation = advisor_output.get("escalation_guidance", "")
        if escalation:
            lines.append(f"Escalation: {escalation}")

        if safety.risk_flags:
            lines.append(f"Risk flags: {', '.join(safety.risk_flags)}")
        lines.append(safety.disclaimer)
        return "\n".join(lines)

    def run(self, user_query: str) -> dict[str, Any]:
        initial_state: VeridictionState = {"user_query": user_query}
        result = self.graph.invoke(initial_state)
        return result.get("final_response", {})


def _build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Veridiction Step 3 LangGraph flow")
    parser.add_argument("--query", type=str, default=None)
    parser.add_argument("--audio-file", type=str, default=None, help="Optional audio file input for Step 4")
    parser.add_argument("--live-mic", action="store_true", help="Capture real-time microphone input")
    parser.add_argument(
        "--record-seconds",
        type=int,
        default=0,
        help="If > 0, records microphone audio and transcribes before running flow",
    )
    parser.add_argument("--record-out", type=str, default="data/audio_recorded.wav")
    parser.add_argument("--audio-max-seconds", type=int, default=30)
    parser.add_argument("--audio-silence-threshold", type=float, default=0.01)
    parser.add_argument("--audio-silence-seconds", type=float, default=2.0)
    parser.add_argument("--audio-model-name", type=str, default="distil-large-v3")
    parser.add_argument("--audio-model-dir", type=str, default="data/models/faster-whisper")
    parser.add_argument("--audio-local-files-only", action="store_true")
    parser.add_argument("--audio-language", type=str, default="en")
    parser.add_argument("--audio-beam-size", type=int, default=5)
    parser.add_argument("--audio-no-vad", action="store_true")
    parser.add_argument("--enable-tts", action="store_true", help="Generate Step 5 TTS audio output")
    parser.add_argument("--tts-output", type=str, default="data/tts/final_response.mp3")
    parser.add_argument("--tts-engine", type=str, default="edge_tts", choices=["edge_tts", "pyttsx3"])
    parser.add_argument("--tts-fallback-engine", type=str, default="pyttsx3", choices=["pyttsx3", "edge_tts"])
    parser.add_argument("--top-k", type=int, default=5)
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = _build_cli().parse_args()

    query_text = args.query
    audio_metadata: dict[str, Any] | None = None

    if args.live_mic or args.audio_file or args.record_seconds > 0:
        from audio.transcriber import (
            AudioTranscriber,
            TranscriberConfig,
            record_microphone_live_to_wav,
            record_microphone_to_wav,
        )

        input_audio = args.audio_file
        if args.live_mic:
            recorded_file = record_microphone_live_to_wav(
                output_wav=args.record_out,
                sample_rate=16000,
                channels=1,
                max_seconds=args.audio_max_seconds,
                silence_threshold=args.audio_silence_threshold,
                silence_seconds=args.audio_silence_seconds,
                enable_enter_to_stop=True,
            )
            input_audio = str(recorded_file)
        elif args.record_seconds > 0:
            recorded_file = record_microphone_to_wav(
                output_wav=args.record_out,
                duration_seconds=args.record_seconds,
                sample_rate=16000,
                channels=1,
            )
            input_audio = str(recorded_file)

        if not input_audio:
            raise ValueError("Audio mode requires --live-mic, --audio-file, or --record-seconds > 0")

        transcriber = AudioTranscriber(
            config=TranscriberConfig(
                model_name=args.audio_model_name,
                model_dir=args.audio_model_dir,
                local_files_only=args.audio_local_files_only,
                language=args.audio_language,
                beam_size=args.audio_beam_size,
                vad_filter=not args.audio_no_vad,
            )
        )
        transcription = transcriber.transcribe_file(
            audio_path=input_audio,
            language=args.audio_language,
            beam_size=args.audio_beam_size,
            vad_filter=not args.audio_no_vad,
        )
        query_text = transcription.get("text", "").strip()
        audio_metadata = {
            "audio_file": input_audio,
            "language": transcription.get("language", args.audio_language),
            "language_probability": transcription.get("language_probability", 0.0),
            "duration": transcription.get("duration", 0.0),
            "segments": len(transcription.get("segments", [])),
        }

    if audio_metadata is not None and not query_text:
        raise ValueError("Transcription produced empty text. Try clearer audio or disable background noise.")

    if not query_text:
        query_text = "My employer has not paid my salary for 3 months."

    flow = VeridictionGraph(top_k=args.top_k)
    output = flow.run(query_text)

    if args.enable_tts:
        from tts.speak import TTSConfig, TTSGenerator

        tts_generator = TTSGenerator(
            config=TTSConfig(
                preferred_engine=args.tts_engine,
                fallback_engine=args.tts_fallback_engine,
                output_dir="data/tts",
            )
        )
        tts_result = tts_generator.speak_to_file(
            text=output.get("final_text", ""),
            output_path=args.tts_output,
            include_disclaimer=False,
        )
        output["tts"] = {
            "engine": tts_result["engine"],
            "audio_path": tts_result["audio_path"],
            "mime_type": tts_result["mime_type"],
            "size_bytes": tts_result["size_bytes"],
        }

    if audio_metadata is not None:
        output["input_mode"] = "audio"
        output["transcript"] = query_text
        output["audio_metadata"] = audio_metadata
    else:
        output["input_mode"] = "text"
    print(json.dumps(output, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
