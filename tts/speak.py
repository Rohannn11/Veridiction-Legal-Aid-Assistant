"""Step 5: Text-to-speech utility (English MVP).

Primary path:
- edge-tts (free, high quality, requires network)
Fallback path:
- pyttsx3 (offline local TTS)

Returns playable audio file path and bytes for Gradio Audio output.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_DISCLAIMER = (
    "This is NOT legal advice. This is an AI research prototype. "
    "Please consult a qualified lawyer immediately."
)


@dataclass(slots=True)
class TTSConfig:
    """Configuration for TTS synthesis."""

    preferred_engine: str = "edge_tts"
    fallback_engine: str = "pyttsx3"
    edge_voice: str = "en-IN-NeerjaNeural"
    edge_rate: str = "+0%"
    max_chars: int = 3500
    output_dir: str = "data/tts"


class TTSError(RuntimeError):
    """Raised when text-to-speech synthesis fails."""


def normalize_tts_text(text: str, max_chars: int = 3500) -> str:
    """Normalize text for safer speech synthesis.

    Steps:
    - Remove markdown and code fences.
    - Remove control chars.
    - Collapse excessive whitespace.
    - Enforce max length.
    """
    if not text or not text.strip():
        raise ValueError("TTS input text is empty")

    normalized = text
    normalized = re.sub(r"```[\s\S]*?```", " ", normalized)
    normalized = re.sub(r"`([^`]*)`", r"\1", normalized)
    normalized = re.sub(r"\[([^\]]+)\]\(([^\)]+)\)", r"\1", normalized)
    normalized = re.sub(r"[*_~#>|]", " ", normalized)
    normalized = re.sub(r"[\x00-\x1F\x7F]", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()

    if len(normalized) > max_chars:
        normalized = normalized[:max_chars].rstrip() + " ..."

    if not normalized:
        raise ValueError("TTS input text became empty after normalization")

    return normalized


class TTSGenerator:
    """English MVP TTS generator with free engine + offline fallback."""

    def __init__(self, config: TTSConfig | None = None) -> None:
        self.config = config or TTSConfig()
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    def speak_to_file(
        self,
        text: str,
        output_path: str | Path | None = None,
        include_disclaimer: bool = True,
    ) -> dict[str, Any]:
        """Synthesize text to audio file and return artifact metadata."""
        final_text = text.strip()
        if include_disclaimer and DEFAULT_DISCLAIMER.lower() not in final_text.lower():
            final_text = f"{final_text}\n\n{DEFAULT_DISCLAIMER}"

        normalized = normalize_tts_text(final_text, max_chars=self.config.max_chars)

        if output_path is None:
            name = f"speech_{uuid.uuid4().hex}.mp3"
            out_path = Path(self.config.output_dir) / name
        else:
            out_path = Path(output_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)

        primary = self.config.preferred_engine
        fallback = self.config.fallback_engine

        try:
            if primary == "edge_tts":
                self._synthesize_edge_tts(normalized, out_path)
                engine_used = "edge_tts"
            elif primary == "pyttsx3":
                out_path = self._synthesize_pyttsx3(normalized, out_path)
                engine_used = "pyttsx3"
            else:
                raise TTSError(f"Unsupported preferred engine: {primary}")
        except Exception as primary_exc:
            if fallback == "pyttsx3" and primary != "pyttsx3":
                out_path = self._synthesize_pyttsx3(normalized, out_path)
                engine_used = "pyttsx3"
            else:
                raise TTSError(f"Primary TTS failed: {primary_exc}") from primary_exc

        audio_bytes = out_path.read_bytes()
        mime = "audio/mpeg" if out_path.suffix.lower() == ".mp3" else "audio/wav"

        return {
            "engine": engine_used,
            "text": normalized,
            "audio_path": str(out_path),
            "audio_bytes": audio_bytes,
            "mime_type": mime,
            "size_bytes": len(audio_bytes),
        }

    def speak_to_bytes(self, text: str, include_disclaimer: bool = True) -> dict[str, Any]:
        """Synthesize text and return bytes-friendly response for Gradio."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
            temp_path = Path(tmp.name)

        result = self.speak_to_file(text=text, output_path=temp_path, include_disclaimer=include_disclaimer)
        return result

    def _synthesize_edge_tts(self, text: str, output_path: Path) -> None:
        try:
            import edge_tts
        except ImportError as exc:
            raise TTSError("edge-tts is not installed. Run: pip install edge-tts") from exc

        if output_path.suffix.lower() != ".mp3":
            output_path = output_path.with_suffix(".mp3")

        async def _run() -> None:
            communicate = edge_tts.Communicate(text=text, voice=self.config.edge_voice, rate=self.config.edge_rate)
            await communicate.save(str(output_path))

        asyncio.run(_run())

    def _synthesize_pyttsx3(self, text: str, output_path: Path) -> Path:
        try:
            import pyttsx3  # type: ignore[import-not-found]
        except ImportError as exc:
            raise TTSError(
                "pyttsx3 is not installed for offline fallback. Run: pip install pyttsx3"
            ) from exc

        final_path = output_path if output_path.suffix.lower() == ".wav" else output_path.with_suffix(".wav")
        engine = pyttsx3.init()
        engine.save_to_file(text, str(final_path))
        engine.runAndWait()

        if not final_path.exists() or final_path.stat().st_size == 0:
            raise TTSError("pyttsx3 produced empty audio output")

        return final_path


def _build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Step 5 TTS utility")
    parser.add_argument("--text", type=str, default=None, help="Text to synthesize")
    parser.add_argument("--text-file", type=str, default=None, help="File containing text to synthesize")
    parser.add_argument("--output", type=str, default=None, help="Audio output file (.mp3 or .wav)")
    parser.add_argument("--engine", type=str, default="edge_tts", choices=["edge_tts", "pyttsx3"])
    parser.add_argument("--fallback-engine", type=str, default="pyttsx3", choices=["pyttsx3", "edge_tts"])
    parser.add_argument("--voice", type=str, default="en-IN-NeerjaNeural")
    parser.add_argument("--rate", type=str, default="+0%")
    parser.add_argument("--max-chars", type=int, default=3500)
    parser.add_argument("--no-disclaimer", action="store_true", help="Do not append legal disclaimer")
    return parser


def _read_input_text(args: argparse.Namespace) -> str:
    if args.text:
        return args.text
    if args.text_file:
        return Path(args.text_file).read_text(encoding="utf-8")
    raise ValueError("Provide --text or --text-file")


def main() -> None:
    args = _build_cli().parse_args()
    input_text = _read_input_text(args)

    tts = TTSGenerator(
        config=TTSConfig(
            preferred_engine=args.engine,
            fallback_engine=args.fallback_engine,
            edge_voice=args.voice,
            edge_rate=args.rate,
            max_chars=args.max_chars,
            output_dir="data/tts",
        )
    )

    result = tts.speak_to_file(
        text=input_text,
        output_path=args.output,
        include_disclaimer=not args.no_disclaimer,
    )

    print(
        json.dumps(
            {
                "engine": result["engine"],
                "audio_path": result["audio_path"],
                "mime_type": result["mime_type"],
                "size_bytes": result["size_bytes"],
                "text_preview": result["text"][:180],
            },
            ensure_ascii=True,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
