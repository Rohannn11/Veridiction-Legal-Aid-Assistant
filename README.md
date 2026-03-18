# Indian-Law-Assistant-GenAI

## Step 4: Audio Transcription (faster-whisper distil-large-v3)

Implemented module: `audio/transcriber.py`

Features:
- Audio file transcription using faster-whisper `distil-large-v3`
- Live microphone input (real-time speaking) with stop by Enter/silence/max duration
- Optional fixed-duration microphone recording to WAV and immediate transcription
- Integrated with pipeline entrypoint `agents/langgraph_flow.py`

### Run Step 4 standalone

Transcribe existing audio file:

```powershell
C:/Users/rohan/miniconda3/envs/veridiction/python.exe audio/transcriber.py --audio-file data/sample.wav --language en --json-out data/transcription_result.json
```

Record from mic for 8 seconds and transcribe:

```powershell
C:/Users/rohan/miniconda3/envs/veridiction/python.exe audio/transcriber.py --record-seconds 8 --record-out data/audio_recorded.wav --language en
```

Live microphone mode (speak directly, stop by Enter or silence):

```powershell
C:/Users/rohan/miniconda3/envs/veridiction/python.exe audio/transcriber.py --live-mic --record-out data/audio_live.wav --max-seconds 30 --silence-seconds 2.0 --language en
```

First run recommendation (pre-download model once):

```powershell
C:/Users/rohan/miniconda3/envs/veridiction/python.exe audio/transcriber.py --download-only --model-name distil-large-v3 --model-dir data/models/faster-whisper
```

Then run live mode fully offline from local cache:

```powershell
C:/Users/rohan/miniconda3/envs/veridiction/python.exe audio/transcriber.py --live-mic --local-files-only --model-dir data/models/faster-whisper --record-out data/audio_live.wav --max-seconds 30 --silence-seconds 2.0 --language en
```

### Run full pipeline with audio input (Step 4 -> Step 3)

Use audio file:

```powershell
C:/Users/rohan/miniconda3/envs/veridiction/python.exe agents/langgraph_flow.py --audio-file data/sample.wav --audio-language en --top-k 5
```

Record audio and run end-to-end:

```powershell
C:/Users/rohan/miniconda3/envs/veridiction/python.exe agents/langgraph_flow.py --record-seconds 8 --record-out data/audio_recorded.wav --audio-language en --top-k 5
```

Live voice end-to-end (Step 4 -> Step 3):

```powershell
C:/Users/rohan/miniconda3/envs/veridiction/python.exe agents/langgraph_flow.py --live-mic --record-out data/audio_live.wav --audio-max-seconds 30 --audio-silence-seconds 2.0 --audio-language en --top-k 5
```

If model is already downloaded, force local cache use:

```powershell
C:/Users/rohan/miniconda3/envs/veridiction/python.exe agents/langgraph_flow.py --live-mic --audio-model-dir data/models/faster-whisper --audio-local-files-only --record-out data/audio_live.wav --audio-max-seconds 30 --audio-silence-seconds 2.0 --audio-language en --top-k 5
```

## Step 6: Gradio UI (End-to-End System Interface)

Launch UI:

```powershell
C:/Users/rohan/miniconda3/envs/veridiction/python.exe app_gradio.py
```

Open in browser:

```text
http://127.0.0.1:7860
```

Quick test flow in UI:
1. Provide microphone input (or text input).
2. Click `Run End-to-End`.
3. Verify outputs:
	- transcript/final query
	- claim type, urgency, confidence
	- retrieved passages
	- advisor final response
	- TTS audio player output

Expected output now includes:
- `input_mode` (`audio` or `text`)
- `transcript` (when audio mode is used)
- `audio_metadata` (language, confidence, duration, segments)

## Step 4 Testing Checklist

1. Transcription quality test:
	- Run standalone transcription on a clear 5-10 second English clip.
	- Confirm `text` is accurate and non-empty.
2. Integration test:
	- Run `agents/langgraph_flow.py` with `--audio-file`.
	- Confirm valid `claim_type`, `retrieved_passages`, and advisor output.
3. Microphone test:
	- Run with `--record-seconds 8`.
	- Confirm recording file is created and transcript appears in output.
4. Noise robustness test:
	- Test one noisy clip and one clean clip.
	- Compare transcript quality and downstream classification consistency.
5. Performance test:
	- Measure total runtime for a 10-second clip.
	- Target on your current setup: roughly 3-10 seconds depending on GPU availability.

### Optional one-command validator

Use this to run transcription checks plus end-to-end pipeline checks and export a PASS/FAIL report:

```powershell
C:/Users/rohan/miniconda3/envs/veridiction/python.exe validate_step4_audio.py --audio-file data/sample.wav --report-out data/step4_validation_report.json
```