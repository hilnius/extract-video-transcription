# Video Call Transcription Tool

Transcribes video calls with speaker separation and real timestamps using Mistral's Voxtral model and Silero VAD.

## Features

- Extracts audio channels from video files (stereo and mono)
- **Voice Activity Detection** (Silero VAD) to skip silence and avoid hallucinated text
- **Real timestamps** based on actual speech timing (not approximations)
- **Conversation-ordered output** — both speakers interleaved chronologically, not grouped by channel
- Customizable speaker names via environment variable
- 4-bit quantized model to fit on 8GB VRAM GPUs
- Chunked processing to avoid GPU memory issues
- Debug mode to preserve intermediate audio files

## Requirements

- FFmpeg (for audio extraction)
- Python 3.8+
- GPU with at least 8GB VRAM (CPU fallback available but slow)
- Python packages:
  - `transformers` (from HuggingFace)
  - `mistral-common[audio]>=1.8.1`
  - `bitsandbytes`
  - `accelerate`
  - `sounddevice`
  - `scipy`
  - `numpy`
  - `torch`
  - `soundfile`

Silero VAD is downloaded automatically via `torch.hub` on first run.

## Installation

```bash
# Install FFmpeg (Ubuntu/Debian)
sudo apt-get install ffmpeg

# Install Python dependencies
pip install git+https://github.com/huggingface/transformers mistral-common[audio]>=1.8.1 bitsandbytes accelerate sounddevice scipy numpy torch soundfile
```

## Usage

```bash
# Basic usage
python transcribe.py input_video.mp4

# With custom speaker names
CHANNEL_NAMES="Alice,Bob" python transcribe.py input_video.mp4

# With a different language (default: French)
VOXTRAL_LANGUAGE=en python transcribe.py input_video.mp4
```

Output is written to `<video_name>_transcription.txt` in the current directory.

### Output Format

The output is a chronological conversation with real timestamps:

```
[00:00:02] Speaker 1: Hello everyone, welcome to the meeting. Today we'll be discussing the project timeline.

[00:00:15] Speaker 2: Thanks for organizing this. I have some questions about the schedule.

[00:00:28] Speaker 1: Sure, go ahead.
```

Consecutive sentences from the same speaker are grouped into paragraphs.

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `VOXTRAL_LANGUAGE` | `fr` | Transcription language |
| `CHANNEL_NAMES` | (none) | Comma-separated speaker names, e.g. `"Alice,Bob"` |

## How It Works

1. **Audio Extraction**: FFmpeg extracts stereo audio and splits into separate mono channels (with fallbacks for different audio layouts)
2. **Voice Activity Detection**: Silero VAD detects speech segments in each channel, providing real start/end timestamps and skipping silence
3. **Chunked Transcription**: Speech segments are grouped into chunks of up to 25 seconds and sent to Voxtral-Mini-3B-2507 (4-bit quantized)
4. **Chronological Merge**: All transcribed segments from both channels are sorted by timestamp and grouped by speaker, producing a conversation-ordered output

## Model Details

- **Model**: mistralai/Voxtral-Mini-3B-2507
- **Quantization**: 4-bit NF4 with double quantization
- **VAD**: Silero VAD (via torch.hub, snakers4/silero-vad)
- **Sample Rate**: 16 kHz (automatically resampled)
- **Max chunk size**: 25 seconds

## Debug Mode

Set `DEBUG_MODE = True` in `transcribe.py` (enabled by default) to preserve:
- Intermediate stereo audio file
- Individual channel WAV files
- Temporary directory structure

## Limitations

- Requires GPU with at least 8GB VRAM for reasonable performance
- Accuracy depends on audio quality and background noise
- Overlapping speech may be attributed to the wrong speaker
- Timestamp precision is at the VAD segment level, not word-level
