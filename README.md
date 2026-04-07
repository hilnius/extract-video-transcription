# Video Call Transcription Tool

A collection of scripts to transcribe video calls with speaker separation and timestamps using LLM-based speech recognition.

## Features

- Extract audio channels from video files (stereo and mono)
- Transcribe each audio channel separately using Mistral's Voxtral model
- Include timestamps for each spoken phrase/sentence
- Output formatted text with speaker identification
- Support for multiple video formats (MP4, AVI, MOV, etc.)
- LLM-based transcription with chunked processing to avoid GPU memory issues
- Debug mode to preserve intermediate audio files for verification
- Automatic GPU detection with CPU warning

## Requirements

- FFmpeg (for audio extraction)
- Python 3.8+
- Required Python packages:
  - `transformers` (from HuggingFace)
  - `mistral-common[audio]>=1.8.1`
  - `bitsandbytes`
  - `accelerate`
  - `sounddevice`
  - `scipy`
  - `numpy`
  - `torch`
  - `soundfile`

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

# Debug mode (keeps intermediate audio files)
# Set DEBUG_MODE = True in transcribe.py before running
```

### Output Format

The script generates a text file with sentence-level timestamps:

```
[Channel 1]
  [00:00:05] Hello everyone, welcome to the meeting.
  [00:00:12] Today we'll be discussing the project timeline.
  [00:00:18] Let's get started with the first topic.

[Channel 2]
  [00:00:22] Thanks for organizing this.
  [00:00:28] I have some questions about the schedule.
```

## How It Works

1. **Audio Extraction**: Uses FFmpeg to extract stereo audio and split into separate channels
2. **Chunked Processing**: Processes audio in 25-second chunks to avoid GPU memory issues
3. **LLM Transcription**: Uses Mistral's Voxtral-Mini-3B-2507 model with 4-bit quantization
4. **Timestamp Generation**: Calculates approximate timestamps for each sentence
5. **Formatting**: Combines all channels into structured output with sentence-level timestamps

## Model Details

- **Model**: mistralai/Voxtral-Mini-3B-2507
- **Quantization**: 4-bit (NF4) to fit on 8GB VRAM GPUs
- **Sample Rate**: 16 kHz (automatically resampled)
- **Language**: Configurable via `VOXTRAL_LANGUAGE` environment variable (default: French)
- **Chunk Size**: 25 seconds (configurable for memory constraints)

## Key Features

- **Memory Management**: Automatic GPU cache clearing and chunked processing
- **Error Handling**: Robust FFmpeg audio extraction with multiple fallback methods
- **Progress Tracking**: Detailed progress output for each processing step
- **Debug Mode**: Option to preserve intermediate files for verification
- **GPU Detection**: Automatic GPU usage detection with CPU warnings

## Limitations

- Requires GPU with at least 8GB VRAM for optimal performance
- Accuracy depends on audio quality and background noise
- May not work well with overlapping speech
- Currently optimized for French (configurable for other languages)
- Timestamp precision is approximate (sentence-level, not word-level)

## Future Enhancements

- Speaker diarization for better channel identification
- Support for multiple languages with automatic detection
- Noise reduction preprocessing
- Batch processing for multiple video files
- Real-time transcription mode
- Word-level timestamp precision
- Automatic language detection

## Debug Mode

Set `DEBUG_MODE = True` in `transcribe.py` to:
- Keep intermediate stereo audio files
- Preserve individual channel WAV files
- Maintain temporary directory structure
- Get detailed file paths for verification

## License

MIT License - see LICENSE file for details

## License

MIT License - see LICENSE file for details
