# Video Call Transcription Tool

A collection of scripts to transcribe video calls with speaker separation and timestamps using LLM-based speech recognition.

## Features

- Extract audio channels from video files
- Transcribe each audio channel separately using Mistral's Voxtral model
- Include timestamps for each spoken phrase
- Output formatted text with speaker identification
- Support for multiple video formats
- LLM-based transcription for improved accuracy

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
python transcribe.py input_video.mp4
```

### Output Format

The script generates a text file with the following format:

```
[Channel 1] [00:00:01] Hello everyone, welcome to the meeting.
[Channel 1] [00:00:05] Today we'll be discussing the project timeline.
[Channel 2] [00:00:08] Thanks for organizing this.
[Channel 1] [00:00:10] Let's get started with the first topic.
```

## How It Works

1. **Audio Extraction**: Uses FFmpeg to extract individual audio channels from the video
2. **LLM Transcription**: Processes each channel using Mistral's Voxtral-Mini-3B-2507 model with 4-bit quantization
3. **Timestamp Tracking**: Records the time offset for each transcribed phrase
4. **Formatting**: Combines all channels into a single output file with clear speaker separation

## Model Details

- **Model**: mistralai/Voxtral-Mini-3B-2507
- **Quantization**: 4-bit (NF4) to fit on 8GB VRAM GPUs
- **Sample Rate**: 16 kHz (automatically resampled)
- **Language**: Configurable (default: French)

## Limitations

- Requires GPU with at least 8GB VRAM for optimal performance
- Accuracy depends on audio quality and background noise
- May not work well with overlapping speech
- Currently optimized for French (can be configured for other languages)

## Future Enhancements

- Speaker diarization for better channel identification
- Support for multiple languages with automatic detection
- Noise reduction preprocessing
- Batch processing for multiple video files
- Real-time transcription mode

## License

MIT License - see LICENSE file for details
