#!/usr/bin/env python
"""
Video call transcription using Mistral's Voxtral model.

Extracts audio channels from video files, transcribes each channel separately
using Voxtral-Mini-3B-2507 with timestamps, and outputs formatted text.

Uses Silero VAD to detect speech segments before transcription, which:
- Avoids hallucinated text on silent segments
- Provides accurate real timestamps for each utterance

Requires:
  pip install git+https://github.com/huggingface/transformers
  pip install mistral-common[audio]>=1.8.1
  pip install bitsandbytes accelerate sounddevice scipy numpy torch soundfile
  sudo apt-get install ffmpeg
"""

import os
import sys
import time
import warnings
import numpy as np
import torch
import tempfile
import soundfile as sf
from scipy.signal import resample_poly
from math import gcd
from transformers import VoxtralForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
import subprocess

# Debug mode - set to True to keep intermediate files for verification
DEBUG_MODE = True

# Suppress noisy warnings
warnings.filterwarnings("ignore", message=".*max_new_tokens.*")
warnings.filterwarnings("ignore", message=".*max_length.*")
warnings.filterwarnings("ignore", message=".*logits_process.*")

# Audio settings
VOXTRAL_SAMPLE_RATE = 16000  # Voxtral expects 16 kHz

# Model settings
MODEL_ID = "mistralai/Voxtral-Mini-3B-2507"

# Transcription settings
LANGUAGE = os.getenv("VOXTRAL_LANGUAGE", "fr")


def initialize_model():
    """Load Voxtral with 4-bit quantization to fit on 8GB VRAM."""
    print(f"Initializing {MODEL_ID} (4-bit quantized)...")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gpu = torch.cuda.get_device_properties(0)
        print(f"GPU: {gpu.name}, VRAM: {gpu.total_memory / 1024**3:.1f} GB")

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = VoxtralForConditionalGeneration.from_pretrained(
        MODEL_ID,
        quantization_config=quantization_config,
        device_map="auto",
        low_cpu_mem_usage=True,
    )

    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        print(f"VRAM used: {allocated:.1f} GB")

    print("Model loaded\n")
    return model, processor


def resample_audio(audio_np, orig_sr, target_sr):
    """Resample audio from orig_sr to target_sr."""
    if orig_sr == target_sr:
        return audio_np
    divisor = gcd(int(orig_sr), int(target_sr))
    up = int(target_sr) // divisor
    down = int(orig_sr) // divisor
    return resample_poly(audio_np, up, down).astype(np.float32)


def extract_audio_channels(video_path, output_dir):
    """Extract audio channels from video file using FFmpeg."""
    print(f"Extracting audio channels from {video_path}...")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    channel_files = []
    
    # First, try to extract stereo audio and split into separate channels
    print("Attempting stereo extraction...")
    
    # Extract full stereo audio first
    stereo_file = os.path.join(output_dir, "stereo_audio.wav")
    cmd_stereo = ['ffmpeg', '-i', video_path, '-map', '0:a', '-ac', '2', '-ar', str(VOXTRAL_SAMPLE_RATE), '-y', stereo_file]
    
    try:
        result = subprocess.run(cmd_stereo, capture_output=True, text=True)
        if result.returncode == 0:
            print("Successfully extracted stereo audio")
            
            # Now split stereo into separate mono channels
            for channel in range(2):
                output_file = os.path.join(output_dir, f"channel_{channel}.wav")
                cmd_split = ['ffmpeg', '-i', stereo_file, '-map_channel', f'0.0.{channel}', '-ac', '1', '-y', output_file]
                
                try:
                    subprocess.run(cmd_split, capture_output=True, check=True)
                    channel_files.append(output_file)
                    print(f"Successfully extracted channel {channel}")
                except subprocess.CalledProcessError as e:
                    print(f"Error extracting channel {channel}: {e}")
                    break
            
            # Clean up stereo file (unless in debug mode)
            if not DEBUG_MODE:
                try:
                    os.remove(stereo_file)
                except:
                    pass
            else:
                print(f"DEBUG: Keeping stereo file for verification: {stereo_file}")
            
            return channel_files
        else:
            print(f"Stereo extraction failed: {result.stderr}")
    except Exception as e:
        print(f"Error with stereo extraction: {e}")

    # If stereo extraction failed, try individual channel extraction
    print("Falling back to individual channel extraction...")
    
    # Try to extract each channel individually
    for channel in range(2):
        output_file = os.path.join(output_dir, f"channel_{channel}.wav")
        cmd = ['ffmpeg', '-i', video_path, '-map', f'0:a:{channel}', '-ac', '1', '-ar', str(VOXTRAL_SAMPLE_RATE), '-y', output_file]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                channel_files.append(output_file)
                print(f"Successfully extracted channel {channel}")
            else:
                print(f"Channel {channel} extraction failed: {result.stderr}")
                break
        except Exception as e:
            print(f"Error extracting channel {channel}: {e}")
            break

    # If we still don't have any channels, try the most basic extraction
    if not channel_files:
        print("Trying basic audio extraction...")
        output_file = os.path.join(output_dir, "channel_0.wav")
        cmd = ['ffmpeg', '-i', video_path, '-map', '0:a', '-ac', '1', '-ar', str(VOXTRAL_SAMPLE_RATE), '-y', output_file]
        
        try:
            subprocess.run(cmd, capture_output=True, check=True)
            channel_files.append(output_file)
            print("Successfully extracted single audio channel")
        except subprocess.CalledProcessError as e:
            print(f"Basic extraction also failed: {e}")

    return channel_files


def detect_speech_segments(audio_np, sample_rate, min_speech_duration=0.5, min_silence_duration=0.8, merge_gap=1.0):
    """Use Silero VAD to detect speech segments in audio.

    Returns list of (start_sec, end_sec) tuples for each speech segment.
    Adjacent segments closer than merge_gap seconds are merged.
    """
    print("  Running Voice Activity Detection (Silero VAD)...")

    # Load Silero VAD model
    vad_model, vad_utils = torch.hub.load(
        repo_or_dir='snakers4/silero-vad',
        model='silero_vad',
        trust_repo=True,
    )
    (get_speech_timestamps, _, _, _, _) = vad_utils

    # Silero VAD expects 16kHz mono float32 tensor
    audio_tensor = torch.from_numpy(audio_np).float()

    # Get speech timestamps (in samples)
    speech_timestamps = get_speech_timestamps(
        audio_tensor,
        vad_model,
        sampling_rate=sample_rate,
        min_speech_duration_ms=int(min_speech_duration * 1000),
        min_silence_duration_ms=int(min_silence_duration * 1000),
        threshold=0.5,
    )

    if not speech_timestamps:
        print("  No speech detected in audio.")
        return []

    # Convert to seconds
    segments = [
        (ts['start'] / sample_rate, ts['end'] / sample_rate)
        for ts in speech_timestamps
    ]

    # Merge segments that are close together (within merge_gap)
    merged = [segments[0]]
    for start, end in segments[1:]:
        if start - merged[-1][1] <= merge_gap:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))

    total_speech = sum(end - start for start, end in merged)
    total_duration = len(audio_np) / sample_rate
    print(f"  VAD found {len(merged)} speech segments ({total_speech:.1f}s speech / {total_duration:.1f}s total)")

    return merged


def transcribe_audio_chunked(model, processor, audio_path, channel_index):
    """Transcribe a single audio file using Voxtral, only on speech segments detected by VAD.

    Returns a list of (start_sec, text) tuples with real timestamps.
    """
    print(f"\n{'='*50}")
    print(f"Processing Channel {channel_index}")
    print(f"{'='*50}")

    try:
        # Load and validate audio file
        audio_np, sample_rate = sf.read(audio_path)
        duration = len(audio_np) / sample_rate

        print(f"Audio validation:")
        print(f"  Duration: {duration:.2f} seconds")
        print(f"  Sample rate: {sample_rate} Hz")
        print(f"  Channels: {audio_np.shape[0] if len(audio_np.shape) > 1 else 1}")
        print(f"  Samples: {len(audio_np)}")

        # Convert to mono if needed
        if len(audio_np.shape) > 1:
            audio_np = audio_np.mean(axis=1)
            print(f"  Converted to mono")

        # Resample to 16 kHz if needed
        if sample_rate != VOXTRAL_SAMPLE_RATE:
            print(f"  Resampling from {sample_rate} Hz to {VOXTRAL_SAMPLE_RATE} Hz...")
            audio_np = resample_audio(audio_np, sample_rate, VOXTRAL_SAMPLE_RATE)
            print(f"  Resampling complete")

        # Detect speech segments with VAD
        speech_segments = detect_speech_segments(audio_np, VOXTRAL_SAMPLE_RATE)

        if not speech_segments:
            print("  No speech found, skipping channel.")
            return []

        # Check GPU usage
        device = model.device
        if device.type == 'cuda':
            print(f"  Using GPU: {torch.cuda.get_device_name(0)}")
            torch.cuda.empty_cache()
        else:
            print(f"\n{'!'*50}")
            print(f"  WARNING: Running on CPU - this will be very slow!")
            print(f"  Consider using a GPU for faster transcription.")
            print(f"{'!'*50}\n")

        # Transcribe each speech segment (or groups of segments up to ~25s)
        MAX_CHUNK_SECONDS = 25
        timestamped_transcriptions = []  # list of (start_sec, text)

        # Group adjacent speech segments into chunks that fit within MAX_CHUNK_SECONDS
        chunks = []  # list of (start_sec, end_sec)
        current_chunk_start = speech_segments[0][0]
        current_chunk_end = speech_segments[0][1]

        for seg_start, seg_end in speech_segments[1:]:
            if seg_end - current_chunk_start <= MAX_CHUNK_SECONDS:
                # Extend current chunk to include this segment
                current_chunk_end = seg_end
            else:
                # Save current chunk and start a new one
                chunks.append((current_chunk_start, current_chunk_end))
                current_chunk_start = seg_start
                current_chunk_end = seg_end
        chunks.append((current_chunk_start, current_chunk_end))

        print(f"  Transcribing {len(chunks)} chunks from {len(speech_segments)} speech segments...")

        for chunk_idx, (chunk_start, chunk_end) in enumerate(chunks):
            start_sample = int(chunk_start * VOXTRAL_SAMPLE_RATE)
            end_sample = int(chunk_end * VOXTRAL_SAMPLE_RATE)
            chunk_audio = audio_np[start_sample:end_sample]

            chunk_duration = len(chunk_audio) / VOXTRAL_SAMPLE_RATE
            print(f"  Chunk {chunk_idx + 1}/{len(chunks)} [{format_timestamp(chunk_start)}-{format_timestamp(chunk_end)}] ({chunk_duration:.1f}s)...")

            # Save chunk to temporary file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name
                sf.write(tmp_path, chunk_audio, VOXTRAL_SAMPLE_RATE)

            try:
                inputs = processor.apply_transcription_request(
                    language=LANGUAGE,
                    audio=tmp_path,
                    model_id=MODEL_ID,
                )
                inputs = inputs.to(model.device, dtype=torch.float16)

                with torch.inference_mode():
                    start = time.perf_counter()
                    outputs = model.generate(**inputs, max_new_tokens=440, temperature=0.0, do_sample=False)
                    elapsed = time.perf_counter() - start

                response_tokens = outputs[0][inputs["input_ids"].shape[1]:]
                chunk_text = processor.batch_decode([response_tokens], skip_special_tokens=True)[0].strip()

                if chunk_text:
                    timestamped_transcriptions.append((chunk_start, chunk_text))

                num_tokens = len(response_tokens)
                tok_per_sec = num_tokens / elapsed if elapsed > 0 else 0
                print(f"    {num_tokens} tokens in {elapsed:.1f}s ({tok_per_sec:.1f} tok/s)")

            except Exception as e:
                print(f"    Error processing chunk {chunk_idx + 1}: {e}")
            finally:
                try:
                    os.unlink(tmp_path)
                except:
                    pass

        total_words = sum(len(t.split()) for _, t in timestamped_transcriptions)
        total_speech = sum(e - s for s, e in speech_segments)
        print(f"\nTranscription Results:")
        print(f"  Total duration: {duration:.2f} seconds")
        print(f"  Speech duration: {total_speech:.2f} seconds")
        print(f"  Segments transcribed: {len(timestamped_transcriptions)}")
        print(f"  Estimated words: {total_words}")

        return timestamped_transcriptions

    except Exception as e:
        print(f"Error transcribing channel {channel_index}: {e}")
        import traceback
        traceback.print_exc()
        return []


def format_timestamp(seconds):
    """Format seconds as HH:MM:SS."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"[{hours:02d}:{minutes:02d}:{seconds:02d}]"


def process_video(video_path, output_file):
    """Main processing function for video transcription."""
    print(f"Processing video: {video_path}")

    # Create temporary directory for audio channels
    temp_dir = "temp_audio_channels"

    # Extract audio channels
    channel_files = extract_audio_channels(video_path, temp_dir)

    if not channel_files:
        print("No audio channels extracted. Exiting.")
        return

    # Initialize model
    model, processor = initialize_model()

    # Transcribe each channel - now returns list of (timestamp_sec, text) tuples
    transcriptions = []
    for i, channel_file in enumerate(channel_files):
        segments = transcribe_audio_chunked(model, processor, channel_file, i)
        transcriptions.append((i, segments))

    # Build a single list of all sentences with real timestamps and speaker labels,
    # then sort by time so the output reads like a conversation.
    # Channel names can be customized via environment variable, e.g. CHANNEL_NAMES="Alice,Bob"
    channel_names_env = os.getenv("CHANNEL_NAMES", "")
    channel_names = [n.strip() for n in channel_names_env.split(",") if n.strip()] if channel_names_env else []

    all_lines = []  # list of (timestamp_sec, speaker_label, sentence_text)
    for channel_index, segments in transcriptions:
        if channel_index < len(channel_names):
            speaker = channel_names[channel_index]
        else:
            speaker = f"Speaker {channel_index + 1}"

        for seg_idx, (timestamp_sec, text) in enumerate(segments):
            # Determine the time span for this segment
            if seg_idx + 1 < len(segments):
                next_ts = segments[seg_idx + 1][0]
            else:
                next_ts = timestamp_sec + 30  # assume ~30s for last segment

            sentences = [s.strip().rstrip('.') for s in text.split('. ') if s.strip()]
            if not sentences:
                continue

            seg_duration = next_ts - timestamp_sec
            for i, sentence in enumerate(sentences):
                sentence_ts = timestamp_sec + (i / len(sentences)) * seg_duration
                all_lines.append((sentence_ts, speaker, sentence))

    # Sort all lines by timestamp to produce a chronological conversation
    all_lines.sort(key=lambda x: x[0])

    # Group consecutive lines from the same speaker into paragraphs
    paragraphs = []
    for ts, speaker, sentence in all_lines:
        if paragraphs and paragraphs[-1][1] == speaker:
            paragraphs[-1] = (paragraphs[-1][0], speaker, paragraphs[-1][2] + " " + sentence + ".")
        else:
            paragraphs.append((ts, speaker, sentence + "."))

    print(f"Writing output to {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        for ts, speaker, text in paragraphs:
            f.write(f"{format_timestamp(ts)} {speaker}: {text}\n\n")

    # Clean up temporary files (unless in debug mode)
    if not DEBUG_MODE:
        for channel_file in channel_files:
            try:
                os.remove(channel_file)
            except:
                pass

        try:
            os.rmdir(temp_dir)
        except:
            pass
    else:
        print(f"DEBUG: Keeping intermediate channel files in: {temp_dir}")
        print(f"DEBUG: Channel files preserved:")
        for i, channel_file in enumerate(channel_files):
            print(f"  Channel {i}: {channel_file}")

    print(f"Transcription complete. Output saved to {output_file}")


def main():
    """Main entry point."""
    if len(sys.argv) != 2:
        print("Usage: python transcribe.py <video_file>")
        sys.exit(1)
    
    video_path = sys.argv[1]
    
    if not os.path.exists(video_path):
        print(f"Error: File {video_path} not found")
        sys.exit(1)
    
    # Generate output filename
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    output_file = f"{base_name}_transcription.txt"
    
    # Process the video
    process_video(video_path, output_file)


if __name__ == "__main__":
    main()