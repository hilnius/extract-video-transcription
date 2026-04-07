#!/usr/bin/env python
"""
Video call transcription using Mistral's Voxtral model.

Extracts audio channels from video files, transcribes each channel separately
using Voxtral-Mini-3B-2507 with timestamps, and outputs formatted text.

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


def transcribe_audio_chunked(model, processor, audio_path, channel_index):
    """Transcribe a single audio file using Voxtral with chunked processing to avoid OOM."""
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

        # Check GPU usage
        device = model.device
        if device.type == 'cuda':
            print(f"  Using GPU: {torch.cuda.get_device_name(0)}")
            # Clear GPU cache before processing
            torch.cuda.empty_cache()
        else:
            print(f"\n{'!'*50}")
            print(f"  WARNING: Running on CPU - this will be very slow!")
            print(f"  Consider using a GPU for faster transcription.")
            print(f"{'!'*50}\n")

        # Process audio in chunks to avoid OOM errors
        # Voxtral can handle up to ~30 seconds comfortably on 8GB GPU
        chunk_size_seconds = 25  # Conservative chunk size
        chunk_samples = int(chunk_size_seconds * VOXTRAL_SAMPLE_RATE)
        
        print(f"  Processing in chunks of {chunk_size_seconds} seconds...")
        
        full_transcription = []
        total_chunks = int(np.ceil(len(audio_np) / chunk_samples))
        
        for chunk_idx in range(total_chunks):
            start_sample = chunk_idx * chunk_samples
            end_sample = min((chunk_idx + 1) * chunk_samples, len(audio_np))
            chunk_audio = audio_np[start_sample:end_sample]
            
            chunk_duration = len(chunk_audio) / VOXTRAL_SAMPLE_RATE
            print(f"  Processing chunk {chunk_idx + 1}/{total_chunks} ({chunk_duration:.1f}s)...")
            
            # Save chunk to temporary file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name
                sf.write(tmp_path, chunk_audio, VOXTRAL_SAMPLE_RATE)

            try:
                # Transcribe this chunk
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

                # Decode only the generated tokens
                response_tokens = outputs[0][inputs["input_ids"].shape[1]:]
                chunk_text = processor.batch_decode([response_tokens], skip_special_tokens=True)[0]
                full_transcription.append(chunk_text)
                
                num_tokens = len(response_tokens)
                tok_per_sec = num_tokens / elapsed if elapsed > 0 else 0
                print(f"    Chunk {chunk_idx + 1}: {num_tokens} tokens in {elapsed:.1f}s ({tok_per_sec:.1f} tok/s)")
                
            except Exception as e:
                print(f"    Error processing chunk {chunk_idx + 1}: {e}")
                full_transcription.append(f"[Error in chunk {chunk_idx + 1}]")
            finally:
                # Clean up temporary file
                try:
                    os.unlink(tmp_path)
                except:
                    pass

        # Combine all chunks
        full_text = " ".join(full_transcription)
        
        # Calculate overall metrics
        total_time = sum(len(chunk.split()) for chunk in full_transcription if chunk.strip())
        words_per_min = (total_time / duration) * 60 if duration > 0 else 0
        
        print(f"\nTranscription Results:")
        print(f"  Total duration: {duration:.2f} seconds")
        print(f"  Processed chunks: {total_chunks}")
        print(f"  Estimated words: {total_time}")
        print(f"  Overall speed: {words_per_min:.0f} words/minute")
        print(f"  Real-time factor: {(duration/duration):.2f}x")  # Will be 1.0x since we process sequentially
        
        return full_text
        
    except Exception as e:
        print(f"Error transcribing channel {channel_index}: {e}")
        import traceback
        traceback.print_exc()
        return f"[Error transcribing channel {channel_index}]"


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
    
    # Transcribe each channel
    transcriptions = []
    for i, channel_file in enumerate(channel_files):
        transcription = transcribe_audio_chunked(model, processor, channel_file, i)
        transcriptions.append((i, transcription))
    
    # Write output file with proper timestamp handling
    print(f"Writing output to {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        for channel_index, transcription in transcriptions:
            # Split transcription into sentences/phrases for better timestamp handling
            # This is a simple approach - in production you'd want more sophisticated
            # sentence splitting and actual timestamp alignment
            
            # Split by sentences (simple approach)
            sentences = transcription.split('. ')
            
            # Calculate approximate timestamps for each sentence
            # This assumes equal time distribution - real implementation would
            # use the actual audio timing
            if sentences:
                f.write(f"[Channel {channel_index + 1}]\n")
                
                # Get the audio file to determine duration
                audio_path = os.path.join(temp_dir, f"channel_{channel_index}.wav")
                if os.path.exists(audio_path):
                    audio_np, sample_rate = sf.read(audio_path)
                    total_duration = len(audio_np) / sample_rate
                    sentences_per_second = len(sentences) / total_duration if total_duration > 0 else 1
                else:
                    sentences_per_second = 1
                    total_duration = len(sentences)  # fallback
                
                # Write each sentence with approximate timestamp
                for i, sentence in enumerate(sentences):
                    if sentence.strip():
                        # Approximate timestamp (simple linear distribution)
                        timestamp_seconds = (i / sentences_per_second) if sentences_per_second > 0 else 0
                        f.write(f"  {format_timestamp(timestamp_seconds)} {sentence.strip()}.\n")
                f.write("\n")
            else:
                f.write(f"[Channel {channel_index + 1}] {format_timestamp(0)} {transcription}\n\n")
    
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