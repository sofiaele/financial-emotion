"""
Speaker diarization and automatic speech recognition using WhisperX.

This module processes audio files specified in a CSV metadata file,
performing speaker diarization and speech-to-text transcription.
"""

import whisperx
from whisperx.diarize import DiarizationPipeline
import torch
import librosa
import csv
import pandas as pd
from whisper_vars import *
import os
import argparse

def diarize_asr(audios_path, csv_file=None):
    """
    Perform speaker diarization and ASR on audio files.

    Args:
        audios_path: Directory containing audio files
        csv_file: Optional CSV file with 'audio' column specifying which files to process.
                 If None, processes all WAV files in the directory.
    """
    # Get list of audio files to process
    if csv_file and os.path.exists(csv_file):
        # Read CSV to get specific audio files to process
        try:
            df = pd.read_csv(csv_file, sep=';')
            if 'audio' not in df.columns:
                df = pd.read_csv(csv_file)
        except:
            df = pd.read_csv(csv_file)

        # Get audio filenames from CSV (filter out empty values)
        audio_filenames = df['audio'].dropna().tolist()
        audio_filenames = [f for f in audio_filenames if f and f.endswith('.wav')]

        # Full paths to audio files from CSV
        audios = [os.path.join(audios_path, audio) for audio in audio_filenames
                  if os.path.exists(os.path.join(audios_path, audio))
                  and ".".join(audio.split(".")[:-1]) + '.csv' not in os.listdir(audios_path)]

        print(f"Processing {len(audios)} audio files from CSV: {csv_file}")
    else:
        # Fallback: process all WAV files in directory (original behavior)
        audios = [os.path.join(audios_path, audio) for audio in os.listdir(audios_path)
                  if audio.endswith(".wav")
                  and ".".join(audio.split(".")[:-1]) + '.csv' not in os.listdir(audios_path)]
        print(f"Processing all {len(audios)} WAV files in directory: {audios_path}")

    if not audios:
        print("No audio files to process.")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load models
    model = whisperx.load_model("medium", device, compute_type=COMPUTE_TYPE)

    # 2. Align whisper output
    model_a, metadata = whisperx.load_align_model(language_code=LANGUAGE, device=device)
    diarize_model = DiarizationPipeline(
        model_name="pyannote/speaker-diarization-3.1",
        use_auth_token=HF_TOKEN,
        device=device
    )

    # Process each audio file
    for i, audio_file in enumerate(audios, 1):
        print(f"\nProcessing ({i}/{len(audios)}): {os.path.basename(audio_file)}")

        try:
            audio = librosa.load(audio_file, sr=16000)[0]
            result = model.transcribe(audio, batch_size=BATCH_SIZE, language=LANGUAGE)

            result = whisperx.align(
                result["segments"], model_a, metadata, audio, device,
                return_char_alignments=False
            )

            diarize_segments = diarize_model(audio)

            result = whisperx.assign_word_speakers(diarize_segments, result)

            # segments are now assigned speaker IDs
            csv_data = []
            for entry in result["segments"]:
                if "speaker" not in entry:
                    entry['speaker'] = "UNKNOWN"
                csv_data.append({
                    'start': entry['start'],
                    'end': entry['end'],
                    'text': entry['text'],
                    'speaker': entry['speaker']
                })

            # Saving to CSV
            csv_filename = ".".join(audio_file.split(".")[:-1]) + ".csv"
            with open(csv_filename, 'w', newline='') as csvfile:
                fieldnames = ['start', 'end', 'text', 'speaker']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                # Write the header
                writer.writeheader()

                # Write data
                for entry in csv_data:
                    writer.writerow(entry)

            print(f"✓ Saved: {os.path.basename(csv_filename)}")

        except Exception as e:
            print(f"✗ Error processing {os.path.basename(audio_file)}: {str(e)}")
            continue


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Speaker diarization and ASR using WhisperX'
    )
    parser.add_argument('--audios',
                        help='Path to directory containing audio files',
                        type=str,
                        default='../data/audio/')
    parser.add_argument('--csv',
                        help='Path to CSV file with audio filenames to process (optional)',
                        type=str,
                        default=None)
    args = parser.parse_args()

    diarize_asr(args.audios, csv_file=args.csv)
