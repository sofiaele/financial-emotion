import whisperx
import torch
import librosa
import csv
from whisper_vars import *
import os
import argparse

def diarize_asr(audios_path):

    #full path to audio files
    audios = [os.path.join(audios_path, audio) for audio in os.listdir(audios_path) if audio.endswith(".wav")
              and ".".join(audio.split(".")[:-1]) + '.csv' not in os.listdir(audios_path)]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = whisperx.load_model("medium", device, compute_type=COMPUTE_TYPE)
    # 2. Align whisper output
    model_a, metadata = whisperx.load_align_model(language_code=LANGUAGE, device=device)
    diarize_model = whisperx.DiarizationPipeline(model_name="pyannote/speaker-diarization-3.1",
                                                 use_auth_token=HF_TOKEN, device=device)

    for audio_file in audios:
        audio = librosa.load(audio_file, sr=16000)[0]
        result = model.transcribe(audio, batch_size=BATCH_SIZE, language=LANGUAGE)

        result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)

        diarize_segments = diarize_model(audio)

        result = whisperx.assign_word_speakers(diarize_segments, result)
        # segments are now assigned speaker IDs
        csv_data = []
        for entry in result["segments"]:
            if "speaker" not in entry:
                entry['speaker'] = "UNKNOWN"
            csv_data.append({'start': entry['start'], 'end': entry['end'], 'text': entry['text'], 'speaker': entry['speaker']})

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--audios', help='path to audio files', type=str, default='../data/audio/')
    args = parser.parse_args()
    diarize_asr(args.audios)