# financial-emotion

Emotion recognition from Federal Reserve press conference speech.

## Overview

This repository implements emotion recognition for Federal Reserve press conferences (2011Q3-2023Q4). The pipeline extracts emotional content from Chair speeches using speaker diarization, target speaker identification, and multitask emotion recognition.

**Input**: `data/all_videos.csv` - List of YouTube video URLs
**Output**: `ground_truths_sentiments.csv` - Emotional posteriors for each conference

## Pipeline

1. Download audio from YouTube videos
   ```bash
   python utils/preprocessing/download_audios_from_csv.py --csv data/all_videos.csv
   ```
   Creates audio files in `data/audio/`

2. Speaker diarization
   ```bash
   python diarization-asr/whisper.py --audios data/audio
   ```
   Creates CSV files with speaker turns for each audio

3. Dominant speaker identification
   ```bash
   python diarization-asr/dominant_speaker.py --csvs data/audio
   ```
   Labels Fed Chair utterances as 'banker'

4. Audio segmentation
   ```bash
   python diarization-asr/split_audio_segments.py
   ```
   Splits audio into individual utterance files

5. Emotion recognition
   ```bash
   python models/emotion_recognition/inference.py \
       --mode batch \
       --model_path data/emotion_models/best_model.pt \
       --audio_dir data/audio \
       --output ground_truths_sentiments.csv
   ```
   Produces emotional posteriors aggregated at conference level

## Output Format

`ground_truths_sentiments.csv` contains:
- conference: Conference identifier
- audio: Audio filename
- emotion_angry, emotion_happy, emotion_neutral, emotion_sad: Emotion posteriors
- valence_negative, valence_neutral, valence_positive: Valence posteriors
- arousal_weak, arousal_neutral, arousal_strong: Arousal posteriors

See `models/emotion_recognition/README.md` for model training details.
