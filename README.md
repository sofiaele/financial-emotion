# financial-emotion

Emotion recognition from Federal Reserve press conference speech.

## Overview

This repository implements emotion recognition for Federal Reserve press conferences (2011Q3-2023Q4). The pipeline extracts emotional content from Chair speeches using speaker diarization, target speaker identification, and multitask emotion recognition.

**Input**: `data/all_videos.csv` - List of YouTube video URLs
**Output**: `ground_truths_sentiments.csv` - Emotional posteriors for each conference

## Installation

### Prerequisites
- Python 3.8 or higher
- FFmpeg and development libraries (required for audio processing)
- pkg-config (required for building PyAV)

### Install FFmpeg and Dependencies

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install -y ffmpeg pkg-config \
    libavformat-dev libavcodec-dev libavdevice-dev \
    libavutil-dev libavfilter-dev libswscale-dev libswresample-dev
```

**macOS:**
```bash
brew install ffmpeg pkg-config
```

**Windows:**
1. Install FFmpeg from [ffmpeg.org](https://ffmpeg.org/download.html) and add to PATH
2. Install pkg-config from [chocolatey](https://chocolatey.org/): `choco install pkgconfiglite`
3. Set PKG_CONFIG_PATH to point to FFmpeg's pkgconfig directory

### Install Python Dependencies

1. Clone the repository:
   ```bash
   git clone https://github.com/sofiaele/financial-emotion.git
   cd financial-emotion
   ```

2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Pipeline

1. Download audio from YouTube videos
   ```bash
   python utils/preprocessing.py --csv data/all_videos.csv
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
