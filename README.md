# Financial Emotion Recognition

Emotion recognition from Federal Reserve press conference speech.

## Overview

This repository implements emotion recognition for Federal Reserve press conferences (2011Q3-2023Q4). The pipeline extracts emotional content from Chair speeches using speaker diarization, target speaker identification, and multitask emotion recognition.

**Input**: Audio files from Federal Reserve press conferences
**Output**: `ground_truths_sentiments.csv` - Emotional posteriors for each conference

## Prerequisites

- Python 3.10
- [uv](https://github.com/astral-sh/uv) package manager (recommended)
- FFmpeg and development libraries (required for audio processing)
- pkg-config (required for building PyAV)

## Installation

### 1. Install uv Package Manager

```bash
# On macOS and Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows (PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Or with pip
pip install uv
```

### 2. Install System Dependencies

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

### 3. Install Python Dependencies

1. Clone the repository:
   ```bash
   git clone https://github.com/sofiaele/financial-emotion.git
   cd financial-emotion
   ```

2. Create a virtual environment with uv:
   ```bash
   uv venv -p 3.10
   source .venv/bin/activate  # On Linux/Mac
   # or
   .venv\Scripts\activate  # On Windows
   ```

3. Install required packages:
   ```bash
   uv pip install -r requirements.txt
   ```

**Note**: Using `uv` is recommended as it's significantly faster than traditional pip and handles dependency resolution more reliably.

## Data Preparation

### Audio Files

This repository expects pre-downloaded audio files from Federal Reserve press conferences. Audio files should be:

- **Format**: WAV
- **Sample rate**: 16000 Hz (16 kHz)
- **Channels**: Mono (1 channel)
- **Location**: `data/audio/` directory

Audio files can be obtained from the [Federal Reserve's official YouTube channel](https://www.youtube.com/user/FederalReserve) or other official sources.

### Link Audio Files to Metadata

Once audio files are placed in `data/audio/`, link them to the metadata CSV:

```bash
python utils/preprocessing.py --csv data/all_videos.csv
```

This creates `all_videos_audio.csv` mapping each video entry to its corresponding audio file.

## Pipeline

### 1. Link Audio Files with Metadata

```bash
python utils/preprocessing.py --csv data/all_videos.csv
```

**Output**: `all_videos_audio.csv` with audio file mappings

### 2. Speaker Diarization

Process only the audio files specified in your CSV:

```bash
python diarization-asr/whisper.py --audios data/audio --csv all_videos_audio.csv
```

Or process all audio files in the directory:

```bash
python diarization-asr/whisper.py --audios data/audio
```

**Output**: CSV files in `data/audio/` with speaker turn information for each audio file

**Note**: The `--csv` parameter ensures only audios listed in your metadata are processed, avoiding unnecessary computation on extra files.

### 3. Dominant Speaker Identification

```bash
python diarization-asr/dominant_speaker.py --csvs data/audio
```

**Output**: Labels Fed Chair utterances as 'banker' in the speaker turn CSV files

### 4. Audio Segmentation

```bash
python diarization-asr/split_audio_segments.py
```

**Output**: Individual utterance audio files split by speaker turns

### 5. Emotion Recognition

```bash
python models/emotion_recognition/inference.py \
    --mode batch \
    --model_path data/emotion_models/best_model.pt \
    --audio_dir data/audio \
    --output ground_truths_sentiments.csv
```

**Output**: `ground_truths_sentiments.csv` with emotional posteriors aggregated at conference level

## Output Format

`ground_truths_sentiments.csv` contains:
- `conference`: Conference identifier
- `audio`: Audio filename
- `emotion_angry`, `emotion_happy`, `emotion_neutral`, `emotion_sad`: Emotion posteriors
- `valence_negative`, `valence_neutral`, `valence_positive`: Valence posteriors
- `arousal_weak`, `arousal_neutral`, `arousal_strong`: Arousal posteriors

## Model Training

For information on training the emotion recognition model, see `models/emotion_recognition/README.md`.

## Contact

seleftheriou@aueb.gr
