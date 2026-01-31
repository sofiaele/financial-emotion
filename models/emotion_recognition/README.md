# Emotion Recognition Module

Multitask emotion recognition system based on TRILLsson3 for Federal Reserve press conference speech analysis. Classifies speech across three dimensions: emotion (4 classes), valence (3 classes), and arousal (3 classes). Outputs posterior probabilities for use in economic analysis.

## Architecture

- Backbone: TRILLsson3 (CNN-based, EfficientNetV2-S inspired) producing 1024-dimensional embeddings
- Task-specific heads: Three linear classifiers (1024×4, 1024×3, 1024×3)
- Input: 80-bin mel-spectrograms
- Output: Posterior probabilities for emotion, valence, and arousal

## Requirements

Dependencies are listed in the main `requirements.txt`:
- torch, torchaudio
- transformers (for TRILLsson3)
- librosa
- speechbrain (optional)
- scikit-learn, pandas

## Dataset Preparation

Download and configure training datasets:

1. **IEMOCAP**: https://sail.usc.edu/iemocap/
2. **MSP-Podcast**: https://ecs.utdallas.edu/research/researchlabs/msp-lab/MSP-Podcast.html

Update paths in `config.py`:
```python
IEMOCAP_PATH = "/path/to/IEMOCAP"
MSP_PODCAST_PATH = "/path/to/MSP-Podcast"
```

## Training

Train on IEMOCAP:
```bash
python train.py \
    --dataset iemocap \
    --data_path /path/to/IEMOCAP \
    --batch_size 32 \
    --num_epochs 100 \
    --checkpoint_dir ../data/emotion_models
```

Train on MSP-Podcast:
```bash
python train.py \
    --dataset msp-podcast \
    --data_path /path/to/MSP-Podcast \
    --batch_size 32 \
    --num_epochs 100 \
    --checkpoint_dir ../data/emotion_models
```

Options:
- `--freeze_backbone`: Freeze TRILLsson3 weights (train only task heads)
- `--use_dynamic_sampler`: Use SpeechBrain's DynamicBatchSampler
- `--learning_rate`: Default 1e-4
- `--patience`: Early stopping patience (default 15)

## Inference

Process single utterance:
```bash
python inference.py \
    --mode single \
    --model_path ../data/emotion_models/best_model.pt \
    --audio_file /path/to/utterance.wav
```

Process single conference:
```bash
python inference.py \
    --mode conference \
    --model_path ../data/emotion_models/best_model.pt \
    --csv_file ../data/audio/conference_2023_01.csv \
    --output ../data/audio/conference_2023_01_emotions.csv
```

Process all conferences:
```bash
python inference.py \
    --mode batch \
    --model_path ../data/emotion_models/best_model.pt \
    --audio_dir ../data/audio \
    --output ../data/conference_emotions.csv
```

## Pipeline Integration

This module integrates with the existing audio processing pipeline:

1. Speaker diarization (WhisperX) - `diarization-asr/whisper.py`
2. Dominant speaker identification - `diarization-asr/dominant_speaker.py`
3. Audio segmentation - `diarization-asr/split_audio_segments.py`
4. Emotion recognition (this module) - `inference.py`

## Output

Utterance-level CSV (per conference):
- Columns: audio_path, start, end, text, speaker, dominant
- Emotion posteriors: emotion_angry, emotion_happy, emotion_neutral, emotion_sad
- Valence posteriors: valence_negative, valence_neutral, valence_positive
- Arousal posteriors: arousal_weak, arousal_neutral, arousal_strong
- Predictions: emotion_predicted, valence_predicted, arousal_predicted

Conference-level CSV (aggregated):
- Averaged posterior probabilities across all Fed Chair utterances per conference
- Format: conference, audio, [emotion posteriors], [valence posteriors], [arousal posteriors]
- Output can be merged with financial data for subsequent economic analysis

## Performance

Expected macro-F1 scores based on dataset benchmarks:

| Dataset | Emotion | Valence | Arousal |
|---------|---------|---------|---------|
| IEMOCAP | 67% | 60% | 57% |
| MSP-Podcast | 56% | 49% | 53% |

## References

- TRILLsson3: Shor et al. (2022)
- WhisperX: Bain et al. (2023)
- IEMOCAP: Busso et al. (2008)
- MSP-Podcast: Lotfian and Busso (2017)
