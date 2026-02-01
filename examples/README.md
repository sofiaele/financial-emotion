# Example Scripts

Example scripts for training and running emotion recognition inference.

## train_emotion_model.sh

Trains the emotion recognition model on IEMOCAP and MSP-Podcast datasets.

Prerequisites:
- Download IEMOCAP: https://sail.usc.edu/iemocap/
- Download MSP-Podcast: https://ecs.utdallas.edu/research/researchlabs/msp-lab/MSP-Podcast.html
- Update paths in `models/emotion_recognition/config.py`

Usage:
```bash
cd examples
chmod +x train_emotion_model.sh
./train_emotion_model.sh
```

## run_inference.sh

Runs emotion recognition inference on all Fed conference audio files.

Prerequisites:
- Trained model checkpoint
- Completed diarization pipeline (steps 1-7 in main README)

Usage:
```bash
cd examples
chmod +x run_inference.sh
./run_inference.sh
```

## Configuration

Edit paths in each script as needed:
- `IEMOCAP_PATH`: IEMOCAP dataset location
- `MSP_PODCAST_PATH`: MSP-Podcast dataset location
- `MODEL_PATH`: Model checkpoint location
- `AUDIO_DIR`: Fed conference audio directory
- `OUTPUT_CSV`: Output file for aggregated results
