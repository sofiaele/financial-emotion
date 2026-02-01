#!/bin/bash
# Example script for running emotion recognition inference on Fed conferences

# Set paths
MODEL_PATH="../data/emotion_models/best_model.pt"
AUDIO_DIR="../data/audio"
OUTPUT_CSV="../data/conference_emotions.csv"

# Check if model exists
if [ ! -f "$MODEL_PATH" ]; then
    echo "Error: Model not found at $MODEL_PATH"
    echo "Please train the model first using train_emotion_model.sh"
    exit 1
fi

# Check if audio directory exists
if [ ! -d "$AUDIO_DIR" ]; then
    echo "Error: Audio directory not found at $AUDIO_DIR"
    echo "Please run the diarization pipeline first (steps 1-7 in README)"
    exit 1
fi

echo "Running emotion recognition inference on all conferences..."
echo "Model: $MODEL_PATH"
echo "Audio dir: $AUDIO_DIR"
echo "Output: $OUTPUT_CSV"

python ../models/emotion_recognition/inference.py \
    --mode batch \
    --model_path $MODEL_PATH \
    --audio_dir $AUDIO_DIR \
    --output $OUTPUT_CSV \
    --device cuda

echo "Inference completed!"
echo "Conference-level results saved to: $OUTPUT_CSV"
echo "Utterance-level results saved in $AUDIO_DIR/*_emotions.csv"
