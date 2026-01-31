#!/bin/bash
# Example script for training emotion recognition model

# NOTE: You need to download IEMOCAP and/or MSP-Podcast datasets first
# Update the paths in models/emotion_recognition/config.py

# Set paths
IEMOCAP_PATH="/path/to/IEMOCAP"
MSP_PODCAST_PATH="/path/to/MSP-Podcast"
CHECKPOINT_DIR="../data/emotion_models"

# Create checkpoint directory
mkdir -p $CHECKPOINT_DIR

echo "Training on IEMOCAP dataset..."
python ../models/emotion_recognition/train.py \
    --dataset iemocap \
    --data_path $IEMOCAP_PATH \
    --batch_size 32 \
    --learning_rate 0.0001 \
    --num_epochs 100 \
    --patience 15 \
    --use_dynamic_sampler \
    --checkpoint_dir $CHECKPOINT_DIR

echo "Training on MSP-Podcast dataset..."
python ../models/emotion_recognition/train.py \
    --dataset msp-podcast \
    --data_path $MSP_PODCAST_PATH \
    --batch_size 32 \
    --learning_rate 0.0001 \
    --num_epochs 100 \
    --patience 15 \
    --checkpoint_dir $CHECKPOINT_DIR

echo "Training completed! Model saved to $CHECKPOINT_DIR/best_model.pt"
