"""Multitask emotion recognition model based on TRILLsson3"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
from transformers import AutoModel
from config import *


class TRILLssonBackbone(nn.Module):
    """TRILLsson3 feature extractor backbone from Hugging Face

    TRILLsson3 is a CNN-based architecture inspired by EfficientNetV2-S,
    pre-trained on large-scale speech data. We use the pre-trained model
    from Hugging Face and fine-tune it on emotion recognition tasks.
    """

    def __init__(self, embedding_dim: int = EMBEDDING_DIM):
        super().__init__()
        self.embedding_dim = embedding_dim

        # Load pre-trained TRILLsson3 model from Hugging Face
        # This is the original TRILLsson3 model as used in the paper
        self.model = AutoModel.from_pretrained("vumichien/nonsemantic-speech-trillsson3")

    def forward(self, mel_spec: torch.Tensor) -> torch.Tensor:
        """
        Extract embeddings from mel-spectrograms

        Args:
            mel_spec: Tensor of shape (batch, time, n_mels)

        Returns:
            Embeddings of shape (batch, embedding_dim)
        """
        # TRILLsson expects mel-spectrograms as input
        # The Hugging Face model handles the processing
        outputs = self.model(mel_spec)

        # Get the pooled output (mean pooling over time dimension)
        # outputs.last_hidden_state has shape (batch, time, embedding_dim)
        embeddings = outputs.last_hidden_state.mean(dim=1)  # (batch, embedding_dim)

        return embeddings


class EmotionRecognitionModel(nn.Module):
    """
    Multitask emotion recognition model as described in the paper.

    Architecture:
    - TRILLsson3 backbone (1024-dimensional embeddings)
    - Three task-specific linear heads:
      * Emotion: 1024×4 (angry, happy, neutral, sad)
      * Valence: 1024×3 (negative, neutral, positive)
      * Arousal: 1024×3 (weak, neutral, strong)
    """

    def __init__(
        self,
        embedding_dim: int = EMBEDDING_DIM,
        num_emotion_classes: int = NUM_EMOTION_CLASSES,
        num_valence_classes: int = NUM_VALENCE_CLASSES,
        num_arousal_classes: int = NUM_AROUSAL_CLASSES,
        freeze_backbone: bool = False
    ):
        super().__init__()

        # TRILLsson3 backbone
        self.backbone = TRILLssonBackbone(embedding_dim=embedding_dim)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Task-specific heads (simple linear classifiers as per paper)
        self.emotion_head = nn.Linear(embedding_dim, num_emotion_classes)
        self.valence_head = nn.Linear(embedding_dim, num_valence_classes)
        self.arousal_head = nn.Linear(embedding_dim, num_arousal_classes)

    def forward(
        self,
        mel_spec: torch.Tensor,
        return_embeddings: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model

        Args:
            mel_spec: Mel-spectrogram tensor of shape (batch, time, n_mels)
            return_embeddings: Whether to return intermediate embeddings

        Returns:
            Dictionary with logits for each task:
            - 'emotion': (batch, num_emotion_classes)
            - 'valence': (batch, num_valence_classes)
            - 'arousal': (batch, num_arousal_classes)
            - 'embeddings': (batch, embedding_dim) if return_embeddings=True
        """
        # Extract embeddings
        embeddings = self.backbone(mel_spec)

        # Compute logits for each task
        emotion_logits = self.emotion_head(embeddings)
        valence_logits = self.valence_head(embeddings)
        arousal_logits = self.arousal_head(embeddings)

        outputs = {
            'emotion': emotion_logits,
            'valence': valence_logits,
            'arousal': arousal_logits,
        }

        if return_embeddings:
            outputs['embeddings'] = embeddings

        return outputs

    def predict_probabilities(self, mel_spec: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Get probability distributions for each task

        Args:
            mel_spec: Mel-spectrogram tensor

        Returns:
            Dictionary with probabilities for each task
        """
        logits = self.forward(mel_spec)

        return {
            'emotion': F.softmax(logits['emotion'], dim=-1),
            'valence': F.softmax(logits['valence'], dim=-1),
            'arousal': F.softmax(logits['arousal'], dim=-1),
        }


class MultitaskLoss(nn.Module):
    """Combined loss for multitask learning"""

    def __init__(self, task_weights: Optional[Dict[str, float]] = None):
        super().__init__()

        # Default equal weights
        self.task_weights = task_weights or {
            'emotion': 1.0,
            'valence': 1.0,
            'arousal': 1.0
        }

        self.criterion = nn.CrossEntropyLoss(reduction='none')

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute multitask loss

        Args:
            predictions: Dictionary of logits from model
            targets: Dictionary of target labels
            mask: Optional mask for valid samples

        Returns:
            Dictionary with total loss and individual task losses
        """
        losses = {}
        total_loss = 0.0

        for task in ['emotion', 'valence', 'arousal']:
            if task in predictions and task in targets:
                loss = self.criterion(predictions[task], targets[task])

                # Apply mask if provided
                if mask is not None:
                    loss = loss * mask
                    loss = loss.sum() / mask.sum()
                else:
                    loss = loss.mean()

                losses[f'{task}_loss'] = loss
                total_loss += self.task_weights[task] * loss

        losses['total_loss'] = total_loss

        return losses
