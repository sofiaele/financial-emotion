"""Emotion recognition module for Fed press conferences"""

from .model import EmotionRecognitionModel, MultitaskLoss
from .preprocessing import AudioPreprocessor
from .inference import EmotionInference, aggregate_conference_emotions
from .config import *

__all__ = [
    'EmotionRecognitionModel',
    'MultitaskLoss',
    'AudioPreprocessor',
    'EmotionInference',
    'aggregate_conference_emotions',
]
