"""Audio preprocessing utilities for emotion recognition"""

import torch
import torchaudio
import numpy as np
from typing import Tuple, Optional
import librosa
from config import *


class AudioPreprocessor:
    """Handles audio loading and mel-spectrogram conversion"""

    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        n_mels: int = N_MELS,
        hop_length: int = HOP_LENGTH,
        win_length: int = WIN_LENGTH,
        n_fft: int = N_FFT
    ):
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_fft = n_fft

        # Initialize mel-spectrogram transform
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mels,
            power=2.0
        )

    def load_audio(self, audio_path: str, target_sr: Optional[int] = None) -> Tuple[torch.Tensor, int]:
        """
        Load audio file and optionally resample

        Args:
            audio_path: Path to audio file
            target_sr: Target sample rate (uses self.sample_rate if None)

        Returns:
            Tuple of (audio tensor, sample rate)
        """
        if target_sr is None:
            target_sr = self.sample_rate

        # Load audio using librosa for better compatibility
        audio, sr = librosa.load(audio_path, sr=target_sr, mono=True)
        audio_tensor = torch.from_numpy(audio).float()

        # Add batch dimension if needed
        if audio_tensor.dim() == 1:
            audio_tensor = audio_tensor.unsqueeze(0)

        return audio_tensor, sr

    def audio_to_melspectrogram(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Convert audio waveform to mel-spectrogram

        Args:
            audio: Audio tensor of shape (batch, samples) or (samples,)

        Returns:
            Mel-spectrogram tensor of shape (batch, time_frames, n_mels)
        """
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)

        # Compute mel-spectrogram
        mel_spec = self.mel_transform(audio)

        # Convert to log scale (add small epsilon for numerical stability)
        log_mel_spec = torch.log(mel_spec + 1e-9)

        # Transpose to (batch, time, freq) format
        log_mel_spec = log_mel_spec.transpose(1, 2)

        return log_mel_spec

    def process_audio_file(self, audio_path: str) -> torch.Tensor:
        """
        Complete preprocessing pipeline: load audio and convert to mel-spectrogram

        Args:
            audio_path: Path to audio file

        Returns:
            Mel-spectrogram tensor of shape (1, time_frames, n_mels)
        """
        audio, _ = self.load_audio(audio_path)
        mel_spec = self.audio_to_melspectrogram(audio)
        return mel_spec


def pad_sequence(sequences, padding_value=0.0):
    """
    Pad a list of variable length tensors with padding_value

    Args:
        sequences: List of tensors with shape (time, features)
        padding_value: Value to use for padding

    Returns:
        Tuple of (padded tensor, lengths tensor)
    """
    lengths = torch.tensor([seq.shape[0] for seq in sequences])
    max_len = lengths.max().item()

    batch_size = len(sequences)
    feature_dim = sequences[0].shape[-1]

    padded = torch.full((batch_size, max_len, feature_dim), padding_value)

    for i, seq in enumerate(sequences):
        length = seq.shape[0]
        padded[i, :length, :] = seq

    return padded, lengths


def create_attention_mask(lengths, max_length):
    """
    Create attention mask for padded sequences

    Args:
        lengths: Tensor of sequence lengths
        max_length: Maximum sequence length

    Returns:
        Boolean mask tensor (True for real data, False for padding)
    """
    batch_size = len(lengths)
    mask = torch.zeros(batch_size, max_length, dtype=torch.bool)

    for i, length in enumerate(lengths):
        mask[i, :length] = True

    return mask
