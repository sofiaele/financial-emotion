"""Dataset loaders for IEMOCAP and MSP-Podcast
Uses parsers.py for dataset parsing and loading
"""

import os
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from preprocessing import AudioPreprocessor
from config import *
from parsers import IEMOCAPParser, MSPPodcastParser, create_combined_dataset


class EmotionDataset(Dataset):
    """Base dataset class for emotion recognition from DataFrame"""

    def __init__(
        self,
        dataframe: pd.DataFrame,
        preprocessor: Optional[AudioPreprocessor] = None
    ):
        """
        Initialize EmotionDataset from DataFrame

        Args:
            dataframe: DataFrame with columns: file, emotion_label, valence_label, arousal_label
            preprocessor: Audio preprocessor instance
        """
        self.df = dataframe.reset_index(drop=True)
        self.preprocessor = preprocessor or AudioPreprocessor()

        # Check for valence and arousal availability
        self.has_valence = 'valence_label' in self.df.columns and not self.df['valence_label'].isna().all()
        self.has_arousal = 'arousal_label' in self.df.columns and not self.df['arousal_label'].isna().all()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Load and preprocess audio
        audio_path = row['file']
        mel_spec = self.preprocessor.process_audio_file(audio_path)
        mel_spec = mel_spec.squeeze(0)  # Remove batch dimension: (time, n_mels)

        # Prepare labels
        labels = {
            'emotion': torch.tensor(int(row['emotion_label']), dtype=torch.long)
        }

        if self.has_valence and not pd.isna(row['valence_label']):
            labels['valence'] = torch.tensor(int(row['valence_label']), dtype=torch.long)

        if self.has_arousal and not pd.isna(row['arousal_label']):
            labels['arousal'] = torch.tensor(int(row['arousal_label']), dtype=torch.long)

        return mel_spec, labels


def load_datasets_from_parsers(
    iemocap_path: Optional[str] = None,
    msp_path: Optional[str] = None,
    msp_csv_path: Optional[str] = None,
    iemocap_split_config: Optional[Dict] = None,
    preprocessor: Optional[AudioPreprocessor] = None
) -> Tuple[EmotionDataset, EmotionDataset, EmotionDataset]:
    """
    Load datasets using parsers and return train/val/test EmotionDatasets

    Args:
        iemocap_path: Path to IEMOCAP root directory
        msp_path: Path to MSP-Podcast root directory
        msp_csv_path: Path to msp.csv file (optional, for MSP-Podcast)
        iemocap_split_config: Split configuration for IEMOCAP
        preprocessor: Audio preprocessor instance

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    # Parse datasets using parsers
    train_df, val_df, test_df = create_combined_dataset(
        iemocap_path=iemocap_path,
        msp_path=msp_path,
        msp_csv_path=msp_csv_path,
        iemocap_split_config=iemocap_split_config
    )

    # Create EmotionDataset instances
    train_dataset = EmotionDataset(train_df, preprocessor=preprocessor)
    val_dataset = EmotionDataset(val_df, preprocessor=preprocessor)
    test_dataset = EmotionDataset(test_df, preprocessor=preprocessor)

    return train_dataset, val_dataset, test_dataset


def load_datasets_from_csv(
    csv_path: str,
    preprocessor: Optional[AudioPreprocessor] = None
) -> Tuple[EmotionDataset, EmotionDataset, EmotionDataset]:
    """
    Load datasets from a pre-generated CSV file

    Args:
        csv_path: Path to CSV file with columns: file, emotion_label, valence_label, arousal_label, split
        preprocessor: Audio preprocessor instance

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    df = pd.read_csv(csv_path)

    train_df = df[df['split'] == 'train'].reset_index(drop=True)
    val_df = df[df['split'] == 'val'].reset_index(drop=True)
    test_df = df[df['split'] == 'test'].reset_index(drop=True)

    train_dataset = EmotionDataset(train_df, preprocessor=preprocessor)
    val_dataset = EmotionDataset(val_df, preprocessor=preprocessor)
    test_dataset = EmotionDataset(test_df, preprocessor=preprocessor)

    return train_dataset, val_dataset, test_dataset


def collate_fn_dynamic(batch):
    """
    Custom collate function for variable-length sequences
    Groups samples by similar lengths (handled by DynamicBatchSampler)

    Args:
        batch: List of (mel_spec, labels) tuples

    Returns:
        Tuple of (padded_mel_specs, labels_dict, lengths)
    """
    mel_specs, labels_list = zip(*batch)

    # Get lengths
    lengths = torch.tensor([mel.shape[0] for mel in mel_specs])

    # Pad sequences
    max_len = lengths.max().item()
    batch_size = len(mel_specs)
    n_mels = mel_specs[0].shape[1]

    padded_mel_specs = torch.zeros(batch_size, max_len, n_mels)
    for i, mel in enumerate(mel_specs):
        padded_mel_specs[i, :mel.shape[0], :] = mel

    # Combine labels - handle missing labels
    combined_labels = {}
    for key in labels_list[0].keys():
        # Check if all samples have this label
        if all(key in labels for labels in labels_list):
            combined_labels[key] = torch.stack([labels[key] for labels in labels_list])

    return padded_mel_specs, combined_labels, lengths


# Backward compatibility aliases
IEMOCAPDataset = EmotionDataset
MSPPodcastDataset = EmotionDataset
