"""Dataset loaders for IEMOCAP and MSP-Podcast"""

import os
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from preprocessing import AudioPreprocessor
from config import *


class EmotionDataset(Dataset):
    """Base dataset class for emotion recognition"""

    def __init__(
        self,
        audio_paths: List[str],
        emotion_labels: List[int],
        valence_labels: Optional[List[int]] = None,
        arousal_labels: Optional[List[int]] = None,
        preprocessor: Optional[AudioPreprocessor] = None
    ):
        self.audio_paths = audio_paths
        self.emotion_labels = emotion_labels
        self.valence_labels = valence_labels
        self.arousal_labels = arousal_labels

        self.preprocessor = preprocessor or AudioPreprocessor()

    def __len__(self):
        return len(self.audio_paths)

    def __getitem__(self, idx):
        # Load and preprocess audio
        mel_spec = self.preprocessor.process_audio_file(self.audio_paths[idx])
        mel_spec = mel_spec.squeeze(0)  # Remove batch dimension: (time, n_mels)

        # Prepare labels
        labels = {
            'emotion': torch.tensor(self.emotion_labels[idx], dtype=torch.long)
        }

        if self.valence_labels is not None:
            labels['valence'] = torch.tensor(self.valence_labels[idx], dtype=torch.long)

        if self.arousal_labels is not None:
            labels['arousal'] = torch.tensor(self.arousal_labels[idx], dtype=torch.long)

        return mel_spec, labels


class IEMOCAPDataset(EmotionDataset):
    """Dataset loader for IEMOCAP"""

    @staticmethod
    def load_iemocap(
        root_path: str,
        val_session: str = IEMOCAP_VAL_SESSION,
        test_session: str = IEMOCAP_TEST_SESSION,
        preprocessor: Optional[AudioPreprocessor] = None
    ) -> Tuple['IEMOCAPDataset', 'IEMOCAPDataset', 'IEMOCAPDataset']:
        """
        Load IEMOCAP dataset and split into train/val/test

        Args:
            root_path: Path to IEMOCAP root directory
            val_session: Session to use for validation (e.g., "Session3")
            test_session: Session to use for test (e.g., "Session5")
            preprocessor: Audio preprocessor instance

        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset)
        """
        # Emotion mapping for IEMOCAP
        emotion_map = {
            'ang': 0,  # angry
            'hap': 1,  # happy
            'neu': 2,  # neutral
            'sad': 3,  # sad
            'exc': 1,  # excitement -> happy
        }

        # Valence mapping: 1-5 scale -> 3 classes
        valence_map = lambda x: 0 if x <= 2 else (1 if x == 3 else 2)  # negative, neutral, positive

        # Arousal mapping: 1-5 scale -> 3 classes
        arousal_map = lambda x: 0 if x <= 2 else (1 if x == 3 else 2)  # weak, neutral, strong

        train_data = {'audio': [], 'emotion': [], 'valence': [], 'arousal': []}
        val_data = {'audio': [], 'emotion': [], 'valence': [], 'arousal': []}
        test_data = {'audio': [], 'emotion': [], 'valence': [], 'arousal': []}

        # Iterate through sessions
        sessions_dir = os.path.join(root_path, "Session")
        for session_num in range(1, 6):  # Sessions 1-5
            session_name = f"Session{session_num}"
            session_path = os.path.join(sessions_dir, session_name)

            if not os.path.exists(session_path):
                continue

            # Load emotion labels from EmoEvaluation files
            eval_path = os.path.join(session_path, "dialog", "EmoEvaluation")
            if not os.path.exists(eval_path):
                continue

            # Parse emotion evaluation files
            for eval_file in os.listdir(eval_path):
                if not eval_file.endswith('.txt'):
                    continue

                eval_file_path = os.path.join(eval_path, eval_file)
                with open(eval_file_path, 'r') as f:
                    for line in f:
                        if line.startswith('['):
                            parts = line.strip().split('\t')
                            if len(parts) < 4:
                                continue

                            # Extract information
                            timestamp = parts[0].strip('[]')
                            emotion = parts[1]
                            valence_arousal = parts[2].strip('[]').split(',')

                            # Skip if emotion not in our map
                            if emotion not in emotion_map:
                                continue

                            # Get valence and arousal scores
                            try:
                                valence_score = float(valence_arousal[0])
                                arousal_score = float(valence_arousal[1])
                            except:
                                continue

                            # Construct audio file path
                            dialog_name = eval_file.replace('.txt', '')
                            audio_file = f"{timestamp}.wav"
                            audio_path = os.path.join(
                                session_path, "sentences", "wav",
                                dialog_name, audio_file
                            )

                            if not os.path.exists(audio_path):
                                continue

                            # Determine split
                            if session_name == val_session:
                                split_data = val_data
                            elif session_name == test_session:
                                split_data = test_data
                            else:
                                split_data = train_data

                            # Add to appropriate split
                            split_data['audio'].append(audio_path)
                            split_data['emotion'].append(emotion_map[emotion])
                            split_data['valence'].append(valence_map(valence_score))
                            split_data['arousal'].append(arousal_map(arousal_score))

        # Create datasets
        train_dataset = IEMOCAPDataset(
            train_data['audio'], train_data['emotion'],
            train_data['valence'], train_data['arousal'], preprocessor
        )
        val_dataset = IEMOCAPDataset(
            val_data['audio'], val_data['emotion'],
            val_data['valence'], val_data['arousal'], preprocessor
        )
        test_dataset = IEMOCAPDataset(
            test_data['audio'], test_data['emotion'],
            test_data['valence'], test_data['arousal'], preprocessor
        )

        return train_dataset, val_dataset, test_dataset


class MSPPodcastDataset(EmotionDataset):
    """Dataset loader for MSP-Podcast"""

    @staticmethod
    def load_msp_podcast(
        root_path: str,
        preprocessor: Optional[AudioPreprocessor] = None
    ) -> Tuple['MSPPodcastDataset', 'MSPPodcastDataset', 'MSPPodcastDataset']:
        """
        Load MSP-Podcast dataset using official splits

        Args:
            root_path: Path to MSP-Podcast root directory
            preprocessor: Audio preprocessor instance

        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset)
        """
        # Emotion mapping for MSP-Podcast
        emotion_map = {
            'A': 0,  # Angry
            'H': 1,  # Happy
            'N': 2,  # Neutral
            'S': 3,  # Sad
        }

        # Valence/Arousal: MSP-Podcast uses continuous scale, discretize to 3 classes
        def discretize_continuous(value, thresholds=[3.5, 5.5]):
            """Discretize 1-7 scale into 3 classes"""
            if value < thresholds[0]:
                return 0  # low
            elif value < thresholds[1]:
                return 1  # neutral
            else:
                return 2  # high

        train_data = {'audio': [], 'emotion': [], 'valence': [], 'arousal': []}
        val_data = {'audio': [], 'emotion': [], 'valence': [], 'arousal': []}
        test_data = {'audio': [], 'emotion': [], 'valence': [], 'arousal': []}

        # Load labels file
        labels_file = os.path.join(root_path, "Labels", "labels_consensus.csv")
        if not os.path.exists(labels_file):
            raise FileNotFoundError(f"Labels file not found: {labels_file}")

        df = pd.read_csv(labels_file)

        # Process each sample
        for _, row in df.iterrows():
            # Get file information
            file_name = row['FileName']
            split = row['Split_Set']  # Train, Development, Test1, Test2

            # Get labels
            emotion = row.get('EmoClass', None)
            valence = row.get('EmoVal', None)
            arousal = row.get('EmoAct', None)

            # Skip if emotion not in our map or labels missing
            if emotion not in emotion_map or pd.isna(valence) or pd.isna(arousal):
                continue

            # Construct audio path
            audio_path = os.path.join(root_path, "Audio", f"{file_name}.wav")
            if not os.path.exists(audio_path):
                continue

            # Determine split (combine Test1 and Test2 into test)
            if split == 'Train':
                split_data = train_data
            elif split == 'Development':
                split_data = val_data
            else:  # Test1 or Test2
                split_data = test_data

            # Add to appropriate split
            split_data['audio'].append(audio_path)
            split_data['emotion'].append(emotion_map[emotion])
            split_data['valence'].append(discretize_continuous(valence))
            split_data['arousal'].append(discretize_continuous(arousal))

        # Create datasets
        train_dataset = MSPPodcastDataset(
            train_data['audio'], train_data['emotion'],
            train_data['valence'], train_data['arousal'], preprocessor
        )
        val_dataset = MSPPodcastDataset(
            val_data['audio'], val_data['emotion'],
            val_data['valence'], val_data['arousal'], preprocessor
        )
        test_dataset = MSPPodcastDataset(
            test_data['audio'], test_data['emotion'],
            test_data['valence'], test_data['arousal'], preprocessor
        )

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

    # Combine labels
    combined_labels = {}
    for key in labels_list[0].keys():
        combined_labels[key] = torch.stack([labels[key] for labels in labels_list])

    return padded_mel_specs, combined_labels, lengths
