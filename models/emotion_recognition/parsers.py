"""
Dataset parsers for IEMOCAP and MSP-Podcast
Based on ssl_ser project structure, adapted for emotion recognition with valence/arousal
"""

import os
import string
import pandas as pd
from typing import Dict, List, Tuple, Optional
from pathlib import Path


class IEMOCAPParser:
    """Parser for IEMOCAP dataset"""

    def __init__(self, data_path: str):
        """
        Initialize IEMOCAP parser

        Args:
            data_path: Path to IEMOCAP root directory (e.g., /path/to/IEMOCAP/)
        """
        self.data_path = Path(data_path)

        # Emotion mapping
        self.emotion_mapping = {
            "neu": "neutral",
            "sad": "sad",
            "ang": "angry",
            "hap": "happy",
            "exc": "happy",  # excitement mapped to happy
            "fru": "other",  # frustration
            "fea": "other",  # fear
            "sur": "other",  # surprise
            "dis": "other",  # disgust
            "oth": "other"
        }

        # Emotion to integer mapping
        self.emotion_to_int = {
            "angry": 0,
            "happy": 1,
            "neutral": 2,
            "sad": 3,
            "other": 4
        }

        # Valence mapping (1-5 scale)
        self.valence_mapping = {
            range(10, 25): "negative",  # 1.0-2.4
            range(25, 36): "neutral",   # 2.5-3.5
            range(36, 56): "positive"   # 3.6-5.0
        }

        # Arousal mapping (1-5 scale)
        self.arousal_mapping = {
            range(10, 25): "low",       # 1.0-2.4
            range(25, 36): "neutral",   # 2.5-3.5
            range(36, 56): "high"       # 3.6-5.0
        }

        # Valence/Arousal to integer
        self.valence_to_int = {"negative": 0, "neutral": 1, "positive": 2}
        self.arousal_to_int = {"low": 0, "neutral": 1, "high": 2}

        # Gender mapping
        self.gender_mapping = {"F": "female", "M": "male"}

        # Check for available sessions
        self.sessions = []
        for i in range(1, 6):
            session_path = self.data_path / f"Session{i}"
            if session_path.exists():
                self.sessions.append(i)

    def get_utterances_per_dialog(self, session: int) -> Dict[str, List[str]]:
        """Get all utterances for each dialog in a session"""
        dialog_to_utterances = {}
        wav_path = self.data_path / f"Session{session}" / "sentences" / "wav"

        if not wav_path.exists():
            return dialog_to_utterances

        for dialog_dir in wav_path.iterdir():
            if dialog_dir.is_dir():
                dialog = dialog_dir.name
                utterances = [
                    str(wav_file) for wav_file in dialog_dir.glob("*.wav")
                ]
                if utterances:
                    dialog_to_utterances[dialog] = utterances

        return dialog_to_utterances

    def get_annotation_per_utterance(self, utterances: List[str], annotation_file: str) -> Dict[str, Dict]:
        """Get annotations for each utterance from annotation file"""
        annotation_per_utterance = {}

        if not os.path.exists(annotation_file):
            return annotation_per_utterance

        with open(annotation_file, "r") as annfile:
            annfile_list = annfile.read().split()

            for utterance in utterances:
                utterance_name = os.path.splitext(os.path.basename(utterance))[0]

                try:
                    utterance_idx = annfile_list.index(utterance_name)
                except ValueError:
                    continue

                # Extract annotation information
                annotation = {
                    'emotion': annfile_list[utterance_idx + 1],
                    'valence': annfile_list[utterance_idx + 2].strip(string.punctuation),
                    'activation': annfile_list[utterance_idx + 3].strip(string.punctuation),
                    'dominance': annfile_list[utterance_idx + 4].strip(string.punctuation),
                    'start': annfile_list[utterance_idx - 3].strip(string.punctuation),
                    'end': annfile_list[utterance_idx - 1].strip(string.punctuation)
                }

                annotation_per_utterance[utterance] = annotation

        return annotation_per_utterance

    def map_emotion(self, emotion_label: str) -> str:
        """Map emotion label using emotion_mapping"""
        if emotion_label in self.emotion_mapping:
            return self.emotion_mapping[emotion_label]
        return "other"

    def map_continuous_label(self, label: str, mapping_dict: Dict) -> Optional[str]:
        """Map continuous label (valence/arousal) to discrete category"""
        try:
            label_rounded = int(10 * round(float(label), 1))
            for value_range in mapping_dict:
                if label_rounded in value_range:
                    return mapping_dict[value_range]
        except:
            pass
        return None

    def find_speaker_id(self, utterance_path: str) -> str:
        """Extract speaker ID from utterance path"""
        utterance_name = os.path.splitext(os.path.basename(utterance_path))[0]
        # Format: Ses01F_impro01_F000
        session_gender = utterance_name[0:5]  # Ses01
        gender = utterance_name[5]  # F or M
        return f"{session_gender}{gender}"

    def parse(self, split_config: Optional[Dict[str, List[int]]] = None) -> pd.DataFrame:
        """
        Parse IEMOCAP dataset and return DataFrame

        Args:
            split_config: Dictionary with keys 'train', 'val', 'test'
                         and values as lists of session numbers
                         Example: {'train': [1, 2, 4], 'val': [3], 'test': [5]}

        Returns:
            DataFrame with columns: file, emotion, valence, arousal, split, speaker_id
        """
        if split_config is None:
            # Default split
            split_config = {
                'train': [1, 2, 4],
                'val': [3],
                'test': [5]
            }

        all_data = []

        for session in self.sessions:
            # Determine split for this session
            if session in split_config.get('train', []):
                split = 'train'
            elif session in split_config.get('val', []):
                split = 'val'
            elif session in split_config.get('test', []):
                split = 'test'
            else:
                continue  # Skip this session

            # Get dialog-utterance mapping
            dialog_to_utterances = self.get_utterances_per_dialog(session)

            # Get annotations for each dialog
            annotation_path = self.data_path / f"Session{session}" / "dialog" / "EmoEvaluation"

            for dialog, utterances in dialog_to_utterances.items():
                annotation_file = annotation_path / f"{dialog}.txt"
                annotations = self.get_annotation_per_utterance(
                    utterances, str(annotation_file)
                )

                for utterance, annotation in annotations.items():
                    # Map emotion
                    emotion = self.map_emotion(annotation['emotion'])
                    if emotion == "other":
                        continue  # Skip "other" emotions

                    # Map valence and arousal
                    valence = self.map_continuous_label(
                        annotation['valence'], self.valence_mapping
                    )
                    arousal = self.map_continuous_label(
                        annotation['activation'], self.arousal_mapping
                    )

                    if valence is None or arousal is None:
                        continue  # Skip if mapping failed

                    # Get speaker ID
                    speaker_id = self.find_speaker_id(utterance)

                    # Add to data
                    all_data.append({
                        'file': utterance,
                        'emotion': emotion,
                        'emotion_label': self.emotion_to_int[emotion],
                        'valence': valence,
                        'valence_label': self.valence_to_int[valence],
                        'arousal': arousal,
                        'arousal_label': self.arousal_to_int[arousal],
                        'split': split,
                        'speaker_id': speaker_id,
                        'dataset': 'iemocap'
                    })

        df = pd.DataFrame(all_data)
        return df


class MSPPodcastParser:
    """Parser for MSP-Podcast dataset"""

    def __init__(self, data_path: str, csv_path: Optional[str] = None):
        """
        Initialize MSP-Podcast parser

        Args:
            data_path: Path to MSP-Podcast root directory
            csv_path: Path to msp.csv file with columns: file, emotion, split
                     If None, will try to find labels_consensus.csv in data_path
        """
        self.data_path = Path(data_path)
        self.csv_path = csv_path

        # Emotion mapping
        self.emotion_mapping = {
            "neutral": "neutral",
            "happy": "happy",
            "sad": "sad",
            "angry": "angry",
            "anger": "angry",
            "happiness": "happy",
            "sadness": "sad",
            "contempt": "other",
            "disgust": "other",
            "fear": "other",
            "surprise": "other"
        }

        # Emotion to integer
        self.emotion_to_int = {
            "angry": 0,
            "happy": 1,
            "neutral": 2,
            "sad": 3,
            "other": 4
        }

        # Valence/Arousal to integer (for 3-class discretization)
        self.valence_to_int = {"negative": 0, "neutral": 1, "positive": 2}
        self.arousal_to_int = {"low": 0, "neutral": 1, "high": 2}

    def discretize_continuous(self, value: float, thresholds: List[float] = [3.5, 5.5]) -> str:
        """
        Discretize continuous value (1-7 scale) into 3 categories

        Args:
            value: Continuous value (typically 1-7 scale)
            thresholds: Two thresholds for discretization

        Returns:
            Category string: 'negative'/'neutral'/'positive' for valence,
                            'low'/'neutral'/'high' for arousal
        """
        if value < thresholds[0]:
            return 0  # low/negative
        elif value < thresholds[1]:
            return 1  # neutral
        else:
            return 2  # high/positive

    def parse_from_csv(self, csv_path: str) -> pd.DataFrame:
        """
        Parse MSP-Podcast from CSV file

        CSV format: file, emotion, split
        Example:
            /home/user/data/MSP_v1.1/Audios/MSP-PODCAST_0001_0008.wav,neutral,test1

        Args:
            csv_path: Path to CSV file

        Returns:
            DataFrame with columns: file, emotion, valence, arousal, split, dataset
        """
        df = pd.read_csv(csv_path)

        all_data = []

        for _, row in df.iterrows():
            file_path = row['file']
            emotion_raw = row['emotion'].lower()
            split_raw = row['split'].lower()

            # Check if file exists
            if not os.path.exists(file_path):
                continue

            # Map emotion
            emotion = self.emotion_mapping.get(emotion_raw, "other")
            if emotion == "other":
                continue

            # Map split (test1, test2 -> test; train1 -> train; val/dev -> val)
            if 'test' in split_raw:
                split = 'test'
            elif 'dev' in split_raw or 'val' in split_raw:
                split = 'val'
            else:
                split = 'train'

            # For CSV without valence/arousal, set to None
            # These will need to be loaded from labels file if needed
            all_data.append({
                'file': file_path,
                'emotion': emotion,
                'emotion_label': self.emotion_to_int[emotion],
                'valence': None,
                'valence_label': None,
                'arousal': None,
                'arousal_label': None,
                'split': split,
                'speaker_id': None,
                'dataset': 'msp_podcast'
            })

        return pd.DataFrame(all_data)

    def parse_from_labels_file(self, labels_file: Optional[str] = None) -> pd.DataFrame:
        """
        Parse MSP-Podcast from official labels_consensus.csv

        Args:
            labels_file: Path to labels_consensus.csv
                        If None, looks in data_path/Labels/labels_consensus.csv

        Returns:
            DataFrame with columns: file, emotion, valence, arousal, split, dataset
        """
        if labels_file is None:
            labels_file = self.data_path / "Labels" / "labels_consensus.csv"

        if not os.path.exists(labels_file):
            raise FileNotFoundError(f"Labels file not found: {labels_file}")

        df = pd.read_csv(labels_file)

        all_data = []

        for _, row in df.iterrows():
            # Get file info (FileName already includes .wav extension)
            file_name = row['FileName']

            # Construct full path
            audio_path = self.data_path / "Audios" / file_name
            if not os.path.exists(audio_path):
                continue

            # Get emotion (EmoClass) - single letter codes: N, A, H, S
            emotion_code = str(row.get('EmoClass', '')).strip().upper()

            # Map single letter codes to full emotion names
            emotion_code_map = {
                'N': 'neutral',
                'A': 'angry',
                'H': 'happy',
                'S': 'sad'
            }

            emotion = emotion_code_map.get(emotion_code, None)
            if emotion is None:
                continue

            # Get valence and arousal
            valence_val = row.get('EmoVal', None)
            arousal_val = row.get('EmoAct', None)

            if pd.isna(valence_val) or pd.isna(arousal_val):
                continue

            # Discretize valence and arousal (1-7 scale)
            valence_cat = self.discretize_continuous(float(valence_val))
            arousal_cat = self.discretize_continuous(float(arousal_val))

            # Get split (Train, Test1, Test2, Development)
            split_raw = str(row.get('Split_Set', 'Train')).strip()
            if split_raw in ['Test1', 'Test2']:
                split = 'test'
            elif split_raw == 'Development':
                split = 'val'
            else:
                split = 'train'

            # Get speaker ID and gender if available
            speaker_id = row.get('SpkrID', None)
            gender = row.get('Gender', None)

            all_data.append({
                'file': str(audio_path),
                'emotion': emotion,
                'emotion_label': self.emotion_to_int[emotion],
                'valence': ['negative', 'neutral', 'positive'][valence_cat],
                'valence_label': valence_cat,
                'arousal': ['low', 'neutral', 'high'][arousal_cat],
                'arousal_label': arousal_cat,
                'split': split,
                'speaker_id': speaker_id,
                'gender': gender,
                'dataset': 'msp_podcast'
            })

        return pd.DataFrame(all_data)

    def parse(self) -> pd.DataFrame:
        """
        Parse MSP-Podcast dataset

        Returns:
            DataFrame with parsed data
        """
        if self.csv_path is not None:
            return self.parse_from_csv(self.csv_path)
        else:
            return self.parse_from_labels_file()


def create_combined_dataset(
    iemocap_path: Optional[str] = None,
    msp_path: Optional[str] = None,
    msp_csv_path: Optional[str] = None,
    iemocap_split_config: Optional[Dict] = None,
    output_csv: Optional[str] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Create combined dataset from IEMOCAP and MSP-Podcast

    Args:
        iemocap_path: Path to IEMOCAP root directory
        msp_path: Path to MSP-Podcast root directory
        msp_csv_path: Path to msp.csv file (optional)
        iemocap_split_config: Split configuration for IEMOCAP
        output_csv: Path to save combined CSV (optional)

    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    all_data = []

    # Parse IEMOCAP if path provided
    if iemocap_path is not None:
        print("Parsing IEMOCAP...")
        iemocap_parser = IEMOCAPParser(iemocap_path)
        iemocap_df = iemocap_parser.parse(split_config=iemocap_split_config)
        all_data.append(iemocap_df)
        print(f"  Found {len(iemocap_df)} IEMOCAP samples")

    # Parse MSP-Podcast if path provided
    if msp_path is not None:
        print("Parsing MSP-Podcast...")
        msp_parser = MSPPodcastParser(msp_path, csv_path=msp_csv_path)
        msp_df = msp_parser.parse()
        all_data.append(msp_df)
        print(f"  Found {len(msp_df)} MSP-Podcast samples")

    if not all_data:
        raise ValueError("At least one dataset path must be provided")

    # Combine datasets
    combined_df = pd.concat(all_data, ignore_index=True)

    # Split into train/val/test
    train_df = combined_df[combined_df['split'] == 'train'].reset_index(drop=True)
    val_df = combined_df[combined_df['split'] == 'val'].reset_index(drop=True)
    test_df = combined_df[combined_df['split'] == 'test'].reset_index(drop=True)

    print(f"\nCombined dataset statistics:")
    print(f"  Train: {len(train_df)} samples")
    print(f"  Val: {len(val_df)} samples")
    print(f"  Test: {len(test_df)} samples")
    print(f"  Total: {len(combined_df)} samples")

    # Save to CSV if requested
    if output_csv is not None:
        combined_df.to_csv(output_csv, index=False)
        print(f"\nSaved combined dataset to: {output_csv}")

    return train_df, val_df, test_df


if __name__ == "__main__":
    # Example usage
    import argparse

    parser = argparse.ArgumentParser(description="Parse IEMOCAP and MSP-Podcast datasets")
    parser.add_argument('--iemocap_path', type=str, help='Path to IEMOCAP root directory')
    parser.add_argument('--msp_path', type=str, help='Path to MSP-Podcast root directory')
    parser.add_argument('--msp_csv', type=str, help='Path to msp.csv file')
    parser.add_argument('--output_csv', type=str, default='combined_dataset.csv',
                       help='Path to output CSV file')

    args = parser.parse_args()

    train_df, val_df, test_df = create_combined_dataset(
        iemocap_path=args.iemocap_path,
        msp_path=args.msp_path,
        msp_csv_path=args.msp_csv,
        output_csv=args.output_csv
    )

    print("\nEmotion distribution:")
    print(train_df['emotion'].value_counts())
