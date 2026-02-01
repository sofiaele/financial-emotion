"""Inference pipeline for emotion recognition on Fed conference utterances"""

import os
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
import argparse
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

from model import EmotionRecognitionModel
from preprocessing import AudioPreprocessor
from config import *


class EmotionInference:
    """Inference class for emotion recognition"""

    def __init__(
        self,
        model_path: str,
        device: str = 'cuda'
    ):
        """
        Initialize inference pipeline

        Args:
            model_path: Path to trained model checkpoint
            device: Device to run inference on ('cuda' or 'cpu')
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.preprocessor = AudioPreprocessor()

        # Load model
        print(f"Loading model from {model_path}...")
        self.model = EmotionRecognitionModel()

        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

        print(f"Model loaded successfully on {self.device}")

    def predict_single_utterance(self, audio_path: str) -> Dict[str, np.ndarray]:
        """
        Predict emotion probabilities for a single utterance

        Args:
            audio_path: Path to audio file

        Returns:
            Dictionary with probability distributions for each task
        """
        # Preprocess audio
        mel_spec = self.preprocessor.process_audio_file(audio_path)
        mel_spec = mel_spec.to(self.device)

        # Get predictions
        with torch.no_grad():
            probabilities = self.model.predict_probabilities(mel_spec)

        # Convert to numpy
        result = {}
        for task, probs in probabilities.items():
            result[task] = probs.cpu().numpy()[0]  # Remove batch dimension

        return result

    def predict_batch(self, audio_paths: List[str], batch_size: int = 16) -> Dict[str, np.ndarray]:
        """
        Predict emotion probabilities for a batch of utterances

        Args:
            audio_paths: List of paths to audio files
            batch_size: Batch size for inference

        Returns:
            Dictionary with arrays of probability distributions for each task
        """
        all_predictions = {'emotion': [], 'valence': [], 'arousal': []}

        # Process in batches
        for i in tqdm(range(0, len(audio_paths), batch_size), desc="Processing utterances"):
            batch_paths = audio_paths[i:i + batch_size]

            # Preprocess batch
            mel_specs = []
            for path in batch_paths:
                try:
                    mel_spec = self.preprocessor.process_audio_file(path)
                    mel_specs.append(mel_spec.squeeze(0))  # Remove batch dim
                except Exception as e:
                    print(f"Warning: Failed to process {path}: {e}")
                    # Add dummy prediction
                    for task in all_predictions.keys():
                        all_predictions[task].append(None)
                    continue

            if not mel_specs:
                continue

            # Pad sequences
            lengths = [mel.shape[0] for mel in mel_specs]
            max_len = max(lengths)
            n_mels = mel_specs[0].shape[1]

            padded_batch = torch.zeros(len(mel_specs), max_len, n_mels)
            for j, mel in enumerate(mel_specs):
                padded_batch[j, :mel.shape[0], :] = mel

            padded_batch = padded_batch.to(self.device)

            # Get predictions
            with torch.no_grad():
                probabilities = self.model.predict_probabilities(padded_batch)

            # Store predictions
            for task in all_predictions.keys():
                probs = probabilities[task].cpu().numpy()
                all_predictions[task].extend(probs)

        # Convert to arrays
        for task in all_predictions.keys():
            all_predictions[task] = np.array(all_predictions[task])

        return all_predictions

    def process_conference_csv(
        self,
        csv_path: str,
        audio_column: str = 'audio_path',
        output_csv: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Process a conference CSV file with utterance paths

        Args:
            csv_path: Path to CSV file containing utterance information
            audio_column: Name of column containing audio paths
            output_csv: Optional path to save output CSV

        Returns:
            DataFrame with added emotion probability columns
        """
        # Load CSV
        df = pd.read_csv(csv_path)

        # Get audio paths
        audio_paths = []
        audio_base_dir = os.path.dirname(csv_path)

        for audio_path in df[audio_column]:
            # Handle relative paths
            if not os.path.isabs(audio_path):
                full_path = os.path.join(audio_base_dir, audio_path)
            else:
                full_path = audio_path
            audio_paths.append(full_path)

        # Run inference
        predictions = self.predict_batch(audio_paths)

        # Add predictions to dataframe
        # Emotion probabilities (angry, happy, neutral, sad)
        for i, label in enumerate(EMOTION_LABELS):
            df[f'emotion_{label}'] = predictions['emotion'][:, i]

        # Valence probabilities (negative, neutral, positive)
        for i, label in enumerate(VALENCE_LABELS):
            df[f'valence_{label}'] = predictions['valence'][:, i]

        # Arousal probabilities (weak, neutral, strong)
        for i, label in enumerate(AROUSAL_LABELS):
            df[f'arousal_{label}'] = predictions['arousal'][:, i]

        # Add predicted classes
        df['emotion_predicted'] = [EMOTION_LABELS[i] for i in np.argmax(predictions['emotion'], axis=1)]
        df['valence_predicted'] = [VALENCE_LABELS[i] for i in np.argmax(predictions['valence'], axis=1)]
        df['arousal_predicted'] = [AROUSAL_LABELS[i] for i in np.argmax(predictions['arousal'], axis=1)]

        # Save if output path provided
        if output_csv:
            df.to_csv(output_csv, index=False, float_format='%.4f')
            print(f"Results saved to {output_csv}")

        return df


def aggregate_conference_emotions(df: pd.DataFrame) -> Dict[str, float]:
    """
    Aggregate emotion probabilities across all utterances in a conference

    As per the paper: "We aggregate these probabilities across each press
    conference by averaging them over all utterances."

    Args:
        df: DataFrame with emotion probabilities for each utterance

    Returns:
        Dictionary with averaged probabilities for the conference
    """
    # Filter for dominant speaker (banker) if available
    if 'dominant' in df.columns:
        df_banker = df[df['dominant'] == 'banker']
        if len(df_banker) > 0:
            df = df_banker

    aggregated = {}

    # Aggregate emotion probabilities
    for label in EMOTION_LABELS:
        col_name = f'emotion_{label}'
        if col_name in df.columns:
            aggregated[col_name] = df[col_name].mean()

    # Aggregate valence probabilities
    for label in VALENCE_LABELS:
        col_name = f'valence_{label}'
        if col_name in df.columns:
            aggregated[col_name] = df[col_name].mean()

    # Aggregate arousal probabilities
    for label in AROUSAL_LABELS:
        col_name = f'arousal_{label}'
        if col_name in df.columns:
            aggregated[col_name] = df[col_name].mean()

    return aggregated


def process_all_conferences(
    audio_dir: str,
    model_path: str,
    output_csv: str
):
    """
    Process all conference CSV files in a directory

    Args:
        audio_dir: Directory containing conference CSV files and audio
        model_path: Path to trained model checkpoint
        output_csv: Path to save aggregated conference-level results
    """
    # Initialize inference
    inference = EmotionInference(model_path)

    # Find all CSV files (one per conference)
    csv_files = [f for f in os.listdir(audio_dir) if f.endswith('.csv')]

    print(f"Found {len(csv_files)} conference files")

    # Process each conference
    conference_results = []

    for csv_file in tqdm(csv_files, desc="Processing conferences"):
        csv_path = os.path.join(audio_dir, csv_file)

        try:
            # Process utterances
            df_utterances = inference.process_conference_csv(csv_path)

            # Aggregate to conference level
            aggregated = aggregate_conference_emotions(df_utterances)

            # Add conference metadata
            conference_name = csv_file.replace('.csv', '')
            aggregated['conference'] = conference_name

            # Extract audio filename for identification
            audio_file = csv_file.replace('.csv', '.wav')
            aggregated['audio'] = audio_file

            conference_results.append(aggregated)

            # Save utterance-level results
            utterance_output = csv_path.replace('.csv', '_emotions.csv')
            df_utterances.to_csv(utterance_output, index=False, float_format='%.4f')

        except Exception as e:
            print(f"Error processing {csv_file}: {e}")
            continue

    # Create conference-level dataframe
    df_conferences = pd.DataFrame(conference_results)

    # Reorder columns
    cols = ['conference', 'audio']
    emotion_cols = [c for c in df_conferences.columns if c.startswith('emotion_')]
    valence_cols = [c for c in df_conferences.columns if c.startswith('valence_')]
    arousal_cols = [c for c in df_conferences.columns if c.startswith('arousal_')]
    df_conferences = df_conferences[cols + emotion_cols + valence_cols + arousal_cols]

    # Save conference-level results
    df_conferences.to_csv(output_csv, index=False, float_format='%.4f')
    print(f"\nConference-level results saved to {output_csv}")
    print(f"Processed {len(conference_results)} conferences")

    return df_conferences


def main(args):
    """Main inference function"""

    if args.mode == 'single':
        # Single file inference
        inference = EmotionInference(
            args.model_path,
            device=args.device
        )

        result = inference.predict_single_utterance(args.audio_file)

        print("\nPrediction Results:")
        print(f"\nEmotion probabilities:")
        for i, label in enumerate(EMOTION_LABELS):
            print(f"  {label}: {result['emotion'][i]:.4f}")

        print(f"\nValence probabilities:")
        for i, label in enumerate(VALENCE_LABELS):
            print(f"  {label}: {result['valence'][i]:.4f}")

        print(f"\nArousal probabilities:")
        for i, label in enumerate(AROUSAL_LABELS):
            print(f"  {label}: {result['arousal'][i]:.4f}")

    elif args.mode == 'conference':
        # Single conference inference
        inference = EmotionInference(
            args.model_path,
            device=args.device
        )

        df = inference.process_conference_csv(args.csv_file, output_csv=args.output)

        # Print aggregated statistics
        aggregated = aggregate_conference_emotions(df)
        print("\nAggregated Conference Emotions:")
        for key, value in aggregated.items():
            print(f"  {key}: {value:.4f}")

    elif args.mode == 'batch':
        # Batch processing of all conferences
        process_all_conferences(
            args.audio_dir,
            args.model_path,
            args.output
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Emotion recognition inference')

    parser.add_argument('--mode', type=str, required=True,
                       choices=['single', 'conference', 'batch'],
                       help='Inference mode')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device for inference')

    # Single mode arguments
    parser.add_argument('--audio_file', type=str,
                       help='Path to audio file (for single mode)')

    # Conference mode arguments
    parser.add_argument('--csv_file', type=str,
                       help='Path to conference CSV file (for conference mode)')

    # Batch mode arguments
    parser.add_argument('--audio_dir', type=str,
                       help='Directory containing conference CSVs (for batch mode)')

    # Output arguments
    parser.add_argument('--output', type=str,
                       help='Output CSV path')

    args = parser.parse_args()

    main(args)
