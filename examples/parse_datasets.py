"""
Example script for parsing IEMOCAP and MSP-Podcast datasets
This script shows how to use the parsers to create combined datasets
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'models', 'emotion_recognition'))

from parsers import IEMOCAPParser, MSPPodcastParser, create_combined_dataset


def parse_iemocap_only(iemocap_path: str, output_csv: str = 'iemocap_dataset.csv'):
    """Parse only IEMOCAP dataset"""
    print("="*60)
    print("Parsing IEMOCAP Dataset")
    print("="*60)

    # Create parser
    parser = IEMOCAPParser(iemocap_path)

    # Parse with custom split configuration
    # Default: train=[1,2,4], val=[3], test=[5]
    split_config = {
        'train': [1, 2, 4],
        'val': [3],
        'test': [5]
    }

    df = parser.parse(split_config=split_config)

    # Display statistics
    print(f"\nTotal samples: {len(df)}")
    print(f"\nSplit distribution:")
    print(df['split'].value_counts())
    print(f"\nEmotion distribution:")
    print(df['emotion'].value_counts())
    print(f"\nValence distribution:")
    print(df['valence'].value_counts())
    print(f"\nArousal distribution:")
    print(df['arousal'].value_counts())

    # Save to CSV
    df.to_csv(output_csv, index=False)
    print(f"\nSaved to: {output_csv}")

    return df


def parse_msp_podcast_only(msp_path: str, msp_csv: str = None, output_csv: str = 'msp_podcast_dataset.csv'):
    """Parse only MSP-Podcast dataset"""
    print("="*60)
    print("Parsing MSP-Podcast Dataset")
    print("="*60)

    # Create parser
    # Option 1: Use existing msp.csv file
    # Option 2: Parse from labels_consensus.csv in msp_path
    parser = MSPPodcastParser(msp_path, csv_path=msp_csv)

    df = parser.parse()

    # Display statistics
    print(f"\nTotal samples: {len(df)}")
    print(f"\nSplit distribution:")
    print(df['split'].value_counts())
    print(f"\nEmotion distribution:")
    print(df['emotion'].value_counts())

    if not df['valence'].isna().all():
        print(f"\nValence distribution:")
        print(df['valence'].value_counts())
        print(f"\nArousal distribution:")
        print(df['arousal'].value_counts())

    # Save to CSV
    df.to_csv(output_csv, index=False)
    print(f"\nSaved to: {output_csv}")

    return df


def parse_combined_datasets(
    iemocap_path: str = None,
    msp_path: str = None,
    msp_csv: str = None,
    output_csv: str = 'combined_dataset.csv'
):
    """Parse and combine both IEMOCAP and MSP-Podcast datasets"""
    print("="*60)
    print("Parsing Combined Datasets (IEMOCAP + MSP-Podcast)")
    print("="*60)

    # Parse both datasets and combine
    train_df, val_df, test_df = create_combined_dataset(
        iemocap_path=iemocap_path,
        msp_path=msp_path,
        msp_csv_path=msp_csv,
        output_csv=output_csv
    )

    # Display detailed statistics per dataset
    combined_df = pd.concat([train_df, val_df, test_df])

    print("\n" + "="*60)
    print("Dataset Breakdown")
    print("="*60)

    if 'dataset' in combined_df.columns:
        print("\nSamples per dataset:")
        print(combined_df['dataset'].value_counts())

        print("\nEmotion distribution per dataset:")
        for dataset in combined_df['dataset'].unique():
            print(f"\n{dataset.upper()}:")
            dataset_df = combined_df[combined_df['dataset'] == dataset]
            print(dataset_df['emotion'].value_counts())

    return train_df, val_df, test_df


if __name__ == "__main__":
    import argparse
    import pandas as pd

    parser = argparse.ArgumentParser(
        description="Parse IEMOCAP and/or MSP-Podcast datasets for emotion recognition",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Parse only IEMOCAP
  python parse_datasets.py --iemocap_path /path/to/IEMOCAP --mode iemocap

  # Parse only MSP-Podcast using msp.csv
  python parse_datasets.py --msp_path /path/to/MSP-Podcast --msp_csv /path/to/msp.csv --mode msp

  # Parse only MSP-Podcast using labels_consensus.csv
  python parse_datasets.py --msp_path /path/to/MSP-Podcast --mode msp

  # Parse both datasets and combine
  python parse_datasets.py --iemocap_path /path/to/IEMOCAP --msp_path /path/to/MSP-Podcast --mode combined
        """
    )

    parser.add_argument('--iemocap_path', type=str,
                       help='Path to IEMOCAP root directory')
    parser.add_argument('--msp_path', type=str,
                       help='Path to MSP-Podcast root directory')
    parser.add_argument('--msp_csv', type=str,
                       help='Path to msp.csv file (optional, for MSP-Podcast)')
    parser.add_argument('--output_csv', type=str, default='combined_dataset.csv',
                       help='Path to output CSV file')
    parser.add_argument('--mode', type=str, choices=['iemocap', 'msp', 'combined'],
                       default='combined',
                       help='Parsing mode: iemocap, msp, or combined')

    args = parser.parse_args()

    # Validate inputs based on mode
    if args.mode == 'iemocap' and not args.iemocap_path:
        parser.error("--iemocap_path is required for 'iemocap' mode")

    if args.mode == 'msp' and not args.msp_path:
        parser.error("--msp_path is required for 'msp' mode")

    if args.mode == 'combined' and not (args.iemocap_path or args.msp_path):
        parser.error("At least one dataset path must be provided for 'combined' mode")

    # Execute parsing based on mode
    if args.mode == 'iemocap':
        parse_iemocap_only(args.iemocap_path, args.output_csv)

    elif args.mode == 'msp':
        parse_msp_podcast_only(args.msp_path, args.msp_csv, args.output_csv)

    elif args.mode == 'combined':
        parse_combined_datasets(
            iemocap_path=args.iemocap_path,
            msp_path=args.msp_path,
            msp_csv=args.msp_csv,
            output_csv=args.output_csv
        )
