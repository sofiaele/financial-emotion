"""
Preprocessing utilities for linking audio files with metadata.

This module provides functionality to link pre-downloaded audio files
with video metadata from CSV files.
"""

import pandas as pd
import argparse
import os
import sys

PROJECT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_PATH)


def create_csv(videos, returns):
    """
    Merge video metadata with financial returns data.

    Args:
        videos: Path to CSV file with video metadata
        returns: Path to CSV file with financial returns
    """
    # Load the CSV files
    df1 = pd.read_csv(returns, sep=';')
    df2 = pd.read_csv(videos, sep=';')

    # Convert the date columns to a common format (month-year)
    df1['MonthYear'] = pd.to_datetime(df1['DATE'], format='%d/%m/%y').dt.to_period('M')

    # Define the date formats
    format1 = '%Y-%m-%d'
    format2 = '%d/%m/%y'
    # Convert dates in the first format
    df2['Published_At_Converted1'] = pd.to_datetime(df2['Published_At'], format=format1, errors='coerce')

    # Convert dates in the second format
    df2['Published_At_Converted2'] = pd.to_datetime(df2['Published_At'], format=format2, errors='coerce')

    # Combine the results, preferring the non-NaT values from the first conversion
    df2['Published_At_Final'] = df2['Published_At_Converted1'].fillna(df2['Published_At_Converted2'])

    # Convert to monthly period
    df2['MonthYear'] = df2['Published_At_Final'].dt.to_period('M')

    # Drop intermediate columns
    df2 = df2.drop(columns=['Published_At_Converted1', 'Published_At_Converted2'])

    tickers = df1['TICKER'].unique()

    # Iterate over each TICKER and perform the merge
    for ticker in tickers:
        # Filter the DataFrames by the current TICKER
        df1_ticker = df1[df1['TICKER'] == ticker]

        # Perform the merge for the current TICKER
        merged_df = pd.merge(df1_ticker, df2, left_on='MonthYear', right_on='MonthYear',
                                          how='inner')
        # Add the 'audio' column to the merged DataFrame using title and suffix .wav
        merged_df['audio'] = merged_df['Title'] + '.wav'
        merged_df.to_csv(f'../data/final_per_ticker/{ticker}.csv', index=False)


def mean_per_month(csv_file):
    """
    Calculate mean financial metrics per month.

    Args:
        csv_file: Path to CSV file with financial data
    """
    df = pd.read_csv(csv_file)

    df = df.drop(["COMNAM", "TICKER", "PERMNO"], axis=1)
    # Identify the columns for grouping (excluding the column to mean)
    columns_to_convert = ['RET', 'LLP', 'ETA', 'EP', 'LTA', 'NII', 'ROA', 'ROE', 'SIZE', 'COST_EF', 'LEV', 'TOBINQ']
    groupby_columns = df.columns.difference(columns_to_convert)
    df[columns_to_convert] = df[columns_to_convert].replace(',', '.', regex=True)
    df[columns_to_convert] = df[columns_to_convert].apply(pd.to_numeric, errors='coerce')
    df = df.dropna(subset=['RET'])
    # Group by the selected columns and calculate the mean for the specified column
    result_df = df.groupby(list(groupby_columns)).mean().reset_index()
    # Save the dataframe to a new CSV file
    name_to_save = csv_file.split('/')[-1].split('.')[0] + '_mean.csv'
    result_df.to_csv(name_to_save, index=False)


def find_samples_missing(initial_csv, final_csv):
    """
    Find samples that are in final_csv but not in initial_csv.

    Args:
        initial_csv: Path to initial CSV file
        final_csv: Path to final CSV file
    """
    initial_df = pd.read_csv(initial_csv)
    final_df = pd.read_csv(final_csv)
    missing_samples = final_df[~final_df['Link'].isin(initial_df['Link'])]
    missing_samples.to_csv('missing_samples.csv', index=False)


def link_audio_files(metadata_file, audio_dir=None):
    """
    Link existing audio files to CSV metadata entries.

    This function matches pre-downloaded audio files in the audio directory
    with entries in the metadata CSV file based on title matching.

    Args:
        metadata_file: Path to CSV file containing video metadata with 'Title' column
        audio_dir: Directory containing audio WAV files (default: data/audio/)

    Returns:
        None. Creates an output CSV file with matched audio filenames.
    """
    if audio_dir is None:
        audio_dir = os.path.join(PROJECT_PATH, "data/audio/")

    # Load the CSV
    try:
        df = pd.read_csv(metadata_file, sep=';')
        if 'Link' not in df.columns:
            df = pd.read_csv(metadata_file)
    except:
        df = pd.read_csv(metadata_file)

    # Get list of existing audio files
    if not os.path.exists(audio_dir):
        print(f"Error: Audio directory does not exist: {audio_dir}")
        return

    audio_files = [f for f in os.listdir(audio_dir) if f.endswith('.wav')]
    print(f"Found {len(audio_files)} audio files in {audio_dir}")

    # Try to match each CSV entry to an audio file
    matched = 0
    not_matched = 0
    not_matched_titles = []

    for index, row in df.iterrows():
        title = row['Title']
        # Look for exact match
        expected_filename = f"{title}.wav"

        if expected_filename in audio_files:
            df.at[index, 'audio'] = expected_filename
            matched += 1
            print(f"✓ Matched: {title}")
        else:
            # Try fuzzy matching (in case of slight differences)
            found = False
            for audio_file in audio_files:
                # Check if title is contained in the filename
                if title.lower() in audio_file.lower():
                    df.at[index, 'audio'] = audio_file
                    matched += 1
                    print(f"✓ Fuzzy matched: {title} → {audio_file}")
                    found = True
                    break

            if not found:
                df.at[index, 'audio'] = ''
                not_matched += 1
                not_matched_titles.append(title)
                print(f"✗ Not found: {title}")

    # Print summary
    print(f"\n{'='*60}")
    print(f"Linking Summary:")
    print(f"  Matched: {matched}")
    print(f"  Not Matched: {not_matched}")
    print(f"  Total: {len(df)}")

    if not_matched_titles:
        print(f"\nNot matched titles:")
        for title in not_matched_titles:
            print(f"  - {title}")
    print(f"{'='*60}\n")

    # Save the DataFrame
    output_file = metadata_file.split('/')[-1].split('.')[0] + '_audio.csv'
    df.to_csv(output_file, index=False)
    print(f"Results saved to: {output_file}")


def merge_csvs(csv1, csv2):
    """
    Merge two CSV files on common columns.

    Args:
        csv1: Path to first CSV file
        csv2: Path to second CSV file
    """
    # Read the CSV files
    df1 = pd.read_csv(csv1)
    df2 = pd.read_csv(csv2, sep=';')
    common_columns = list(df1.columns.intersection(df2.columns))

    # Merge the dataframes on common columns
    merged_df = pd.merge(df1, df2, on=common_columns, how='outer')

    # Write the merged dataframe to a new CSV file
    merged_df.to_csv('merged.csv', index=False)


def merge_missing_audio_with_csv_mean():
    """Merge missing audio samples with mean CSV data."""
    # Read the CSV files
    df1 = pd.read_csv("missing_samples_audio.csv")
    df2 = pd.read_csv("non_introductory_mean.csv")

    # Merge the DataFrames, preserving all rows from df2
    merged_df = pd.merge(df2, df1[['DATE', 'audio']], on='DATE', how='left')

    # Fill missing values in the 'audio' column with empty strings
    merged_df['audio'].fillna('', inplace=True)

    merged_df.to_csv('merged_file.csv', index=False)


def copy_audio_info_for_existing_audios():
    """Copy audio information for existing audios from old data."""
    # Load the CSV files into pandas DataFrames
    df1 = pd.read_csv('../old_data/non_introductory_mean_audio.csv')
    df2 = pd.read_csv('merged_file.csv')

    # Iterate through each row in df1
    for index, row in df2.iterrows():
        link_value = row['Link']
        # Find matching row in df2 based on 'Link' value
        matching_row = df1[df1['Link'] == link_value]
        if not matching_row.empty:
            # Update 'audio' value in df1 with the value from matching row in df2
            df2.at[index, 'audio'] = matching_row['audio'].iloc[0]

    # Save the updated df1 to a new CSV file
    df2.to_csv('final.csv', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Preprocessing utilities for linking audio files with metadata'
    )
    parser.add_argument('--videos',
                        help='Path to CSV file with video metadata',
                        type=str, default='videos.csv', required=False)
    parser.add_argument('--returns',
                        help='Path to CSV file with financial returns',
                        type=str, default='returns.csv', required=False)
    parser.add_argument('--csv',
                        help='Path to CSV file for linking with audio files',
                        type=str, required=False)
    parser.add_argument('--audio-dir',
                        help='Directory containing audio WAV files (default: data/audio/)',
                        type=str, default=None, required=False)
    args = parser.parse_args()

    if args.csv:
        # Link audio files mode
        link_audio_files(args.csv, audio_dir=args.audio_dir)
    else:
        # Create merged CSV mode
        create_csv(args.videos, args.returns)
