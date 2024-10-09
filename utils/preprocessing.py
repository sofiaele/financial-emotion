import pandas as pd
import argparse
import os
import yt_dlp as youtube_dl
import sys
PROJECT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_PATH)
def create_csv(videos, returns):
    # Load the CSV files
    df1 = pd.read_csv(returns, sep=';')
    df2 = pd.read_csv(videos, sep=';')

    # Convert the date columns to a common format (month-year)
    df1['MonthYear'] = pd.to_datetime(df1['DATE'], format='%d/%m/%y').dt.to_period('M')
    #df2['MonthYear'] = pd.to_datetime(df2['Published_At'], format='%d/%m/%y').dt.to_period('M')

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

    '''# Second merge considering 'Introductory' in 'Title'
    df_introductory = df2[df2['Title'].str.contains('Introductory')]
    merged_df_introductory = pd.merge(df1, df_introductory, left_on='MonthYear', right_on='MonthYear', how='inner')
    # Save the merged dataframe to a new CSV file
    merged_df_introductory.to_csv('introductory.csv', index=False)
    '''

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

    '''# Second merge considering rows where 'Title' does not contain 'Introductory'
    df_non_introductory = df2[~df2['Title'].str.contains('Introductory')]

    merged_df_non_introductory = pd.merge(df1, df2, left_on='MonthYear', right_on='MonthYear',
                                          how='inner')
    # Save the merged dataframe to a new CSV file
    merged_df_non_introductory.to_csv('non_introductory.csv', index=False)'''


def mean_per_month(csv_file):
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
    initial_df = pd.read_csv(initial_csv)
    final_df = pd.read_csv(final_csv)
    missing_samples = final_df[~final_df['Link'].isin(initial_df['Link'])]
    missing_samples.to_csv('missing_samples.csv', index=False)

def download_audio(link):

    options = {
        'format': 'bestaudio/best',
        'extractaudio': True,
        'audioformat': 'wav',
        'outtmpl': os.path.join(PROJECT_PATH, "data/audio/", '%(title)s.%(ext)s'),
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
        'postprocessor_args': [
            '-ac', '1',  # Set to mono channel
            '-ar', '16000'  # Set to 16kHz sample rate
        ],
        'prefer_ffmpeg': True,
    }
    with youtube_dl.YoutubeDL(options) as ydl:
        info_dict = ydl.extract_info(link, download=True)
    return os.path.join(f"{info_dict['title']}.wav")


def download_audios_from_csv(metadata_file):
    # Load your DataFrame with the 'Link' column
    df = pd.read_csv(metadata_file)
    # Apply the function to each row in the DataFrame and create a new column
    for index, row in df.iterrows():
        df.at[index, 'audio'] = download_audio(row['Link'])
        print(f"Downloaded audio for {row['Link']}")

    # Save the DataFrame with the local paths
    df.to_csv(metadata_file.split('/')[-1].split('.')[0] + '_audio.csv', index=False)


def merge_csvs(csv1, csv2):
    # Read the CSV files
    df1 = pd.read_csv(csv1)
    df2 = pd.read_csv(csv2, sep=';')
    common_columns = list(df1.columns.intersection(df2.columns))

    # Merge the dataframes on common columns
    merged_df = pd.merge(df1, df2, on=common_columns,  how='outer')

    # Write the merged dataframe to a new CSV file
    merged_df.to_csv('merged.csv', index=False)

def merge_missing_audio_with_csv_mean():
    # Read the CSV files
    df1 = pd.read_csv("missing_samples_audio.csv")
    df2 = pd.read_csv("non_introductory_mean.csv")

    # Merge the DataFrames, preserving all rows from df2
    merged_df = pd.merge(df2, df1[['DATE', 'audio']], on='DATE', how='left')

    # Fill missing values in the 'audio' column with empty strings
    merged_df['audio'].fillna('', inplace=True)

    merged_df.to_csv('merged_file.csv', index=False)
def copy_audio_info_for_existing_audios():
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--videos', help='csv of video metadata', type=str, default='videos.csv', required=False)
    parser.add_argument('--returns', help='csv of financial metadata', type=str, default='returns.csv', required=False)
    args = parser.parse_args()
    create_csv(args.videos, args.returns)
