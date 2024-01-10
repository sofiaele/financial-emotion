import pandas as pd
import argparse
import os
import yt_dlp as youtube_dl
import sys
PROJECT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_PATH)
def create_csv(videos, returns):
    # Load the CSV files
    df1 = pd.read_csv(returns)
    df2 = pd.read_csv(videos)

    # Convert the date columns to a common format (month-year)
    df1['MonthYear'] = pd.to_datetime(df1['Names Date'], format='%d/%m/%Y').dt.to_period('M')
    df2['MonthYear'] = pd.to_datetime(df2['Published At']).dt.to_period('M')

    # Second merge considering 'Introductory' in 'Title'
    df_introductory = df2[df2['Title'].str.contains('Introductory')]
    merged_df_introductory = pd.merge(df1, df_introductory, left_on='MonthYear', right_on='MonthYear', how='inner')
    # Save the merged dataframe to a new CSV file
    merged_df_introductory.to_csv('introductory.csv', index=False)

    # Second merge considering rows where 'Title' does not contain 'Introductory'
    df_non_introductory = df2[~df2['Title'].str.contains('Introductory')]
    merged_df_non_introductory = pd.merge(df1, df_non_introductory, left_on='MonthYear', right_on='MonthYear',
                                          how='inner')
    # Save the merged dataframe to a new CSV file
    merged_df_non_introductory.to_csv('non_introductory.csv', index=False)


def mean_per_month(csv_file):
    df = pd.read_csv(csv_file)

    #drop the columns 'Ticker Symbol' and 'Company Name'
    df = df.drop("Company Name", axis=1)
    df = df.drop("Ticker Symbol", axis=1)
    # Identify the columns for grouping (excluding the column to mean)
    groupby_columns = df.columns.difference(['Returns'])
    # Group by the selected columns and calculate the mean for the specified column
    result_df = df.groupby(list(groupby_columns)).mean().reset_index()
    # Save the dataframe to a new CSV file
    name_to_save = csv_file.split('/')[-1].split('.')[0] + '_mean.csv'
    result_df.to_csv(name_to_save, index=False)
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
    #df['audio'] = df['Link'].apply(download_audio)
    for index, row in df.iterrows():
        df.at[index, 'audio'] = download_audio(row['Link'])
        print(f"Downloaded audio for {row['Link']}")

    # Save the DataFrame with the local paths
    df.to_csv(metadata_file.split('/')[-1].split('.')[0] + '_audio.csv', index=False)

download_audios_from_csv('../data/non_introductory_mean.csv')
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--videos', help='csv of video metadata', type=str, default='videos.csv')
    parser.add_argument('--returns', help='csv of financial metadata', type=str, default='returns.csv')
    args = parser.parse_args()
    create_csv(args.videos, args.returns)