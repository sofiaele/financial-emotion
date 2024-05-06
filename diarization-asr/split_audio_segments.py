import pandas as pd
from pydub import AudioSegment
import argparse
import os

def split_wav(input_wav, output_folder, segments_df):
    # Load the input WAV file
    audio = AudioSegment.from_wav(input_wav)

    # Iterate through each row in the CSV file
    for index, row in segments_df.iterrows():
        start_time = int(row['start'] * 1000)  # Convert seconds to milliseconds
        end_time = int(row['end'] * 1000)  # Convert seconds to milliseconds

        # Extract the segment based on start and end times
        segment = audio[start_time:end_time]
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        output_filename = input_wav.split("/")[-1].split(".")[0] + "_segment_" + str(index + 1) + ".wav"
        # Define the output filename for the segment
        output_filename = f"{output_folder}/{output_filename}"

        # Export the segment as a new WAV file
        segment.export(output_filename, format="wav")
        print(f"Segment {index + 1} exported to {output_filename}")

        # Update CSV with the filename of the segmented WAV
        segments_df.at[index, 'audio_path'] = f"utterances/{output_filename.split('/')[-1]}"
    return segments_df
def make_utterances():
    csvs = [os.path.join('../data/new_audio_2/', csv) for csv in os.listdir('../data/new_audio_2/') if csv.endswith(".csv")]
    # for loop csvs
    for file in csvs:
        # Load the CSV file with start and end times
        segments_df = pd.read_csv(file)
        # Iterate through each row in the CSV file
        for index, row in segments_df.iterrows():
            if row["text"].isupper():
                start_time = row['start']
                segments_df.drop(index=index, inplace=True)
                next_row_index = index + 1
                segments_df.at[next_row_index, 'start'] = start_time
        segments_df = split_wav(".".join(file.split(".")[:-1]) + ".wav", "../data/new_audio_2/utterances", segments_df)
        segments_df.to_csv(file, index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    make_utterances()
