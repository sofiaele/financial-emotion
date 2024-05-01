import os
import argparse
import numpy as np
import pandas as pd
def dominant_speaker_assignment(csvs_path):
    #read all csvs from data/audio directory
    csvs = [os.path.join(csvs_path, csv) for csv in os.listdir(csvs_path) if csv.endswith(".csv")]
    #for loop csvs
    for file in csvs:
        #read each csv
        df = pd.read_csv(file)
        if "dominant" not in df.columns:
            #calculate the most frequent speaker in each file from speaker column
            print(df["speaker"].value_counts())
            dominant_speaker = df["speaker"].value_counts().idxmax()
            #add a new column to the csv with the "banker" label where dominant speaker is in "speaker" column
            df['dominant'] = np.where(df['speaker'] == dominant_speaker, 'banker', 'non-banker')
            #save the csv with the new column
            df.to_csv(file, index=False)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csvs', help='path to csv files', type=str, default='../data/audio/', required=False)
    args = parser.parse_args()

    dominant_speaker_assignment(args.csvs)



