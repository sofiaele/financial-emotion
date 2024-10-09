import sys
sys.path.insert(0,'../utils')
import os
import pandas as pd
import random
if __name__ == '__main__':
    # list the csvs in data/final_per_ticker path
    csvs = os.listdir('../data/final_per_ticker')
    # Shuffle the list to ensure randomness

    filtered_csvs = []
    for csv in csvs:
        #open csv and check how many rows has
        df = pd.read_csv(os.path.join('../data/final_per_ticker', csv))
        if df.shape[0] < 68:
            continue
        filtered_csvs.append(csv)
    random.shuffle(filtered_csvs)
    sublists = [filtered_csvs[i:i + 10] for i in range(0, len(filtered_csvs), 10)]
    for sublist in sublists:
        dataframes = []
        for csv in sublist:
            df = pd.read_csv(os.path.join('../data/final_per_ticker', csv))
            print(df.shape[0])
            dataframes.append(df)
        concatenated_df = pd.concat(dataframes, ignore_index=True)
        output_file = f'../data/batch_20_tickers/concatenated_{sublist[0].split(".")[0]}.csv'
        concatenated_df.to_csv(output_file, index=False)