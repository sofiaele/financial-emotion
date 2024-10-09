import pandas as pd
from skllm.config import SKLLMConfig
import argparse
import os
import pickle
def inference(path_csvs):
    SKLLMConfig.set_openai_key("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")
    with open('financial_sentiment_classifier.pkl', 'rb') as f:
        clf = pickle.load(f)
    csvs = [os.path.join('../../data/audio', csv) for csv in os.listdir(path_csvs) if
            csv.endswith(".csv")]
    # for loop csvs
    for file in csvs:
        # read each csv
        df = pd.read_csv(file)
        if 'text_sentiment' in df.columns:
            continue
        sentences = df['text']
        y = clf.predict(sentences)
        # add y to df:
        df['text_sentiment'] = y

        # save df to csv:df
        df.to_csv(file, index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csvs', help='path to csvs', type=str, default='../../data/audio')
    args = parser.parse_args()
    inference(args.csvs)
