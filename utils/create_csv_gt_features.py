import pandas as pd
import os
import argparse

def aggregate_sentiment_features_per_recording(csv):
    df = pd.read_csv(csv)
    emotion_columns = df.filter(regex=r'^emotion', axis=1).columns.tolist()
    emotion_means = df[emotion_columns].mean()
    valence_columns = df.filter(regex=r'^positivity', axis=1).columns.tolist()
    valence_means = df[valence_columns].mean()
    arousal_columns = df.filter(regex=r'^strength', axis=1).columns.tolist()
    arousal_means = df[arousal_columns].mean()
    return emotion_means, arousal_means, valence_means
def create_final_csv(ground_truths):
    df_gt = pd.read_csv(ground_truths, delimiter=';')
    print(df_gt["audio"])
    for index, row in df_gt.iterrows():
        recording = row["audio"]
        recording_csv = os.path.join('../data/audio/', recording.split(".")[0] + ".csv")
        emotion_means, arousal_means, valence_means = aggregate_sentiment_features_per_recording(recording_csv)
        for key, value in emotion_means.items():
            df_gt.at[index, key] = value
        for key, value in arousal_means.items():
            df_gt.at[index, key] = value
        for key, value in valence_means.items():
            df_gt.at[index, key] = value
    df_gt.to_csv("../ground_truths_sentiments.csv", index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', help='path to csv of ground truths', type=str, default='../data/non_intoductory_mean_audio.csv')
    args = parser.parse_args()
    create_final_csv(args.csv)