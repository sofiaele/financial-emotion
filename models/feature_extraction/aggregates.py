import pandas as pd
import os
import numpy as np

def aggregate_sentiment_features_per_recording(csv):
    df = pd.read_csv(csv)
    emotion_columns = df.filter(regex=r'^emotion', axis=1).columns.tolist()
    emotion_means = df[emotion_columns].mean()
    valence_columns = df.filter(regex=r'^positivity', axis=1).columns.tolist()
    valence_means = df[valence_columns].mean()
    arousal_columns = df.filter(regex=r'^strength', axis=1).columns.tolist()
    arousal_means = df[arousal_columns].mean()
    result_array = emotion_means.to_list() + valence_means.to_list() + arousal_means.to_list()
    return result_array

def aggregate_independent_variables_per_recording(df, index):
    #import pdb; pdb.set_trace()
    result_array = [df.loc[index, 'LLP'], df.loc[index, 'ETA'], df.loc[index, 'EP'],
    df.loc[index, 'LTA'], df.loc[index, 'NII'], df.loc[index, 'ROA'],
    df.loc[index, 'ROE'], df.loc[index, 'SIZE'], df.loc[index, 'COST_EF'],
    df.loc[index, 'LEV'], df.loc[index, 'TOBINQ']]
    return result_array

def extract_features(config):
    feature_type = config['features']['type']
    ground_truths = config['ground_truths']['path']
    df_gt = pd.read_csv(ground_truths, delimiter=',')
    X = []
    y = []
    speakers = []
    if feature_type=='sentiment':
        for recording, gt, speaker in zip(df_gt["audio"], df_gt["RET"], df_gt["Speaker"]):
            recording_csv = os.path.join('../data/audio/', recording.split(".")[0] + ".csv")
            recording_features = aggregate_sentiment_features_per_recording(recording_csv)
            X.append(recording_features)
            y.append(gt)
            speakers.append(speaker)
    elif feature_type=='independent':
        for index, (recording, gt, speaker) in enumerate(zip(df_gt["audio"], df_gt["RET"], df_gt["Speaker"])):
            recording_features = aggregate_independent_variables_per_recording(df_gt, index)
            X.append(recording_features)
            y.append(gt)
            speakers.append(speaker)
    else:
        for index, (recording, gt, speaker) in enumerate(zip(df_gt["audio"], df_gt["RET"], df_gt["Speaker"])):
            recording_csv = os.path.join('../data/audio/', recording.split(".")[0] + ".csv")
            recording_sentiment_features = aggregate_sentiment_features_per_recording(recording_csv)
            recording_independent_features = aggregate_independent_variables_per_recording(df_gt, index)
            X.append(recording_sentiment_features + recording_independent_features)
            y.append(gt)
            speakers.append(speaker)
    nan_indices = np.isnan(X).any(axis=1)
    X = np.array(X)[~nan_indices]
    y = np.array(y)[~nan_indices]
    speakers = np.array(speakers)[~nan_indices]
    return X, y, speakers