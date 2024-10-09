import pandas as pd
import os
import numpy as np


def load_embeddings(config):
    ground_truths = config['ground_truths']['path']
    features_type = config['features']['type']
    cache = config[features_type]['out_folder']
    df_gt = pd.read_csv(ground_truths, delimiter=',')
    X = []
    y = []
    speakers = []
    for recording, gt, speaker in zip(df_gt["audio"], df_gt["RET"], df_gt["Speaker"]):
        recording_csv = os.path.join('../../data/audio/', recording.split(".")[0] + ".csv")
        embedding_filename = os.path.join(cache, recording_csv.split("/")[-1].split(".")[0] + ".npy")
        recording_features = np.load(embedding_filename)
        X.append(recording_features)
        y.append(gt)
        speakers.append(speaker)
    return X, y, speakers