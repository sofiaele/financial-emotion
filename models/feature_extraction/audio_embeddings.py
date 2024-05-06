import sys
sys.path.append("../../google-research")
from non_semantic_speech_benchmark.trillsson.models import get_keras_model
import pandas as pd
import os
import librosa
import numpy as np

def aggregate_embeddings_per_recording(csv, model):
    df = pd.read_csv(csv)
    embeddings = []
    for audio_path in df['audio_path']:
        full_path = os.path.join('../data/audio/', audio_path)
        audio_samples, sample_rate = librosa.load(full_path, sr=None)
        embeddings.append(model(audio_samples)['embedding'].numpy(()[0]))

    return np.mean(embeddings, axis=0)

def extract_trillsson_embeddings(config):
    # what frame_hop to use here?
    model = get_keras_model("efficientnetv2bS", frame_hop=195)
    model.load_weights("trillson_saved")

    ground_truths = config['ground_truths']['path']
    df_gt = pd.read_csv(ground_truths, delimiter=',')
    X = []
    y = []
    speakers = []
    for recording, gt, speaker in zip(df_gt["audio"], df_gt["RET"], df_gt["Speaker"]):
        recording_csv = os.path.join('../data/audio/', recording.split(".")[0] + ".csv")
        recording_features = aggregate_embeddings_per_recording(recording_csv, model)
        X.append(recording_features)
        y.append(gt)
        speakers.append(speaker)

    return X, y, speakers