import sys
sys.path.append("google-research")
from non_semantic_speech_benchmark.trillsson.models import get_keras_model
import pandas as pd
import os
import librosa
import numpy as np
import argparse
import yaml 
from tqdm import tqdm

def aggregate_embeddings_per_recording(csv, model, cache):
    df = pd.read_csv(csv)
    embeddings = []
    
    for audio_path in tqdm(df['audio_path'], desc=f"Extracting embeddings for {csv}"):
        full_path = os.path.join('../../data/audio/', audio_path)
        audio_samples, _ = librosa.load(full_path, sr=None)
        audio_samples = np.expand_dims(audio_samples, axis=0)
        embeddings.append(model(audio_samples)['embedding'].numpy()[0])
    print("Aggregating embeddings of ", csv)
    aggregated_embeddings = np.mean(embeddings, axis=0)
    embedding_file_name = csv.split("/")[-1].split(".")[0] + ".npy"
    embedding_filename = os.path.join(cache, embedding_file_name)
    np.save(embedding_filename, aggregated_embeddings)
    return aggregated_embeddings


def extract_trillsson_embeddings(config):
    # what frame_hop to use here?
    model = get_keras_model("efficientnetv2bS", frame_hop=195)
    model.load_weights("trillson_saved")

    ground_truths = config['ground_truths']['path']
    cache = config["audio_embeddings"]
    # Ensure the embeddings directory exists
    if not os.path.exists(cache):
        os.makedirs(cache)
    df_gt = pd.read_csv(ground_truths, delimiter=',')
    X = []
    y = []
    speakers = []
    for recording, gt, speaker in zip(df_gt["audio"], df_gt["RET"], df_gt["Speaker"]):
        recording_csv = os.path.join('../../data/audio/', recording.split(".")[0] + ".csv")
        embedding_filename = os.path.join(cache, recording_csv.split("/")[-1].split(".")[0] + ".npy")
        if not os.path.exists(embedding_filename):
            recording_features = aggregate_embeddings_per_recording(recording_csv, model, cache)
        else:
            recording_features = np.load(embedding_filename)
        X.append(recording_features)
        y.append(gt)
        speakers.append(speaker)

    return X, y, speakers

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--config', type=str, default='../config.yml')
    args = argparser.parse_args()
    with open(args.config, 'r') as config_file:
        config = yaml.safe_load(config_file)
    X, y, speakers = extract_trillsson_embeddings(config)
    