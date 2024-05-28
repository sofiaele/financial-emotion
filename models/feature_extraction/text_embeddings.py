import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import yaml
from sentence_transformers import SentenceTransformer

def aggregate_embeddings_per_recording(csv, model, tokenizer, cache, model_name):
    df = pd.read_csv(csv)
    embeddings = []

    for sentence in tqdm(df['text'], desc=f"Extracting embeddings for {csv}"):
        sentence = [sentence]

        if model_name=="llama":
            t_input = tokenizer(sentence, padding=True, return_tensors="pt")
            with torch.no_grad():
                last_hidden_state = model(**t_input, output_hidden_states=True).hidden_states[-1]
            weights_for_non_padding = t_input.attention_mask * torch.arange(start=1,
                                                                            end=last_hidden_state.shape[
                                                                                    1] + 1).unsqueeze(0)

            sum_embeddings = torch.sum(last_hidden_state * weights_for_non_padding.unsqueeze(-1), dim=1)
            num_of_none_padding_tokens = torch.sum(weights_for_non_padding, dim=-1).unsqueeze(-1)
            sentence_embeddings = sum_embeddings / num_of_none_padding_tokens
        else:
            sentence_embeddings = model.encode(sentence)

        embeddings.append(sentence_embeddings.numpy()[0])
    print("Aggregating embeddings of ", csv)
    aggregated_embeddings = np.mean(embeddings, axis=0)
    embedding_file_name = csv.split("/")[-1].split(".")[0] + ".npy"
    embedding_filename = os.path.join(cache, embedding_file_name)
    np.save(embedding_filename, aggregated_embeddings)
    return aggregated_embeddings

def extract_llama_embeddings(config):
    model_name = config["text_embeddings"]["model"]
    if model_name == 'llama':
        token = "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"
        model_id = "meta-llama/Llama-2-7b-chat-hf"
        tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto", device_map="auto", token=token)
    else:
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    model.eval()

    ground_truths = config['ground_truths']['path']
    cache = config["text_embeddings"]["out_folder"]
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
            recording_features = aggregate_embeddings_per_recording(recording_csv, model, tokenizer, cache, model_name)
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
    X, y, speakers = extract_llama_embeddings(config)