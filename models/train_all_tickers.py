from feature_extraction.aggregates import extract_features
import yaml
import sys
sys.path.insert(0,'../utils')
from load_features import load_embeddings
from train import train_classifier, train_regressor, train_mlp_classifier, train_mlp_regressor
import os
import pandas as pd
if __name__ == '__main__':
    # list the csvs in data/final_per_ticker path
    csvs = os.listdir('../data/final_per_ticker')
    results = {}
    for csv in csvs:
        #open csv and check how many rows has
        df = pd.read_csv(os.path.join('../data/final_per_ticker', csv))
        if len(df) < 68:
            continue
        with open("config.yml", 'r') as config_file:
            config = yaml.safe_load(config_file)
        config['ground_truths']['path'] = os.path.join('../data/final_per_ticker', csv)
        X, y, speakers, feature_names = extract_features(config)
        if config['model_type'] == 'regressor':
            mae, base_mae = train_regressor(X, y, speakers, config['features']['type'], feature_names)
            results[csv.split('.')[0]] = {'mae': mae, 'base_mae': base_mae}
        elif config['model_type'] == 'classifier':
            roc_auc, base_roc_auc, f1, base_f1 = train_classifier(X, y, config['features']['type'], feature_names)
            results[csv.split('.')[0]] = {'roc_auc': roc_auc, 'base_roc_auc': base_roc_auc, 'f1': f1, 'base_f1': base_f1}
    print(results)