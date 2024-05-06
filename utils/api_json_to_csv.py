import os
import pandas as pd
import json
import argparse
import numpy as np
def api_json_to_csv(json_file):
    with open(json_file, 'r') as json_file:
        data = json.load(json_file)
    csvs = [os.path.join('../data/new_audio_2', csv) for csv in os.listdir('../data/new_audio_2') if csv.endswith(".csv")]
    # for loop csvs
    for file in csvs:
        # read each csv
        df = pd.read_csv(file)
        for utterance_path in df['audio_path']:
            utterance = utterance_path.split("/")[-1]
            matching_audio_paths = [audio_path for audio_path in data.keys() if audio_path.endswith(utterance)]
            for task_dict in data[matching_audio_paths[0]]:
                task = task_dict["task"]
                if task != 'features':
                    for class_prediction in task_dict["prediction"]:
                        label = class_prediction["label"]
                        posterior = class_prediction["posterior"]
                        column_name = task + "_" + label
                        if column_name not in df.columns:
                            # Add the new column if it does not exist
                            df[column_name] = None
                        print(posterior)
                        df.at[df[df['audio_path'] == utterance_path].index[0], column_name] = float(posterior)
        df.to_csv(file, index=False, float_format='%.4f')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--json', help='path to json api results', type=str, default='../data/new_audio_2/utterances/results/final_results.json')
    args = parser.parse_args()
    api_json_to_csv(args.json)