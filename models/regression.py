import pandas as pd
import os
from sklearn.linear_model import LinearRegression
import numpy as np
import argparse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_predict, LeaveOneOut
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
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
def train(ground_truths):
    X = []
    y = []
    files = []
    speakers = []

    df_gt = pd.read_csv(ground_truths, delimiter=';')
    '''
    for recording, gt, speaker in zip(df_gt["audio"], df_gt["RET"], df_gt["Speaker"]):
        recording_csv = os.path.join('../data/audio/', recording.split(".")[0] + ".csv")
        recording_features = aggregate_sentiment_features_per_recording(recording_csv)
        X.append(recording_features)
        y.append(gt)
        files.append(recording)
        speakers.append(speaker)


    for index, (recording, gt, speaker) in enumerate(zip(df_gt["audio"], df_gt["RET"], df_gt["Speaker"])):
        recording_features = aggregate_independent_variables_per_recording(df_gt, index)
        X.append(recording_features)
        y.append(gt)
        files.append(recording)
        speakers.append(speaker)'''

    for index, (recording, gt, speaker) in enumerate(zip(df_gt["audio"], df_gt["RET"], df_gt["Speaker"])):
        recording_csv = os.path.join('../data/audio/', recording.split(".")[0] + ".csv")
        recording_sentiment_features = aggregate_sentiment_features_per_recording(recording_csv)
        recording_independent_features = aggregate_independent_variables_per_recording(df_gt, index)
        X.append(recording_sentiment_features + recording_independent_features)
        y.append(gt)
        files.append(recording)
        speakers.append(speaker)
    #X_train, X_test, y_train, y_test, file_names_train, file_names_test = train_test_split(X, y, files, test_size=0.2, random_state=42)
    '''scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    '''
    reg = LinearRegression()
    # Initialize LeaveOneOut Cross-Validation
    #loo = LeaveOneOut()
    clf = make_pipeline(StandardScaler(), reg)
    # Perform Leave-One-Out Cross-Validation
    predictions = cross_val_predict(clf, np.array(X), np.array(y), cv=len(X))

    mse =  mean_absolute_error(y, predictions)
    print("Mean Absolute Error:", mse)

    # Get unique speaker names
    unique_speakers = list(set(speakers))

    # Scatter plot with color coded points for each speaker
    for speaker in unique_speakers:
        indices = [i for i, s in enumerate(speakers) if s == speaker]
        plt.scatter(np.array(y)[indices], np.array(predictions)[indices], label=speaker)
    # Plot predictions and ground truths
    #plt.scatter(y, predictions, c=speakers, cmap='tab10')
    plt.xlabel('Ground Truth')
    plt.ylabel('Predictions')
    plt.title('Ground Truth vs. Predictions')
    plt.plot([np.array(y).min(), np.array(y).max()], [np.array(y).min(), np.array(y).max()], 'k--', lw=2)  # Plot the y=x line
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', help='path to csv of ground truths', type=str, default='../data/non_introductory_mean_audio.csv')
    args = parser.parse_args()
    train(args.csv)