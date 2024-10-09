import pandas as pd
import os
import numpy as np
import math
from sklearn.preprocessing import LabelEncoder


def aggregate_sentiment_features_per_recording(csv, feature_modality):
    result_array = []
    feature_names = []
    df = pd.read_csv(csv)
    if feature_modality == 'audio' or feature_modality == 'all':
        emotion_columns = df.filter(regex=r'^emotion', axis=1).columns.tolist()
        emotion_means = df[emotion_columns].mean()
        valence_columns = df.filter(regex=r'^positivity', axis=1).columns.tolist()
        valence_means = df[valence_columns].mean()
        arousal_columns = df.filter(regex=r'^strength', axis=1).columns.tolist()
        arousal_means = df[arousal_columns].mean()
        result_array = emotion_means.to_list() + valence_means.to_list() + arousal_means.to_list()
        feature_names = emotion_columns + valence_columns + arousal_columns
    if feature_modality == 'text' or feature_modality == 'all':
        # 2. One-hot encode the 'text_sentiment' column
        sentiment_dummies = pd.get_dummies(df['text_sentiment'], prefix='sentiment')

        # 3. Calculate the mean for the one-hot encoded columns
        sentiment_means = sentiment_dummies.mean()
        result_array = result_array + sentiment_means.to_list()
        feature_names = feature_names + sentiment_dummies.columns.tolist()
    return result_array, feature_names

def aggregate_independent_variables_per_recording(df, index):
    #import pdb; pdb.set_trace()
    result_array = [df.loc[index, 'LLP'], df.loc[index, 'ETA'], df.loc[index, 'EP'],
    df.loc[index, 'LTA'], df.loc[index, 'NII'], df.loc[index, 'ROA'],
    df.loc[index, 'ROE'], df.loc[index, 'SIZE'], df.loc[index, 'COST_EF'],
    df.loc[index, 'LEV'], df.loc[index, 'TOBINQ']]
    feature_names = ['LLP', 'ETA', 'EP', 'LTA', 'NII', 'ROA', 'ROE', 'SIZE', 'COST_EF', 'LEV', 'TOBINQ']
    return result_array, feature_names


def remove_nan_strings(string_list):
    # Identify indices of elements that are None or empty strings
    string_list = [s.replace(',', '.') if isinstance(s, str) else s for s in string_list]

    nan_indices = [i for i, s in enumerate(string_list) if s is None or s == 'nan' or (isinstance(s, str) and s.strip() == '')]

    # Handle the conversion and filtering for actual NaN values
    def is_nan(value):
        try:
            return np.isnan(float(value))
        except ValueError:
            return False

    nan_indices += [i for i, s in enumerate(string_list) if is_nan(s)]

    # Create a mask for the NumPy array
    mask = np.ones(len(string_list), dtype=bool)
    mask[nan_indices] = False

    # Remove elements from the list
    cleaned_list = [s for i, s in enumerate(string_list) if mask[i]]

    return cleaned_list, mask

def clean_and_convert_to_float(string_list):
    # Replace commas with dots
    string_list = [s.replace(',', '.') if s is not None else s for s in string_list]

    # Identify indices of invalid elements
    invalid_strings = {'', 'nan', '#DIV/0!'}
    invalid_indices = [i for i, s in enumerate(string_list) if s in invalid_strings]

    # Create a mask for the NumPy array
    mask = np.ones(len(string_list), dtype=bool)
    mask[invalid_indices] = False

    # Remove invalid elements and convert to float
    cleaned_list = [s for i, s in enumerate(string_list) if i not in invalid_indices]
    #cleaned_list = [float(s) for s in cleaned_list if s not in invalid_strings]

    return cleaned_list, mask

def extract_features(config):
    feature_type = config['features']['type']
    feature_modality = config['features']['modality']
    ground_truths = config['ground_truths']['path']
    df_gt = pd.read_csv(ground_truths, delimiter=',')

    # Define special strings to look for
    special_strings = ['#DIV/0!']
    # Identify indices of rows containing special strings
    mask = df_gt.apply(lambda x: x.isin(special_strings)).any(axis=1)
    # Remove rows containing special strings
    df_gt = df_gt[~mask]

    # Convert 'DATE' column to datetime format
    df_gt['DATE'] = pd.to_datetime(df_gt['DATE'], format='%d/%m/%y')
    # Sort the DataFrame by 'DATE' column
    df_gt = df_gt.sort_values(by='DATE')

    # Reset index if needed
    df_gt = df_gt.reset_index(drop=True)

    if 'TICKER' in df_gt.columns:
        #one_hot_encoded_tickers = pd.get_dummies(df_gt['TICKER'], prefix='TICKER')
        # Instantiate LabelEncoder
        label_encoder = LabelEncoder()

        # Fit and transform the 'TICKER' column to nominal labels (integer encoding)
        nominal_encoded_tickers = label_encoder.fit_transform(df_gt['TICKER'])

        # Create a new DataFrame with the nominal representation
        one_hot_encoded_tickers = pd.DataFrame(nominal_encoded_tickers, columns=['TICKER_nominal'])

    X = []
    y = []
    speakers = []
    tickers = []
    dates_list = []
    if feature_type=='sentiment':
        for recording, gt, speaker, one_hot, ticker, dates in zip(df_gt["audio"], df_gt["RET"], df_gt["Speaker"], one_hot_encoded_tickers.values, df_gt['TICKER'], df_gt['DATE']):
            recording_csv = os.path.join('../data/audio/', recording.split(".")[0] + ".csv")
            recording_features, feature_names = aggregate_sentiment_features_per_recording(recording_csv, feature_modality)
            #recording_features = np.char.replace(np.array(recording_features), ',', '.').astype(np.float64)
            # Create a DataFrame from recording_features if it's not already one
            if isinstance(recording_features, pd.DataFrame):
                recording_df = recording_features
            else:
                recording_df = pd.DataFrame([recording_features], columns=feature_names)

            # Create a DataFrame for the one-hot vector
            one_hot_df = pd.DataFrame([one_hot], columns=one_hot_encoded_tickers.columns)
            # Concatenate the recording features with the one-hot encoded ticker
            combined_features = pd.concat([recording_df, one_hot_df], axis=1)
            X.append(combined_features)
            y.append(gt)
            dates_list.append(dates)
            tickers.append(ticker)
            speakers.append(speaker)
        feature_names = feature_names + one_hot_encoded_tickers.columns.tolist()
    elif feature_type=='independent':
        for index, (recording, gt, speaker, one_hot, ticker, dates) in enumerate(zip(df_gt["audio"], df_gt["RET"], df_gt["Speaker"], one_hot_encoded_tickers.values, df_gt['TICKER'], df_gt['DATE'])):
            recording_features, feature_names = aggregate_independent_variables_per_recording(df_gt, index)
            recording_df = pd.DataFrame([recording_features], columns=feature_names)
            # Create a DataFrame for the one-hot vector
            one_hot_df = pd.DataFrame([one_hot], columns=one_hot_encoded_tickers.columns)

            # Concatenate the recording features with the one-hot encoded ticker
            combined_features = pd.concat([recording_df, one_hot_df], axis=1)

            X.append(combined_features)
            y.append(gt)
            dates_list.append(dates)
            tickers.append(ticker)
            speakers.append(speaker)
        feature_names = feature_names + one_hot_encoded_tickers.columns.tolist()
    else:
        for index, (recording, gt, speaker, one_hot, ticker, dates) in enumerate(zip(df_gt["audio"], df_gt["RET"], df_gt["Speaker"], one_hot_encoded_tickers.values, df_gt['TICKER'], df_gt['DATE'])):
            recording_csv = os.path.join('../data/audio/', recording.split(".")[0] + ".csv")
            recording_sentiment_features, sentiment_feature_names = aggregate_sentiment_features_per_recording(recording_csv, feature_modality)
            recording_sentiment_features = pd.DataFrame([recording_sentiment_features], columns=sentiment_feature_names)
            recording_independent_features, independent_feature_names = aggregate_independent_variables_per_recording(df_gt, index)
            recording_independent_features = pd.DataFrame([recording_independent_features], columns=independent_feature_names)
            # Create a DataFrame for the one-hot vector
            one_hot_df = pd.DataFrame([one_hot], columns=one_hot_encoded_tickers.columns)
            # Concatenate the recording features with the one-hot encoded ticker
            combined_features = pd.concat([recording_sentiment_features, recording_independent_features, one_hot_df], axis=1)
            #combined_features = pd.concat([recording_sentiment_features, recording_independent_features],
            #                              axis=1)
            X.append(combined_features)
            y.append(gt)
            dates_list.append(dates)
            tickers.append(ticker)
            speakers.append(speaker)
        feature_names = sentiment_feature_names + independent_feature_names + one_hot_encoded_tickers.columns.tolist()
        print("number of features:", len(feature_names))
        #feature_names = sentiment_feature_names + independent_feature_names
    print(len(y))
    # Find indices of lists in X that contain '#DIV/0!'
    indices_to_remove = [i for i, sublist in enumerate(X) if '#DIV/0!' in sublist]

    # Remove corresponding elements in Y
    y = [value for i, value in enumerate(y) if i not in indices_to_remove]

    # Remove corresponding lists in X
    X = [sublist for i, sublist in enumerate(X) if i not in indices_to_remove]
    # Remove corresponding elements in dates
    dates_list = [value for i, value in enumerate(dates_list) if i not in indices_to_remove]

    speakers = [sublist for i, sublist in enumerate(speakers) if i not in indices_to_remove]
    tickers = [sublist for i, sublist in enumerate(tickers) if i not in indices_to_remove]
    y, mask = remove_nan_strings(y)
    X = [df for df, keep in zip(X, mask) if keep]
    X_new = []
    for df in X:
        X_element = df.iloc[0].tolist()
        float64_list = [np.float64(str(item).replace(',', '.')) for item in X_element]
        X_new.append(float64_list)

    X= X_new
    tickers = [ticker for ticker, keep in zip(tickers, mask) if keep]
    dates_list = [date for date, keep in zip(dates_list, mask) if keep]

    y = np.char.replace(y, ',', '.').astype(np.float64)

    # -------------------- Concatenate the deltas per ticker to the original features --------------------
    # Get the group identifiers (last 10 features)
    X = np.array(X)
    if feature_type != 'independent':
        #groups = X[:, -1:]
        groups = tickers
        groups = np.array(groups).reshape(-1, 1)
        # Initialize an array to store deltas
        deltas = np.zeros_like(X[:, :10])

        # Identify unique groups
        unique_groups = np.unique(groups, axis=0)

        # Calculate deltas for each group
        for group in unique_groups:
            # Find indices of samples belonging to the current group
            group_indices = np.all(groups == group, axis=1)
            #group_indices = groups == group
            # Extract samples for this group
            group_samples = X[group_indices]
            group_dates = np.array(dates_list)[group_indices]
            # Calculate deltas for this group
            group_deltas = np.zeros_like(group_samples[:, :10])
            for i in range(1, len(group_samples)):
                group_deltas[i] = group_samples[i, :10] - group_samples[i - 1, :10]
            # Store deltas in the main deltas array
            deltas[group_indices] = group_deltas
        # Concatenate the deltas to the original features
        X = np.hstack((X, deltas))
        feature_names = feature_names + ['delta_' + name for name in feature_names[:10]]

    return X, y, speakers, feature_names, tickers, dates_list