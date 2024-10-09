from sklearn.linear_model import LinearRegression
from sklearn.dummy import DummyRegressor, DummyClassifier
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut, cross_val_predict, GridSearchCV, PredefinedSplit
from sklearn.pipeline import make_pipeline
from feature_extraction.aggregates import extract_features
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score
import yaml
from mlp import MLPRegressor, MLPClassifier
import torch.nn as nn
import torch.optim as optim
import torch
import sys
sys.path.insert(0,'../utils')
from plots import plot_predictions_gt, plot_auc, vizualize_posteriors_classification, vizualize_posteriors_regression, vizualize_embeddings_regression, vizualize_embeddings_classifier
from load_features import load_embeddings
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import mean_absolute_error
import pandas as pd
from utils import split_in_chronological_order

def train_classifier(X, y, feature_type, feature_names, tickers, dates):
    # Now, y_binary contains 1 for positive or zero values of y, and 0 for negative values of y
    y = np.where(np.array(y) >= 0.0, 1, 0)
    #vizualize_posteriors_classification(X, y, feature_names, feature_type)
    class_counts = np.bincount(y)
    print("Class Counts:", class_counts)


    param_grid = {
        'svc__C': [0.1, 1, 10],
        'svc__gamma': [0.001, 0.01, 0.1, 1]
    }


    '''param_grid = {
        'decisiontreeclassifier__max_depth': [None, 10, 20, 30, 40],
        'decisiontreeclassifier__min_samples_split': [2, 5, 10],
        'decisiontreeclassifier__min_samples_leaf': [1, 2, 4],
        'decisiontreeclassifier__max_features': [None, 'sqrt', 'log2'],
        'decisiontreeclassifier__criterion': ['gini', 'entropy'],
        'decisiontreeclassifier__splitter': ['best', 'random']
    }'''
    '''
    param_grid = {
        'bernoullinb__alpha': [0.1, 0.5, 1.0, 5.0, 10.0],
        'bernoullinb__binarize': [0.0, 0.5, 1.0]
    }'''

    # -------------- Splitting the data in predefined set --------------
    X_train, y_train, dates_train, \
        tickers_train, X_val, y_val, \
        dates_val, tickers_val, X_test, \
        y_test, dates_test, tickers_test = split_in_chronological_order(X, y, dates, tickers)
    class_counts = np.bincount(y_train)
    print("Train Class Counts:", class_counts)
    class_counts = np.bincount(y_val)
    print("Validation Class Counts:", class_counts)
    class_counts = np.bincount(y_test)
    print("Test Class Counts:", class_counts)
    # Define the splitting strategy
    # Combine training and validation data
    X_combined = np.vstack((X_train, X_val))
    y_combined = np.hstack((y_train, y_val))

    # Create an array indicating training and validation indices
    # -1 for training data, 0 for validation data
    validation_fold = np.array([-1] * len(X_train) + [0] * len(X_val))

    # Create PredefinedSplit
    ps = PredefinedSplit(test_fold=validation_fold)


    pipeline = make_pipeline(StandardScaler(), SVC(probability=True))
    grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=ps, n_jobs=-1, scoring='roc_auc')

    '''
    # Create a pipeline with StandardScaler and GaussianNB
    pipeline = make_pipeline(StandardScaler(), BernoulliNB())
    # Define the grid search
    grid_search = GridSearchCV(estimator=pipeline,
                               param_grid=param_grid,
                               cv=ps,
                               n_jobs=-1,
                               scoring='roc_auc')'''

    grid_search.fit(X_combined, y_combined)
    best_model = grid_search.best_estimator_
    best_model.fit(X_combined, y_combined)
    probas = best_model.predict_proba(X_test)
    predictions = probas[:, 1]

    # Calculate F1 score
    y_pred = (predictions > 0.5).astype(int)  # Convert probabilities to binary predictions
    f1 = f1_score(y_test, y_pred, average='macro')
    print("Macro f1:", f1)

    ## Baseline
    clf = make_pipeline(StandardScaler(), DummyClassifier(strategy='stratified', random_state=42))
    clf.fit(X_train, y_train)
    base_predictions = clf.predict_proba(X_test)[:, 1]
    # Compute the F1 score for each class
    base_y_pred = (base_predictions > 0.5).astype(int)
    base_f1 = f1_score(y_test, base_y_pred, average='macro')
    print("Baseline Macro f1:", base_f1)
    fpr, tpr, thresholds = roc_curve(y_test, predictions)
    roc_auc = auc(fpr, tpr)
    b_fpr, b_tpr, b_thresholds = roc_curve(y_test, base_predictions)
    base_roc_auc = auc(b_fpr, b_tpr)
    print("ROC AUC:", roc_auc, "Baseline ROC AUC:", base_roc_auc, "F1:", f1, "Baseline F1:", base_f1)

    # Convert to DataFrame for easy grouping
    df = pd.DataFrame({
        'y': y_test,
        'y_pred': y_pred,
        'base_y_pred': base_y_pred,
        'ticker': tickers_test,
        'predictions': predictions,
        'base_predictions': base_predictions
    })
    '''
    # -------------- Using loo --------------
    # Outer loop: Leave-One-Out cross-validation
    loo = LeaveOneOut()
    predictions = np.zeros(len(X))
    for train_index, test_index in tqdm(loo.split(X), total=loo.get_n_splits(X), desc="Processing splits"):
        X_train = [X[i] for i in train_index]
        X_test = [X[i] for i in test_index]
        X_train = np.array(X_train)
        X_test = np.array(X_test)

        # Zero out the deltas for the next sample of the test
        X_train_final = X_train.copy()
        if test_index[0] < len(X) - 1:
            X_train_final[test_index[0], -10:] = np.zeros(10)

        y_train, y_test = y[train_index], y[test_index]

        # -------------- Grid Search --------------

        # Define the pipeline
        pipeline = make_pipeline(StandardScaler(), SVC(probability=True))

        # Inner loop: Hyperparameter tuning using GridSearchCV
        grid_search = GridSearchCV(pipeline, param_grid, cv=5,
                                   scoring='roc_auc')  # Use 5-fold cross-validation for inner loop
        grid_search.fit(X_train_final, y_train)

        # -------------- End of Grid Search --------------

        # Train the best model on the entire training set
        best_model = grid_search.best_estimator_
        best_model.fit(X_train_final, y_train)

        # Predict probabilities on the test set
        probas = best_model.predict_proba(X_test)
        predictions[test_index] = probas[:, 1]
    y_pred = (predictions > 0.5).astype(int)  # Convert probabilities to binary predictions
    f1 = f1_score(y, y_pred, average='macro')
    
    ## Baseline
    clf = make_pipeline(StandardScaler(), DummyClassifier(strategy='stratified', random_state=42))
    base_predictions = cross_val_predict(clf, np.array(X), np.array(y), cv=loo, method='predict_proba')[:, 1]
    # Compute the F1 score for each class
    base_y_pred = (base_predictions > 0.5).astype(int)
    base_f1 = f1_score(y, base_y_pred, average='macro')
    print("Baseline Macro f1:", base_f1)
    fpr, tpr, thresholds = roc_curve(y, predictions)
    roc_auc = auc(fpr, tpr)
    b_fpr, b_tpr, b_thresholds = roc_curve(y, base_predictions)
    base_roc_auc = auc(b_fpr, b_tpr)
    print("ROC AUC:", roc_auc, "Baseline ROC AUC:", base_roc_auc, "F1:", f1, "Baseline F1:", base_f1)
    # Convert to DataFrame for easy grouping
    df = pd.DataFrame({
        'y': y,
        'y_pred': y_pred,
        'base_y_pred': base_y_pred,
        'ticker': tickers,
        'predictions': predictions,
        'base_predictions': base_predictions
    })
    '''




    #return roc_auc, base_roc_auc, f1, base_f1
    #plot_auc(y, predictions, base_predictions, feature_type, f1, base_f1)



    # Get unique tickers
    unique_tickers = df['ticker'].unique()
    results = {}
    # Calculate metrics per ticker
    for ticker in unique_tickers:
        # Filter data for the current ticker
        ticker_df = df[df['ticker'] == ticker]

        # True and predicted values for the current ticker
        y_true = ticker_df['y']
        y_pred_ticker = ticker_df['y_pred']
        base_y_pred_ticker = ticker_df['base_y_pred']
        predictions_ticker = ticker_df['predictions']
        base_predictions_ticker = ticker_df['base_predictions']

        # Calculate F1 score
        f1 = f1_score(y_true, y_pred_ticker, average='macro')
        base_f1 = f1_score(y_true, base_y_pred_ticker, average='macro')

        fpr, tpr, thresholds = roc_curve(y_true, predictions_ticker)
        roc_auc = auc(fpr, tpr)
        b_fpr, b_tpr, b_thresholds = roc_curve(y_true, base_predictions_ticker)
        base_roc_auc = auc(b_fpr, b_tpr)

        results[ticker] = {'roc_auc': roc_auc, 'base_roc_auc': base_roc_auc, 'f1': f1, 'base_f1': base_f1}
    print(results)

def train_regressor(X, y, speakers, feature_type, feature_names):
    #vizualize_posteriors_regression(X, y, feature_names,  feature_type)
    reg = LinearRegression()
    clf = make_pipeline(StandardScaler(), reg)
    # Perform Leave-One-Out Cross-Validation
    predictions = cross_val_predict(clf, np.array(X), np.array(y), cv=len(X))

    ## Baseline
    baseline = DummyRegressor()
    clf = make_pipeline(StandardScaler(), baseline)
    # Perform Leave-One-Out Cross-Validation
    base_predictions = cross_val_predict(clf, np.array(X), np.array(y), cv=len(X))

    mae = mean_absolute_error(y, predictions)

    base_mae = mean_absolute_error(y, base_predictions)

    #plot_predictions_gt(y, predictions, base_predictions, speakers, feature_type)
    return mae, base_mae

def train_mlp_regressor(X, y, speakers, feature_type):
    X = np.array(X)  # Assuming X is your list of embeddings
    y = np.array(y)  # Assuming y is your list of labels
    #vizualize_embeddings_regression(X, y, feature_type)
    loo = LeaveOneOut()
    input_size = X.shape[1]
    all_losses = []
    predictions = []
    for train_index, test_index in loo.split(X):
        # Split the data
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Convert to PyTorch tensors
        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
        X_test = torch.tensor(X_test, dtype=torch.float32)
        y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

        # Define hyperparameters
        learning_rate = 0.001
        num_epochs = 100

        # Initialize the model, loss function, and optimizer
        model = MLPRegressor(input_size, 1)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # Training loop
        for epoch in range(num_epochs):
            # Forward pass
            model.train()
            outputs = model(X_train)
            loss = criterion(outputs, y_train)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Print progress
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

        # Evaluate the model on the test set
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test)
            predictions.append(test_outputs.item())
            test_loss = criterion(test_outputs, y_test).item()
            all_losses.append(test_loss)
            print(f'Test Loss for the left-out sample: {test_loss:.4f}')
    baseline = DummyRegressor()
    clf = make_pipeline(StandardScaler(), baseline)
    # Perform Leave-One-Out Cross-Validation
    base_predictions = cross_val_predict(clf, np.array(X), np.array(y), cv=len(X))
    plot_predictions_gt(y, predictions, base_predictions, speakers, feature_type)
    # Calculate the average test loss
    avg_test_loss = np.mean(all_losses)
    print(f'Average Test Loss: {avg_test_loss:.4f}')

    return avg_test_loss

def train_mlp_classifier(X, y, speakers, feature_type):
    X = np.array(X)  # Assuming X is your list of embeddings
    y = np.where(np.array(y) >= 0.03, 1, 0)
    #vizualize_embeddings_classifier(X, y, feature_type)

    loo = LeaveOneOut()
    input_size = X.shape[1]
    all_losses = []
    predictions = []
    for train_index, test_index in loo.split(X):
        # Split the data
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Convert to PyTorch tensors
        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
        X_test = torch.tensor(X_test, dtype=torch.float32)
        y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

        # Define hyperparameters
        learning_rate = 0.001
        num_epochs = 100

        # Initialize the model, loss function, and optimizer
        model = MLPClassifier(input_size)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # Training loop
        for epoch in range(num_epochs):
            # Forward pass
            model.train()
            outputs = model(X_train)
            loss = criterion(outputs, y_train)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Print progress
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

        # Evaluate the model on the test set
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test)
            predictions.append(test_outputs.item())
            test_loss = criterion(test_outputs, y_test).item()
            all_losses.append(test_loss)
            print(f'Test Loss for the left-out sample: {test_loss:.4f}')

    clf = make_pipeline(StandardScaler(), DummyClassifier(strategy='stratified', random_state=42))
    base_predictions = cross_val_predict(clf, np.array(X), np.array(y), cv=loo, method='predict_proba')[:, 1]
    # Compute the F1 score for each class
    base_y_pred = (base_predictions > 0.5).astype(int)
    base_f1 = f1_score(y, base_y_pred, average='macro')
    print("Baseline Macro f1:", base_f1)
    y_pred = [1 if pred > 0.5 else 0 for pred in predictions]  # Convert probabilities to binary predictions
    f1 = f1_score(y, y_pred, average='macro')
    plot_auc(y, predictions, base_predictions, feature_type, f1, base_f1)
    # Calculate the average test loss
    avg_test_loss = np.mean(all_losses)
    print(f'Average Test Loss: {avg_test_loss:.4f}')

    return avg_test_loss

if __name__ == '__main__':
    # Load configuration from YAML file
    with open("config.yml", 'r') as config_file:
        config = yaml.safe_load(config_file)
    if config['ground_truths']['type'] == 'aggregates':
        if 'embeddings' in config['features']['type']:
            X, y, speakers = load_embeddings(config)
            if config['model_type'] == 'regressor':
                train_mlp_regressor(X, y, speakers, config['features']['type'])
            elif config['model_type'] == 'classifier':
                train_mlp_classifier(X, y, speakers, config['features']['type'])
        else:
            X, y, speakers, feature_names, tickers, dates = extract_features(config)
            if config['model_type'] == 'regressor':
                train_regressor(X, y, speakers, config['features']['type'], feature_names)
            elif config['model_type'] == 'classifier':
                train_classifier(X, y, config['features']['type'], feature_names, tickers, dates)