from sklearn.linear_model import LinearRegression
from sklearn.dummy import DummyRegressor, DummyClassifier
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut, cross_val_predict, GridSearchCV
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from feature_extraction.aggregates import extract_features
#from feature_extraction.audio_embeddings import extract_trillsson_embeddings
from sklearn.svm import SVC
from sklearn.metrics import f1_score
import yaml
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.linear_model import LogisticRegression
from mlp import MLP
import torch.nn as nn
import torch.optim as optim
import torch

def train_classifier(X, y, feature_type):
    # Now, y_binary contains 1 for positive or zero values of y, and 0 for negative values of y
    y = np.where(np.array(y) >= 0.03, 1, 0)

    class_counts = np.bincount(y)
    print("Class Counts:", class_counts)

    param_grid = {
        'svc__C': [0.1, 1, 10],
        'svc__gamma': [0.001, 0.01, 0.1, 1]
    }

    # Outer loop: Leave-One-Out cross-validation
    loo = LeaveOneOut()
    predictions = np.zeros(len(y))

    for train_index, test_index in loo.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Define the pipeline
        pipeline = make_pipeline(StandardScaler(), SVC(probability=True))

        # Inner loop: Hyperparameter tuning using GridSearchCV
        grid_search = GridSearchCV(pipeline, param_grid, cv=5,
                                   scoring='roc_auc')  # Use 5-fold cross-validation for inner loop
        grid_search.fit(X_train, y_train)

        # Train the best model on the entire training set
        best_model = grid_search.best_estimator_
        best_model.fit(X_train, y_train)

        # Predict probabilities on the test set
        probas = best_model.predict_proba(X_test)
        predictions[test_index] = probas[:, 1]
    '''clf = make_pipeline(StandardScaler(), LogisticRegression())
    loo = LeaveOneOut()

    predictions = cross_val_predict(clf, np.array(X), np.array(y), cv=loo, method='predict_proba')[:, 1]'''

    # Calculate F1 score
    y_pred = (predictions > 0.5).astype(int)  # Convert probabilities to binary predictions
    f1 = f1_score(y, y_pred, average='macro')
    print("Macro f1:", f1)

    ## Baseline
    clf = make_pipeline(StandardScaler(), DummyClassifier(strategy='stratified', random_state=42))
    base_predictions = cross_val_predict(clf, np.array(X), np.array(y), cv=loo, method='predict_proba')[:, 1]
    # Compute the F1 score for each class
    base_y_pred = (base_predictions > 0.5).astype(int)
    base_f1 = f1_score(y, base_y_pred, average='macro')
    print("Baseline Macro f1:", base_f1)
    plot_auc(y, predictions, base_predictions, feature_type, f1, base_f1)
def train_regressor(X, y, speakers, feature_type):
    reg = LinearRegression()
    clf = make_pipeline(StandardScaler(), reg)
    # Perform Leave-One-Out Cross-Validation
    predictions = cross_val_predict(clf, np.array(X), np.array(y), cv=len(X))

    ## Baseline
    baseline = DummyRegressor()
    clf = make_pipeline(StandardScaler(), baseline)
    # Perform Leave-One-Out Cross-Validation
    base_predictions = cross_val_predict(clf, np.array(X), np.array(y), cv=len(X))

    plot_predictions_gt(y, predictions, base_predictions, speakers, feature_type)

def plot_predictions_gt(y, predictions, base_predictions, speakers, feature_type):
    mae = mean_absolute_error(y, predictions)
    print("Mean Absolute Error:", mae)

    base_mae = mean_absolute_error(y, base_predictions)
    print("Baseline mean Absolute Error:", base_mae)

    # Get unique speaker names
    unique_speakers = sorted(set(speakers))
    num_colors = len(unique_speakers)
    color_map = plt.cm.get_cmap('tab20', num_colors)
    # Scatter plot with color coded points for each speaker
    for i, speaker in enumerate(unique_speakers):
        indices = [i for i, s in enumerate(speakers) if s == speaker]
        plt.scatter(np.array(y)[indices], np.array(predictions)[indices], label=speaker, color=color_map(i))

    # Plot predictions and ground truths
    # plt.scatter(y, predictions, c=speakers, cmap='tab10')
    plt.xlabel('Ground Truth')
    plt.ylabel('Predictions')

    plt.title(f'Features: {feature_type}, MAE: {mae:.2f}, Baseline MAE: {base_mae:.2f}')

    plt.legend(title='Speakers')
    plt.plot([np.array(y).min(), np.array(y).max()], [np.array(y).min(), np.array(y).max()], 'k--',
             lw=2)  # Plot the y=x line
    plt.savefig(f'../result_figures/features_{feature_type}_model_regressor.png')

    plt.show()

def plot_auc(y_test, probs, base_probs, feature_type, f1, base_f1):
    # Plot ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, probs)
    roc_auc = auc(fpr, tpr)
    b_fpr, b_tpr, b_thresholds = roc_curve(y_test, base_probs)
    base_roc_auc = auc(b_fpr, b_tpr)
    print("Roc AUC:", roc_auc)
    print("Baseline Roc AUC:", base_roc_auc)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'Features: {feature_type}, AUC = %0.2f, F1: %0.2f, '
                                                 f'Base AUC = %0.2f, Base F1: %0.2f' % (roc_auc, f1, base_roc_auc, base_f1))
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig(f'../result_figures/features_{feature_type}_roc_model_classifier.png')
    plt.show()
    # Compute Precision-Recall pairs
    precision, recall, _ = precision_recall_curve(y_test, probs)

    # Compute PR AUC
    pr_auc = auc(recall, precision)
    b_precision, b_recall, _ = precision_recall_curve(y_test, base_probs)
    base_pr_auc = auc(b_recall, b_precision)
    print("PR AUC:", pr_auc)
    print("Baseline PR AUC:", base_pr_auc)
    # Plot the Precision-Recall curve
    plt.figure()
    plt.plot(recall, precision, label=f'Features: {feature_type}, AUC = %0.2f, F1: %0.2f, '
                                      f'Base AUC = %0.2f, Base F1: %0.2f' % (pr_auc, f1, base_pr_auc, base_f1))
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower right")
    plt.savefig(f'../result_figures/features_{feature_type}_pr_model_classifier.png')
    plt.show()

def train_mlp(X, y, speakers):
    X = np.array(X)  # Assuming X is your list of embeddings
    X = torch.tensor(X, dtype=torch.float32)  # Convert to PyTorch tensor
    # Define hyperparameters
    learning_rate = 0.001
    num_epochs = 100
    input_size=1024
    # Initialize the model, loss function, and optimizer
    model = MLP(input_size, 1)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        # Forward pass
        outputs = model(X)
        loss = criterion(outputs, y)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

if __name__ == '__main__':
    # Load configuration from YAML file
    with open("config.yml", 'r') as config_file:
        config = yaml.safe_load(config_file)
    if config['ground_truths']['type'] == 'aggregates':
        '''if config['features']['type'] == 'embeddings':
            X, y = extract_trillsson_embeddings(config)'''
        X, y, speakers = extract_features(config)
    if config['model_type'] == 'regressor':
        train_regressor(X, y, speakers, config['features']['type'])
    elif config['model_type'] == 'classifier':
        train_classifier(X, y, config['features']['type'])
    else:
        train_mlp(X, y, speakers)