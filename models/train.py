from sklearn.linear_model import LinearRegression
from sklearn.dummy import DummyRegressor, DummyClassifier
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from feature_extraction.aggregates import extract_features
from feature_extraction.audio_embeddings import extract_trillsson_embeddings
from sklearn.svm import SVC
from sklearn.metrics import f1_score
import yaml
from sklearn.metrics import roc_curve, auc
from sklearn.linear_model import LogisticRegression
from mlp import MLP
import torch.nn as nn
import torch.optim as optim
import torch

def train_classifier(X, y, feature_type):
    # Now, y_binary contains 1 for positive or zero values of y, and 0 for negative values of y
    y = np.where(np.array(y) >= 0, 1, 0)

    class_counts = np.bincount(y)
    print("Class Counts:", class_counts)

    #clf = make_pipeline(StandardScaler(), SVC(probability=True))
    clf = make_pipeline(StandardScaler(), LogisticRegression())
    loo = LeaveOneOut()

    predictions = cross_val_predict(clf, np.array(X), np.array(y), cv=loo, method='predict_proba')[:, 1]

    # Calculate F1 score
    y_pred = (predictions > 0.5).astype(int)  # Convert probabilities to binary predictions
    f1 = f1_score(y, y_pred, average='macro')
    print("Macro f1:", f1)

    ## Baseline
    clf = make_pipeline(StandardScaler(), DummyClassifier())
    base_predictions = cross_val_predict(clf, np.array(X), np.array(y), cv=loo)
    # Compute the F1 score for each class
    base_f1 = f1_score(y, base_predictions, average='macro')
    print("Baseline Macro f1:", base_f1)
    plot_roc(y, predictions, feature_type, f1, base_f1)
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

def plot_roc(y_test, probs, feature_type, f1, base_f1):
    # Plot ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, probs)
    roc_auc = auc(fpr, tpr)
    print("AUC:", roc_auc)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'Features: {feature_type}, AUC = %0.2f, F1: %0.2f, Base F1: %0.2f' % (roc_auc, f1, base_f1))
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig(f'../result_figures/features_{feature_type}_model_classifier.png')
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
        if config['features']['type'] == 'embeddings':
            X, y = extract_trillsson_embeddings(config)
        else:
            X, y, speakers = extract_features(config)
    if config['model_type'] == 'regressor':
        train_regressor(X, y, speakers, config['features']['type'])
    elif config['model_type'] == 'classifier':
        train_classifier(X, y, config['features']['type'])
    else:
        train_mlp(X, y, speakers)