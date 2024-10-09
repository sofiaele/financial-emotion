import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
from data import FinancialDataset
import yaml
import numpy as np
from feature_extraction.aggregates import extract_features
from utils import split_in_chronological_order, custom_collate_fn
from model import LSTMClassifier
import argparse
from tqdm import tqdm

def train_model(model, optimizer, train_loader, dev_loader, criterion, epochs, patience=5):
    best_val_roc_auc = 0.0
    epochs_without_improvement = 0
    best_model_state = None

    for epoch in tqdm(range(epochs), desc="Training", unit="epoch"):

        model.train()
        total_loss = 0.0
        y_true = []
        y_pred = []

        for X_batch, lengths, y_batch, tickers_batch, dates_batch in train_loader:
            optimizer.zero_grad()
            # Forward pass (no softmax/sigmoid inside the model)
            outputs = model(X_batch.float())

            # Apply sigmoid manually since the model output is raw logits
            predictions = torch.sigmoid(outputs.squeeze())

            # Calculate binary cross-entropy loss
            loss = criterion(predictions, y_batch.float())
            total_loss += loss.item()

            # Backpropagation and optimization
            loss.backward()
            optimizer.step()

            # Convert predictions to binary (0 or 1) for accuracy
            pred_labels = (predictions >= 0.5).int()

            # Collect true labels and predictions for metrics
            y_true += y_batch.int().tolist()
            y_pred += pred_labels.tolist()

        # Calculate accuracy and ROC AUC for training
        train_acc = accuracy_score(y_true, y_pred)
        train_roc_auc = roc_auc_score(y_true, y_pred)

        # Evaluate on the validation set
        val_loss, val_acc, val_roc_auc, _, _ = evaluate_model(model, dev_loader, criterion)

        print(f"Train loss: {total_loss / len(train_loader):.4f} - acc: {train_acc:.4f} - roc_auc: {train_roc_auc:.4f}")
        print(f"Validation loss: {val_loss:.4f} - acc: {val_acc:.4f} - roc_auc: {val_roc_auc:.4f}")

        # Early stopping check
        if val_roc_auc > best_val_roc_auc:
            best_val_roc_auc = val_roc_auc
            epochs_without_improvement = 0
            best_model_state = model.state_dict()  # Save the best model's state
        else:
            epochs_without_improvement += 1

        # If no improvement in ROC AUC for 'patience' epochs, stop training
        if epochs_without_improvement >= patience:
            print(f'Early stopping triggered after {epoch + 1} epochs. Best ROC AUC: {best_val_roc_auc:.4f}')
            break

    # Load the best model state if early stopping was triggered
    if best_model_state:
        model.load_state_dict(best_model_state)

    return model


def evaluate_model(model, dev_loader, criterion):
    model.eval()
    total_val_loss = 0.0
    y_true = []
    y_pred = []

    with torch.no_grad():
        for X_batch, lengths, y_batch, tickers_batch, dates_batch in dev_loader:
            # Forward pass
            outputs = model(X_batch.float())

            # Apply sigmoid to get probabilities
            predictions = torch.sigmoid(outputs.squeeze())

            # Calculate the validation loss
            loss = criterion(predictions, y_batch.float())
            total_val_loss += loss.item()

            # Convert predictions to binary (0 or 1) for accuracy
            pred_labels = (predictions >= 0.5).int()

            # Collect true labels and predictions for metrics
            y_true += y_batch.int().tolist()
            y_pred += pred_labels.tolist()

    # Calculate accuracy and ROC AUC for validation
    val_acc = accuracy_score(y_true, y_pred)
    val_roc_auc = roc_auc_score(y_true, y_pred)

    return total_val_loss / len(dev_loader), val_acc, val_roc_auc, y_true, y_pred

def train(config):
    X, y, speakers, feature_names, tickers, dates = extract_features(config)
    y = np.where(np.array(y) >= 0.0, 1, 0)
    X_train, y_train, dates_train, \
        tickers_train, X_val, y_val, \
        dates_val, tickers_val, X_test, \
        y_test, dates_test, tickers_test = split_in_chronological_order(X, y, dates, tickers)
    # Create datasets
    num_sequence_months = config["lstm"]["num_sequence_months"]
    ticker_mode = config["lstm"]["ticker_mode"]
    train_dataset = FinancialDataset(X_train, y_train, tickers_train, dates_train, num_sequence_months=num_sequence_months, ticker_mode=ticker_mode)
    val_dataset = FinancialDataset(X_val, y_val, tickers_val, dates_val, num_sequence_months=num_sequence_months, ticker_mode=ticker_mode)
    test_dataset = FinancialDataset(X_test, y_test, tickers_test, dates_test, num_sequence_months=num_sequence_months, ticker_mode=ticker_mode)

    max_length = train_dataset.max_length
    # Create dataloaders
    for key, value in config["lstm"].items():
        setattr(args, key, value)
    output_size = 1
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=lambda batch: custom_collate_fn(batch, max_length))
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=lambda batch: custom_collate_fn(batch, max_length))
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=lambda batch: custom_collate_fn(batch, max_length))
    sample_batch = next(iter(train_loader))  # Get a batch from your DataLoader
    X_sample, _, _, _, _ = sample_batch  # Unpack the batch

    # Determine the feature dimension
    input_size = X_sample.size(-1)
    model = LSTMClassifier(input_size, args.embedding_dim, args.hidden_size, output_size, args.num_layers)

    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    model = train_model(model, optimizer, train_loader, val_loader, criterion, args.epochs, args.patience)

    _, test_acc, test_roc_auc, y_true, y_pred = evaluate_model(model, test_loader, criterion)
    print(f"Test acc: {test_acc:.4f} - roc_auc: {test_roc_auc:.4f}")
    # Print classification report and confusion matrix
    print(f"Classification Report:\n{classification_report(y_true, y_pred)}")
    print(f"Confusion Matrix:\n{confusion_matrix(y_true, y_pred)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run the pipeline.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--config", "-c", default="config.yml",
                        help="Path to the config .yaml file, containing the configuration.")
    args = parser.parse_args()
    # If the config file is specified, then parse the arguments from it
    with open(args.config, 'r') as f:
        config_data = yaml.safe_load(f)
    train(config_data)