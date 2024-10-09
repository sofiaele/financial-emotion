import numpy as np
from collections import defaultdict

import torch
from torch.nn.utils.rnn import pad_sequence

def custom_collate_fn(batch, max_length):
    """
    Custom collate function for variable-length sequences.
    Pads sequences to a fixed maximum length passed as a parameter.
    """
    # Unpack the batch
    X_sequences, y_targets, ticker_targets, date_targets = zip(*batch)

    # Convert X_sequences to tensors and pad them
    X_sequences = [torch.tensor(seq) for seq in X_sequences]
    X_padded = pad_sequence(X_sequences, batch_first=True, padding_value=0)

    # Trim or pad sequences to the max_length
    X_padded = X_padded[:, :max_length] if X_padded.size(1) > max_length else torch.nn.functional.pad(X_padded, (0, max_length - X_padded.size(1)))

    # Get the lengths of each sequence
    lengths = torch.tensor([min(len(seq), max_length) for seq in X_sequences])

    # Convert y_targets to tensor
    y_targets = torch.tensor(y_targets)

    return X_padded, lengths, y_targets, ticker_targets, date_targets


def split_in_chronological_order(X, y, dates, tickers):
    # Step 1: Combine X, y, and dates into a list of tuples
    data = list(zip(X, y, dates, tickers))

    # Step 3: Group data by month-year
    month_year_groups = defaultdict(list)
    for x, y, date, ticker in data:
        month_year = (date.year, date.month)
        month_year_groups[month_year].append((x, y, date, ticker))

    # Step 4: Split month-year groups into train, validation, and test sets
    all_month_years = sorted(month_year_groups.keys())
    n = len(all_month_years)
    train_end = int(0.7 * n)
    val_end = int(0.8 * n)

    train_month_years = set(all_month_years[:train_end])
    val_month_years = set(all_month_years[train_end:val_end])
    test_month_years = set(all_month_years[val_end:])

    # Step 5: Collect data for each set based on month-year groups
    train_data = [item for month_year in train_month_years for item in month_year_groups[month_year]]
    val_data = [item for month_year in val_month_years for item in month_year_groups[month_year]]
    test_data = [item for month_year in test_month_years for item in month_year_groups[month_year]]

    # Step 6: Separate X, y, and dates for each set
    X_train, y_train, dates_train, tickers_train = zip(*train_data) if train_data else ([], [], [])
    X_val, y_val, dates_val, tickers_val = zip(*val_data) if val_data else ([], [], [])
    X_test, y_test, dates_test, tickers_test = zip(*test_data) if test_data else ([], [], [])

    # Convert lists to numpy arrays
    X_train = np.array(X_train)
    X_val = np.array(X_val)
    X_test = np.array(X_test)
    y_train = np.array(y_train)
    y_val = np.array(y_val)
    y_test = np.array(y_test)

    return X_train, y_train, dates_train, tickers_train, X_val, y_val, dates_val, tickers_val, X_test, y_test, dates_test, tickers_test
