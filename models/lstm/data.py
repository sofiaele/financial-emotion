import random
from collections import defaultdict
import torch
from torch.utils.data import Dataset
import itertools


class FinancialDataset(Dataset):
    def __init__(self, X, y, tickers, dates, num_sequence_months=3, ticker_mode='all'):
        """
        Args:
            X (list): Feature data (list of numpy arrays).
            y (list or np.array): Labels for each sequence.
            tickers (list): List of tickers corresponding to each sequence.
            dates (list): List of dates corresponding to each sequence.
            sequence_months (int): Number of consecutive months to use in the sequence.
            ticker_mode (str): Defines how to handle tickers.
                'same': Ensure that the sequence contains the same ticker across months.
                'all': Use all tickers across the month-years (default).
                'one': Use one ticker per month-year in the sequence.
        """
        self.X = X
        self.y = y
        self.tickers = tickers
        self.dates = dates
        self.num_sequence_months = num_sequence_months
        self.ticker_mode = ticker_mode

        # Group data by (year, month)
        self.month_year_groups = self._group_by_month_year()
        self.samples, self.max_length = self._create_sequences()

    def _group_by_month_year(self):
        """
        Group data by (year, month) and return a dictionary mapping (year, month) to indices.
        """
        month_year_groups = defaultdict(list)
        for idx, date in enumerate(self.dates):
            month_year = (date.year, date.month)
            month_year_groups[month_year].append(idx)
        return month_year_groups

    def _create_sequences(self):
        """
        Create sequences based on the ticker_mode.
        """
        samples = []
        all_month_years = sorted(self.month_year_groups.keys())

        for i in range(len(all_month_years) - self.num_sequence_months + 1):
            # Take a window of 3 consecutive months
            sequence_months = all_month_years[i:i + self.num_sequence_months]

            if self.ticker_mode == 'same':
                # Sequence samples from consecutive months that belong to the same ticker
                samples += self._create_same_ticker_sequences(sequence_months)

            elif self.ticker_mode == 'one':
                # Use only one ticker per month-year in the sequence
                samples += self._create_one_ticker_per_month_sequences(sequence_months)

            else:  # self.ticker_mode == 'all'
                # Use all tickers from the month-year groups (current approach)
                samples += self._create_all_tickers_sequences(sequence_months)
        if self.ticker_mode == 'all':
            sequence_length = max(len(X_sequence) for X_sequence, y_target, ticker_target, date_target in samples)
        else:
            sequence_length = self.num_sequence_months
        return samples, sequence_length

    def _create_same_ticker_sequences(self, sequence_months):
        """
        Create sequences that have the same ticker across all months in the sequence.
        """
        samples = []
        common_tickers = None

        # Find common tickers that exist in all the months of the sequence
        for month_year in sequence_months:
            month_tickers = set(self.tickers[idx] for idx in self.month_year_groups[month_year])
            if common_tickers is None:
                common_tickers = month_tickers
            else:
                common_tickers &= month_tickers  # Intersection of tickers across months

        # If no common tickers exist, skip this sequence
        if not common_tickers:
            return []

        # For each common ticker, create a sequence
        for ticker in common_tickers:
            sequence_indices = []
            for month_year in sequence_months:
                # Find the index for the specific ticker in each month
                for idx in self.month_year_groups[month_year]:
                    if self.tickers[idx] == ticker:
                        sequence_indices.append(idx)
                        break
            if len(sequence_indices) == self.num_sequence_months:
                last_idx = sequence_indices[-1]
                X_sequence = [self.X[idx] for idx in sequence_indices]
                y_target = self.y[last_idx]
                ticker_target = self.tickers[last_idx]
                date_target = self.dates[last_idx]
                samples.append((X_sequence, y_target, ticker_target, date_target))

        return samples

    def _create_one_ticker_per_month_sequences(self, sequence_months):
        """
        Create sequences with all possible combinations of one ticker per month-year in the sequence.
        """
        samples = []
        ticker_indices_per_month = []

        # Collect all ticker indices for each month in the sequence
        for month_year in sequence_months:
            ticker_indices_per_month.append(self.month_year_groups[month_year])

        # Generate all possible combinations of ticker indices (one from each month)
        all_combinations = list(itertools.product(*ticker_indices_per_month))

        # For each combination, create a sequence
        for sequence_indices in all_combinations:
            last_idx = sequence_indices[-1]
            X_sequence = [self.X[idx] for idx in sequence_indices]
            y_target = self.y[last_idx]
            ticker_target = self.tickers[last_idx]
            date_target = self.dates[last_idx]
            samples.append((X_sequence, y_target, ticker_target, date_target))

        return samples

    def _create_all_tickers_sequences(self, sequence_months):
        """
        Create sequences that use all tickers from the month-years.
        """
        samples = []
        sequence_indices = []

        # Collect all the samples from those months
        for month_year in sequence_months:
            sequence_indices += self.month_year_groups[month_year]

        # Ensure the sequence is the right length
        if len(sequence_indices) == self.num_sequence_months:
            last_idx = sequence_indices[-1]
            X_sequence = [self.X[idx] for idx in sequence_indices]
            y_target = self.y[last_idx]
            ticker_target = self.tickers[last_idx]
            date_target = self.dates[last_idx]
            samples.append((X_sequence, y_target, ticker_target, date_target))

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Given an index, return the corresponding sequence of features, label, ticker, and date.
        """
        X_sequence, y_target, ticker_target, date_target = self.samples[idx]
        # Convert features and labels to torch tensors
        X_tensor = torch.tensor(X_sequence, dtype=torch.float32)  # (sequence_length, input_size)
        y_tensor = torch.tensor(y_target, dtype=torch.float32)  # Label for the last month in sequence

        return X_tensor, y_tensor, ticker_target, date_target
