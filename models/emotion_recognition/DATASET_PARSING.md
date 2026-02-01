# Dataset Parsing Guide

This guide explains how to parse and load IEMOCAP and MSP-Podcast datasets for emotion recognition training.

## Overview

The parsing system consists of:
- **`parsers.py`**: Dataset parsers for IEMOCAP and MSP-Podcast
- **`dataset.py`**: PyTorch Dataset classes that use the parsers
- **`examples/parse_datasets.py`**: Example scripts for dataset parsing

## Features

- ✅ **Multi-task learning**: Parses emotion, valence, and arousal labels
- ✅ **Combined training**: Supports simultaneous training on both datasets
- ✅ **Flexible splits**: Customizable train/val/test splits for IEMOCAP
- ✅ **CSV caching**: Save parsed datasets to CSV for faster loading

## Label Mappings

### Emotions (4 classes)
Both datasets are mapped to 4 primary emotions:
- **0**: Angry
- **1**: Happy (includes excitement for IEMOCAP)
- **2**: Neutral
- **3**: Sad

### Valence (3 classes)
- **0**: Negative (low valence)
- **1**: Neutral (medium valence)
- **2**: Positive (high valence)

### Arousal (3 classes)
- **0**: Low (weak arousal)
- **1**: Neutral (medium arousal)
- **2**: High (strong arousal)

## IEMOCAP Dataset

### Structure Expected
```
IEMOCAP/
├── Session1/
│   ├── sentences/
│   │   └── wav/
│   │       ├── Ses01F_impro01/
│   │       │   ├── Ses01F_impro01_F000.wav
│   │       │   └── ...
│   │       └── ...
│   └── dialog/
│       └── EmoEvaluation/
│           ├── Ses01F_impro01.txt
│           └── ...
├── Session2/
├── Session3/
├── Session4/
└── Session5/
```

### Parsing IEMOCAP

```python
from parsers import IEMOCAPParser

# Initialize parser
parser = IEMOCAPParser('/path/to/IEMOCAP')

# Parse with default split (train: [1,2,4], val: [3], test: [5])
df = parser.parse()

# Or customize splits
split_config = {
    'train': [1, 2],
    'val': [3],
    'test': [4, 5]
}
df = parser.parse(split_config=split_config)

# Save to CSV
df.to_csv('iemocap_dataset.csv', index=False)
```

### DataFrame Columns
- `file`: Full path to audio file
- `emotion`: Emotion label (string: "angry", "happy", "neutral", "sad")
- `emotion_label`: Emotion label (int: 0-3)
- `valence`: Valence label (string: "negative", "neutral", "positive")
- `valence_label`: Valence label (int: 0-2)
- `arousal`: Arousal label (string: "low", "neutral", "high")
- `arousal_label`: Arousal label (int: 0-2)
- `split`: Data split ("train", "val", "test")
- `speaker_id`: Speaker ID (e.g., "Ses01F")
- `dataset`: Dataset name ("iemocap")

## MSP-Podcast Dataset

### Structure Expected

**Option 1: Using msp.csv**
```
msp.csv with columns:
- file: /path/to/audio.wav
- emotion: neutral/happy/sad/angry
- split: train1/test1/test2
```

**Option 2: Using official structure**
```
MSP-Podcast/
├── Audios/
│   ├── MSP-PODCAST_0001_0008.wav
│   └── ...
└── Labels/
    └── labels_consensus.csv
```

### Parsing MSP-Podcast

```python
from parsers import MSPPodcastParser

# Option 1: From msp.csv
parser = MSPPodcastParser(
    data_path='/path/to/MSP-Podcast',
    csv_path='/path/to/msp.csv'
)
df = parser.parse()

# Option 2: From labels_consensus.csv (includes valence/arousal)
parser = MSPPodcastParser('/path/to/MSP-Podcast')
df = parser.parse()

# Save to CSV
df.to_csv('msp_podcast_dataset.csv', index=False)
```

### Valence/Arousal Discretization

MSP-Podcast provides continuous valence/arousal on a 1-7 scale. We discretize to 3 classes:
- **< 3.5**: Low/Negative (class 0)
- **3.5 - 5.5**: Neutral (class 1)
- **> 5.5**: High/Positive (class 2)

## Combined Dataset (IEMOCAP + MSP-Podcast)

### Parse Both Datasets

```python
from parsers import create_combined_dataset

# Parse and combine both datasets
train_df, val_df, test_df = create_combined_dataset(
    iemocap_path='/path/to/IEMOCAP',
    msp_path='/path/to/MSP-Podcast',
    msp_csv_path='/path/to/msp.csv',  # Optional
    output_csv='combined_dataset.csv'  # Optional: save to file
)

print(f"Train: {len(train_df)} samples")
print(f"Val: {len(val_df)} samples")
print(f"Test: {len(test_df)} samples")
```

### Dataset Statistics

The combined dataset will show:
```
Combined dataset statistics:
  Train: XXXX samples
  Val: XXXX samples
  Test: XXXX samples
  Total: XXXX samples
```

Each sample has a `dataset` column indicating origin ("iemocap" or "msp_podcast").

## Using with PyTorch

### Load from Parsers

```python
from dataset import load_datasets_from_parsers
from preprocessing import AudioPreprocessor

preprocessor = AudioPreprocessor()

train_dataset, val_dataset, test_dataset = load_datasets_from_parsers(
    iemocap_path='/path/to/IEMOCAP',
    msp_path='/path/to/MSP-Podcast',
    msp_csv_path='/path/to/msp.csv',  # Optional
    preprocessor=preprocessor
)

# Use with DataLoader
from torch.utils.data import DataLoader
from dataset import collate_fn_dynamic

train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True,
    collate_fn=collate_fn_dynamic
)
```

### Load from CSV (Faster)

If you've already parsed and saved to CSV:

```python
from dataset import load_datasets_from_csv
from preprocessing import AudioPreprocessor

preprocessor = AudioPreprocessor()

train_dataset, val_dataset, test_dataset = load_datasets_from_csv(
    csv_path='combined_dataset.csv',
    preprocessor=preprocessor
)
```

## Command-Line Usage

### Using the Example Script

```bash
# Parse only IEMOCAP
python examples/parse_datasets.py \
    --iemocap_path /path/to/IEMOCAP \
    --mode iemocap \
    --output_csv iemocap_dataset.csv

# Parse only MSP-Podcast (with msp.csv)
python examples/parse_datasets.py \
    --msp_path /path/to/MSP-Podcast \
    --msp_csv /path/to/msp.csv \
    --mode msp \
    --output_csv msp_podcast_dataset.csv

# Parse only MSP-Podcast (with labels_consensus.csv)
python examples/parse_datasets.py \
    --msp_path /path/to/MSP-Podcast \
    --mode msp \
    --output_csv msp_podcast_dataset.csv

# Parse and combine both datasets
python examples/parse_datasets.py \
    --iemocap_path /path/to/IEMOCAP \
    --msp_path /path/to/MSP-Podcast \
    --msp_csv /path/to/msp.csv \
    --mode combined \
    --output_csv combined_dataset.csv
```

### Using Parsers Directly

```bash
# IEMOCAP only
cd models/emotion_recognition
python parsers.py \
    --iemocap_path /path/to/IEMOCAP \
    --output_csv iemocap_dataset.csv

# MSP-Podcast with msp.csv
python parsers.py \
    --msp_path /path/to/MSP-Podcast \
    --msp_csv /path/to/msp.csv \
    --output_csv msp_dataset.csv

# Both datasets combined
python parsers.py \
    --iemocap_path /path/to/IEMOCAP \
    --msp_path /path/to/MSP-Podcast \
    --output_csv combined_dataset.csv
```

## Training with Combined Dataset

### Update Training Script

```python
from dataset import load_datasets_from_parsers
from preprocessing import AudioPreprocessor
from model import EmotionRecognitionModel
from train import Trainer

# Load datasets
preprocessor = AudioPreprocessor()
train_dataset, val_dataset, test_dataset = load_datasets_from_parsers(
    iemocap_path='/path/to/IEMOCAP',
    msp_path='/path/to/MSP-Podcast',
    preprocessor=preprocessor
)

# Initialize model for multi-task learning
model = EmotionRecognitionModel(
    num_emotions=4,
    num_valence_classes=3,
    num_arousal_classes=3
)

# Train
trainer = Trainer(
    model=model,
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    config=config
)
trainer.train(num_epochs=100)
```

## Best Practices

### 1. CSV Caching

Parse once, save to CSV, then load from CSV for faster subsequent runs:

```python
# First time: Parse and save
train_df, val_df, test_df = create_combined_dataset(
    iemocap_path='/path/to/IEMOCAP',
    msp_path='/path/to/MSP-Podcast',
    output_csv='combined_dataset.csv'
)

# Subsequent times: Load from CSV
train_dataset, val_dataset, test_dataset = load_datasets_from_csv(
    csv_path='combined_dataset.csv',
    preprocessor=preprocessor
)
```

### 2. Verify Dataset Quality

```python
df = pd.read_csv('combined_dataset.csv')

# Check for missing files
missing_files = []
for file_path in df['file']:
    if not os.path.exists(file_path):
        missing_files.append(file_path)

if missing_files:
    print(f"Warning: {len(missing_files)} files not found")
    # Remove missing files
    df = df[df['file'].apply(os.path.exists)]
    df.to_csv('combined_dataset_cleaned.csv', index=False)
```

### 3. Balance Datasets

If one dataset dominates, consider balancing:

```python
# Check dataset distribution
print(df['dataset'].value_counts())

# Sample to balance
from sklearn.utils import resample

iemocap_df = df[df['dataset'] == 'iemocap']
msp_df = df[df['dataset'] == 'msp_podcast']

# Upsample smaller dataset or downsample larger
min_size = min(len(iemocap_df), len(msp_df))
iemocap_balanced = resample(iemocap_df, n_samples=min_size, random_state=42)
msp_balanced = resample(msp_df, n_samples=min_size, random_state=42)

balanced_df = pd.concat([iemocap_balanced, msp_balanced])
```

## Troubleshooting

### Issue: Files Not Found

**Solution**: Check that paths in your CSV are absolute paths and files exist:
```python
df = pd.read_csv('dataset.csv')
df['exists'] = df['file'].apply(os.path.exists)
print(f"Files exist: {df['exists'].sum()}/{len(df)}")
```

### Issue: Missing Valence/Arousal

**Solution**: Some datasets may not have valence/arousal. The dataset handles this gracefully:
```python
# Labels will only include available annotations
# Check with:
print(f"Has valence: {dataset.has_valence}")
print(f"Has arousal: {dataset.has_arousal}")
```

### Issue: Different Label Schemes

**Solution**: Parsers handle mapping automatically. Check mappings in `parsers.py`:
- IEMOCAP: Maps "exc" (excitement) → "happy"
- MSP-Podcast: Maps various emotion labels to 4 core emotions

## Reference

Based on the `ssl_ser` project structure, adapted for this emotion recognition task with multi-task learning support (emotion + valence + arousal).

### Key Differences from ssl_ser:
1. Supports combined dataset loading (IEMOCAP + MSP-Podcast)
2. Parses and returns DataFrames for flexibility
3. Includes valence and arousal labels
4. Simplified for emotion classification (vs. SSL feature extraction)

### Related Files:
- `models/emotion_recognition/parsers.py` - Dataset parsers
- `models/emotion_recognition/dataset.py` - PyTorch Dataset classes
- `examples/parse_datasets.py` - Example usage script
