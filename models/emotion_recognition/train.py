"""Training script for emotion recognition model"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from tqdm import tqdm
import argparse
from sklearn.metrics import f1_score, accuracy_score
import yaml
import warnings
warnings.filterwarnings('ignore')

# Try to import SpeechBrain's DynamicBatchSampler
try:
    from speechbrain.dataio.sampler import DynamicBatchSampler
    SPEECHBRAIN_AVAILABLE = True
except ImportError:
    SPEECHBRAIN_AVAILABLE = False
    print("Warning: SpeechBrain not available. Using standard DataLoader.")

from model import EmotionRecognitionModel, MultitaskLoss
from dataset import IEMOCAPDataset, MSPPodcastDataset, collate_fn_dynamic
from preprocessing import AudioPreprocessor
from config import *


class Trainer:
    """Trainer class for emotion recognition model"""

    def __init__(
        self,
        model: EmotionRecognitionModel,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: MultitaskLoss,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        device: str = 'cuda',
        checkpoint_dir: str = MODEL_CHECKPOINT_DIR
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.checkpoint_dir = checkpoint_dir

        os.makedirs(checkpoint_dir, exist_ok=True)

        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0

    def train_epoch(self, epoch: int):
        """Train for one epoch"""
        self.model.train()

        total_losses = {
            'total_loss': 0.0,
            'emotion_loss': 0.0,
            'valence_loss': 0.0,
            'arousal_loss': 0.0
        }

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} [Train]")
        for batch_idx, (mel_specs, labels, lengths) in enumerate(pbar):
            mel_specs = mel_specs.to(self.device)
            labels = {k: v.to(self.device) for k, v in labels.items()}

            # Forward pass
            predictions = self.model(mel_specs)

            # Compute loss
            losses = self.criterion(predictions, labels)

            # Backward pass
            self.optimizer.zero_grad()
            losses['total_loss'].backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), GRADIENT_CLIP)

            self.optimizer.step()

            # Accumulate losses
            for key in total_losses.keys():
                total_losses[key] += losses[key].item()

            # Update progress bar
            pbar.set_postfix({
                'loss': losses['total_loss'].item(),
                'emotion': losses['emotion_loss'].item(),
                'valence': losses['valence_loss'].item(),
                'arousal': losses['arousal_loss'].item()
            })

        # Average losses
        for key in total_losses.keys():
            total_losses[key] /= len(self.train_loader)

        return total_losses

    def validate(self, epoch: int):
        """Validate the model"""
        self.model.eval()

        total_losses = {
            'total_loss': 0.0,
            'emotion_loss': 0.0,
            'valence_loss': 0.0,
            'arousal_loss': 0.0
        }

        all_predictions = {'emotion': [], 'valence': [], 'arousal': []}
        all_targets = {'emotion': [], 'valence': [], 'arousal': []}

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f"Epoch {epoch} [Val]")
            for mel_specs, labels, lengths in pbar:
                mel_specs = mel_specs.to(self.device)
                labels = {k: v.to(self.device) for k, v in labels.items()}

                # Forward pass
                predictions = self.model(mel_specs)

                # Compute loss
                losses = self.criterion(predictions, labels)

                # Accumulate losses
                for key in total_losses.keys():
                    total_losses[key] += losses[key].item()

                # Store predictions and targets
                for task in ['emotion', 'valence', 'arousal']:
                    pred_labels = torch.argmax(predictions[task], dim=1)
                    all_predictions[task].extend(pred_labels.cpu().numpy())
                    all_targets[task].extend(labels[task].cpu().numpy())

                # Update progress bar
                pbar.set_postfix({'loss': losses['total_loss'].item()})

        # Average losses
        for key in total_losses.keys():
            total_losses[key] /= len(self.val_loader)

        # Compute metrics
        metrics = {}
        for task in ['emotion', 'valence', 'arousal']:
            f1 = f1_score(all_targets[task], all_predictions[task], average='macro')
            acc = accuracy_score(all_targets[task], all_predictions[task])
            metrics[f'{task}_f1'] = f1
            metrics[f'{task}_acc'] = acc

        return total_losses, metrics

    def train(self, num_epochs: int, patience: int = EARLY_STOPPING_PATIENCE):
        """Full training loop"""
        print(f"\nStarting training for {num_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        for epoch in range(1, num_epochs + 1):
            # Train
            train_losses = self.train_epoch(epoch)

            # Validate
            val_losses, val_metrics = self.validate(epoch)

            # Update scheduler
            self.scheduler.step(val_losses['total_loss'])

            # Print epoch summary
            print(f"\nEpoch {epoch}/{num_epochs}")
            print(f"Train Loss: {train_losses['total_loss']:.4f} | Val Loss: {val_losses['total_loss']:.4f}")
            print(f"Emotion F1: {val_metrics['emotion_f1']:.4f} | Valence F1: {val_metrics['valence_f1']:.4f} | Arousal F1: {val_metrics['arousal_f1']:.4f}")
            print(f"LR: {self.optimizer.param_groups[0]['lr']:.6f}")

            # Save best model
            if val_losses['total_loss'] < self.best_val_loss:
                self.best_val_loss = val_losses['total_loss']
                self.epochs_without_improvement = 0

                checkpoint_path = os.path.join(self.checkpoint_dir, 'best_model.pt')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_losses['total_loss'],
                    'val_metrics': val_metrics,
                }, checkpoint_path)
                print(f"✓ Saved best model (val_loss: {val_losses['total_loss']:.4f})")
            else:
                self.epochs_without_improvement += 1

            # Early stopping
            if self.epochs_without_improvement >= patience:
                print(f"\nEarly stopping triggered after {epoch} epochs")
                break

        print("\nTraining completed!")


def create_dataloaders(
    train_dataset,
    val_dataset,
    batch_size: int = BATCH_SIZE,
    use_dynamic_sampler: bool = True
):
    """Create data loaders with optional DynamicBatchSampler"""

    if use_dynamic_sampler and SPEECHBRAIN_AVAILABLE:
        print("Using DynamicBatchSampler from SpeechBrain")

        # Get sequence lengths
        train_lengths = [train_dataset.preprocessor.process_audio_file(path).shape[1]
                        for path in train_dataset.audio_paths]

        # Create dynamic batch sampler
        train_sampler = DynamicBatchSampler(
            train_dataset,
            max_batch_length=batch_size * 500,  # Approximate max frames per batch
            num_buckets=50,
            length_func=lambda x: train_lengths[x],
            shuffle=True,
        )

        train_loader = DataLoader(
            train_dataset,
            batch_sampler=train_sampler,
            collate_fn=collate_fn_dynamic,
            num_workers=4
        )
    else:
        print("Using standard DataLoader")
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn_dynamic,
            num_workers=4
        )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn_dynamic,
        num_workers=4
    )

    return train_loader, val_loader


def main(args):
    """Main training function"""

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)

    # Create preprocessor
    preprocessor = AudioPreprocessor()

    # Load datasets
    print(f"\nLoading {args.dataset} dataset...")
    if args.dataset == 'iemocap':
        train_dataset, val_dataset, test_dataset = IEMOCAPDataset.load_iemocap(
            args.data_path, preprocessor=preprocessor
        )
    elif args.dataset == 'msp-podcast':
        train_dataset, val_dataset, test_dataset = MSPPodcastDataset.load_msp_podcast(
            args.data_path, preprocessor=preprocessor
        )
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    # Create data loaders
    train_loader, val_loader = create_dataloaders(
        train_dataset, val_dataset,
        batch_size=args.batch_size,
        use_dynamic_sampler=args.use_dynamic_sampler
    )

    # Create model
    print("\nCreating model...")
    model = EmotionRecognitionModel(freeze_backbone=args.freeze_backbone)

    # Create loss function
    criterion = MultitaskLoss()

    # Create optimizer
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)

    # Create scheduler
    scheduler = ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )

    # Create trainer
    trainer = Trainer(
        model, train_loader, val_loader, criterion, optimizer, scheduler,
        device=device, checkpoint_dir=args.checkpoint_dir
    )

    # Train
    trainer.train(num_epochs=args.num_epochs, patience=args.patience)

    # Test best model
    print("\nEvaluating best model on test set...")
    checkpoint = torch.load(os.path.join(args.checkpoint_dir, 'best_model.pt'))
    model.load_state_dict(checkpoint['model_state_dict'])

    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn_dynamic, num_workers=4
    )

    model.eval()
    all_predictions = {'emotion': [], 'valence': [], 'arousal': []}
    all_targets = {'emotion': [], 'valence': [], 'arousal': []}

    with torch.no_grad():
        for mel_specs, labels, lengths in tqdm(test_loader, desc="Testing"):
            mel_specs = mel_specs.to(device)
            predictions = model(mel_specs)

            for task in ['emotion', 'valence', 'arousal']:
                pred_labels = torch.argmax(predictions[task], dim=1)
                all_predictions[task].extend(pred_labels.cpu().numpy())
                all_targets[task].extend(labels[task].cpu().numpy())

    # Compute test metrics
    print("\nTest Results:")
    for task in ['emotion', 'valence', 'arousal']:
        f1 = f1_score(all_targets[task], all_predictions[task], average='macro')
        acc = accuracy_score(all_targets[task], all_predictions[task])
        print(f"{task.capitalize()} - F1: {f1:.4f}, Accuracy: {acc:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train emotion recognition model')

    # Dataset arguments
    parser.add_argument('--dataset', type=str, required=True,
                       choices=['iemocap', 'msp-podcast'],
                       help='Dataset to use for training')
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to dataset root directory')

    # Model arguments
    parser.add_argument('--freeze_backbone', action='store_true',
                       help='Freeze TRILLsson3 backbone weights (only train task heads)')

    # Training arguments
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE,
                       help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE,
                       help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=NUM_EPOCHS,
                       help='Number of epochs')
    parser.add_argument('--patience', type=int, default=EARLY_STOPPING_PATIENCE,
                       help='Early stopping patience')
    parser.add_argument('--use_dynamic_sampler', action='store_true',
                       help='Use DynamicBatchSampler from SpeechBrain')

    # Output arguments
    parser.add_argument('--checkpoint_dir', type=str, default=MODEL_CHECKPOINT_DIR,
                       help='Directory to save model checkpoints')

    args = parser.parse_args()

    main(args)
