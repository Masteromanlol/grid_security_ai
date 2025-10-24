#!/usr/bin/env python
"""Script to split the full dataset into train, validation, and test sets."""

import torch
import os
from sklearn.model_selection import train_test_split

def split_dataset(full_dataset_path, train_path, val_path, test_path, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_state=42):
    """Split the dataset into train, val, test sets.

    Args:
        full_dataset_path: Path to the full dataset file
        train_path: Path to save train dataset
        val_path: Path to save val dataset
        test_path: Path to save test dataset
        train_ratio: Ratio for training set
        val_ratio: Ratio for validation set
        test_ratio: Ratio for test set
        random_state: Random state for reproducibility
    """
    # Load the full dataset
    dataset = torch.load(full_dataset_path, weights_only=False)
    print(f"Loaded dataset with {len(dataset)} samples")

    # Split indices
    indices = list(range(len(dataset)))
    train_val_indices, test_indices = train_test_split(indices, test_size=test_ratio, random_state=random_state)
    train_indices, val_indices = train_test_split(train_val_indices, test_size=val_ratio/(train_ratio + val_ratio), random_state=random_state)

    # Create subsets
    train_dataset = [dataset[i] for i in train_indices]
    val_dataset = [dataset[i] for i in val_indices]
    test_dataset = [dataset[i] for i in test_indices]

    print(f"Train set: {len(train_dataset)} samples")
    print(f"Val set: {len(val_dataset)} samples")
    print(f"Test set: {len(test_dataset)} samples")

    # Ensure output directories exist
    for path in [train_path, val_path, test_path]:
        os.makedirs(os.path.dirname(path), exist_ok=True)

    # Save datasets
    torch.save(train_dataset, train_path)
    torch.save(val_dataset, val_path)
    torch.save(test_dataset, test_path)

    print("Datasets saved successfully")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Split dataset into train, val, test")
    parser.add_argument("--full_dataset", default="data/processed/full_dataset.pt", help="Path to full dataset")
    parser.add_argument("--train_output", default="data/processed/train.pt", help="Path to save train dataset")
    parser.add_argument("--val_output", default="data/processed/val.pt", help="Path to save val dataset")
    parser.add_argument("--test_output", default="data/processed/test.pt", help="Path to save test dataset")
    parser.add_argument("--train_ratio", type=float, default=0.7, help="Train ratio")
    parser.add_argument("--val_ratio", type=float, default=0.15, help="Val ratio")
    parser.add_argument("--test_ratio", type=float, default=0.15, help="Test ratio")

    args = parser.parse_args()

    split_dataset(
        args.full_dataset,
        args.train_output,
        args.val_output,
        args.test_output,
        args.train_ratio,
        args.val_ratio,
        args.test_ratio
    )
