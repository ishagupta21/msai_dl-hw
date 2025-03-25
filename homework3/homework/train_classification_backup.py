import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.utils.tensorboard as tb

from .datasets.classification_dataset import load_data

from .models import load_model, save_model
from .metrics import AccuracyMetric  # Import AccuracyMetric


def train(
    exp_dir: str = "logs",
    model_name: str = "Classifier",
    num_epoch: int = 1,
    lr: float = 1e-3,
    batch_size: int = 128,
    seed: int = 2024,
    **kwargs,
):
    # Set random seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)


    # Set device (GPU, MPS, or CPU)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Load training and validation data
    train_loader = load_data("classification_data/train", shuffle=False, batch_size=batch_size, num_workers=2, transform_pipeline="aug")
    val_loader = load_data("classification_data/val", shuffle=False, batch_size=batch_size, num_workers=2, transform_pipeline="default")

    # Initialize model, loss, optimizer, and metrics
    model = load_model(model_name, **kwargs).to(device)
    model.train()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    #optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.92, weight_decay=1e-4)
    train_accuracy_metric = AccuracyMetric()
    val_accuracy_metric = AccuracyMetric()
    # Training loop
    for epoch in range(num_epoch):
        print(f"Epoch {epoch + 1}/{num_epoch}")

        # Training phase
        model.train()
        train_accuracy_metric.reset()

        train_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            logits = model(images)
            loss = criterion(logits, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Convert logits to predicted class indices
            predicted_classes = logits.argmax(dim=1)

            # Update metrics
            train_accuracy_metric.add(predicted_classes, labels)  # Pass predicted classes instead of raw logits

            
        train_accuracy = train_accuracy_metric.compute()
        print(f"Train Accuracy: {train_accuracy}")

        # Validation phase
        with torch.inference_mode():
            model.eval()
            val_accuracy_metric.reset()
   
        #with torch.no_grad():
            for images_v, labels_v in val_loader:
                images_v, labels_v = images.to(device), labels.to(device)

                # Forward pass
                logits_v = model(images_v)
                loss = criterion(logits_v, labels_v)

                predicted_classes_v = logits_v.argmax(dim=1)

                # Update metrics
                val_accuracy_metric.add(predicted_classes_v, labels_v)

            # Compute and print validation accuracy
            val_accuracy = val_accuracy_metric.compute()
            print(f"Validation Accuracy: {val_accuracy}")

    # Save the model
        save_model(model)

    print("Training complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--exp_dir", type=str, default="logs")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--num_epoch", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--batch_size", type=int, default=128)

    # optional: additional model hyperparamters
    # parser.add_argument("--num_layers", type=int, default=3)

    # pass all arguments to train
    train(**vars(parser.parse_args()))