import argparse
import torch
from torch.utils.data import DataLoader
from .datasets.road_dataset import load_data
from .models import load_model, save_model
from .metrics import ConfusionMatrix  # For mIoU calculation

def train(
    exp_dir: str = "logs",
    model_name: str = "Detector",
    num_epoch: int = 50,
    lr: float = 1e-3,
    batch_size: int = 16,
    seed: int = 2024,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    **kwargs,
):
    # Set random seed for reproducibility
    torch.manual_seed(seed)

    # Load training and validation datasets
    train_loader = load_data("drive_data/train", shuffle=True, batch_size=batch_size, num_workers=2)
    val_loader = load_data("drive_data/val", shuffle=False, batch_size=batch_size, num_workers=2)

    # Initialize model, loss functions, optimizer, and metrics
    model = load_model(model_name, **kwargs).to(device)
    criterion = torch.nn.CrossEntropyLoss()  # For segmentation
    depth_loss_fn = torch.nn.L1Loss()  # For depth regression
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Metric for segmentation (mIoU)
    confusion_matrix = ConfusionMatrix(num_classes=3)

    # Training loop
    for epoch in range(num_epoch):
        print(f"Epoch {epoch + 1}/{num_epoch}")

        # Training phase
        model.train()
        train_segmentation_loss = 0.0
        train_depth_loss = 0.0
        for batch in train_loader:
            images = batch["image"].to(device)  # Input images
            segmentation_labels = batch["track"].to(device)  # Segmentation labels
            depth_labels = batch["depth"].to(device)  # Depth labels

            # Forward pass
            logits, depth = model(images)

            # Compute losses
            segmentation_loss = criterion(logits, segmentation_labels)
            depth_loss = depth_loss_fn(depth, depth_labels)
            loss = segmentation_loss + depth_loss

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Accumulate losses
            train_segmentation_loss += segmentation_loss.item()
            train_depth_loss += depth_loss.item()

        print(f"Train Segmentation Loss: {train_segmentation_loss / len(train_loader):.4f}")
        print(f"Train Depth Loss: {train_depth_loss / len(train_loader):.4f}")

        # Validation phase
        model.eval()
        val_segmentation_loss = 0.0
        val_depth_loss = 0.0
        confusion_matrix.reset()
        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device)
                segmentation_labels = batch["track"].to(device)
                depth_labels = batch["depth"].to(device)

                # Forward pass
                logits, depth = model(images)

                # Compute losses
                segmentation_loss = criterion(logits, segmentation_labels)
                depth_loss = depth_loss_fn(depth, depth_labels)

                # Accumulate losses
                val_segmentation_loss += segmentation_loss.item()
                val_depth_loss += depth_loss.item()

                # Update confusion matrix for mIoU calculation
                predictions = logits.argmax(dim=1)  # Convert logits to class predictions
                confusion_matrix.add(predictions, segmentation_labels)

        # Compute mIoU
        metrics = confusion_matrix.compute()
        miou = metrics["iou"]  # Extract mean IoU
        accuracy = metrics["accuracy"]  # Extract accuracy (optional)

        print(f"Val Segmentation Loss: {val_segmentation_loss / len(val_loader):.4f}")
        print(f"Val Depth Loss: {val_depth_loss / len(val_loader):.4f}")
        print(f"Val mIoU: {miou:.4f}")

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

    # optional: additional model hyperparamters
    # parser.add_argument("--num_layers", type=int, default=3)

    # pass all arguments to train
    train(**vars(parser.parse_args()))