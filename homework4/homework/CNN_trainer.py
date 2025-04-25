import argparse
import torch
from torch.nn import SmoothL1Loss,HuberLoss
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR

from grader.metrics import PlannerMetric
from homework.datasets.road_dataset import load_data
from homework.models import CNNPlanner, save_model

def train(
    exp_dir: str = "logs",
    model_name: str = "cnn_planner",
    num_epoch: int = 50,
    lr: float = 1e-4,
    batch_size: int = 16,
    seed: int = 2024,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    **kwargs,
):
    """
    Train the CNNPlanner model.

    Args:
        exp_dir (str): Directory to save logs and models.
        model_name (str): Name of the model to train.
        num_epoch (int): Number of training epochs.
        lr (float): Learning rate for the optimizer.
        batch_size (int): Batch size for training.
        seed (int): Random seed for reproducibility.
        device (str): Device to use for training ("cuda" or "cpu").
    """
    # Set random seed for reproducibility
    torch.manual_seed(seed)

    # Load training and validation datasets
    train_loader = load_data("drive_data/train", transform_pipeline="aug", shuffle=True, batch_size=batch_size, num_workers=2)
    val_loader = load_data("drive_data/val", transform_pipeline="aug",shuffle=False, batch_size=batch_size, num_workers=2)

    # Initialize the model, loss function, and optimizer
    model = CNNPlanner(n_waypoints=kwargs.get("n_waypoints", 3)).to(device)
    criterion = SmoothL1Loss(reduction="none")  # Loss function for regression
    optimizer = AdamW(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=2, gamma=0.1)  # Reduce LR by 10x every 2 epochs

    # Initialize PlannerMetric
    train_planner_metric = PlannerMetric()
    val_planner_metric = PlannerMetric()
    
    # Training loop
    for epoch in range(num_epoch):
        print(f"Epoch {epoch + 1}/{num_epoch}")

        # Training phase
        model.train()
        train_loss = 0.0
        for batch_idx, batch in enumerate(train_loader):
            # Load batch data
            images = batch["image"].to(device)  # Input images
            waypoints = batch["waypoints"].to(device)  # Target waypoints
            waypoints_mask = batch["waypoints_mask"].to(device)  # Mask for valid waypoints

            # Forward pass
            optimizer.zero_grad()
            outputs = model(images)  # Predicted waypoints

            # Apply mask to ignore invalid waypoints
            mask = waypoints_mask.unsqueeze(-1)  # Shape: (B, n_waypoints, 1)

            outputs_masked = outputs * mask
            waypoints_masked = waypoints * mask

            # Use SmoothL1Loss to calculate the loss
            loss = criterion(outputs_masked, waypoints_masked)

            loss = loss.sum() / waypoints_mask.sum()  # Average loss over valid waypoints

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            #train_planner_metric.add(outputs, waypoints, waypoints_masked)


        # Step the learning rate scheduler
        scheduler.step()

        # Print average training loss for the epoch
        print(f"Epoch {epoch + 1}/{num_epoch}, Train Loss: {train_loss / len(train_loader):.4f}")
 
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_lat_error = 0.0
        val_lon_error = 0.0
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                images = batch["image"].to(device)
                waypoints = batch["waypoints"].to(device)
                waypoints_mask = batch["waypoints_mask"].to(device)

                outputs = model(images)
                mask = waypoints_mask.unsqueeze(-1)
                
                outputs_masked = outputs * mask
                waypoints_masked = waypoints * mask

                # Use SmoothL1Loss to calculate the loss
                loss = criterion(outputs_masked, waypoints_masked)
                loss = loss.sum() / waypoints_mask.sum()  # Average loss over valid waypoints

                val_loss += loss.item()
                #val_planner_metric.add(outputs, waypoints, waypoints_mask)


        # Print average validation loss and errors for the epoch
       
        
        print(f"Epoch [{epoch}/{num_epoch}], Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(val_loader):.4f}")
        # print(f"Train L1 Error: {train_planner_metric.compute()['l1_error']:.4f}")
        # print(f"Validation L1 Error: {val_planner_metric.compute()['l1_error']:.4f}")
        # print(f"Train Longitudinal Error: {train_planner_metric.compute()['longitudinal_error']:.4f}")
        # print(f"Validation Longitudinal Error: {val_planner_metric.compute()['longitudinal_error']:.4f}")
        # print(f"Train Lateral Error: {train_planner_metric.compute()['lateral_error']:.4f}")
        # print(f"Validation Lateral Error: {val_planner_metric.compute()['lateral_error']:.4f}")

        # Save the model after every epoch
        save_model(model)

    print("Training complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--exp_dir", type=str, default="logs")
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--num_epoch", type=int, default=50)
    parser.add_argument("--lr", type=float, default=((1e-2)))
    parser.add_argument("--seed", type=int, default=2024)

    # optional: additional model hyperparamters
    # parser.add_argument("--num_layers", type=int, default=3)

    # pass all arguments to train
    train(**vars(parser.parse_args()))