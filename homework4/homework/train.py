import warnings
import numpy as np
from torch import get_device
from torch.utils.data import DataLoader
import torch

from grader.datasets import road_dataset
from grader.metrics import PlannerMetric
from homework.models import MLPPlanner, TransformerPlanner, load_model, save_model

DATA_SPLIT = "drive_data/val"

def get_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    else:
        warnings.warn(
            "No hardware acceleration found. Using CPU for grading.",
            category=RuntimeWarning,
            stacklevel=1,
        )
        device = torch.device("cpu")
    print(f"Using device: {device}")
    return device

def train(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int,
    learning_rate: float,
):
    """
    Training pipeline for MLPPlanner.
    """
    # Define optimizer and loss function
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = torch.nn.SmoothL1Loss()  # Loss function for regression
    device = get_device()  # Ensure device is retrieved once
    model.to(device)  # Move model to the appropriate device
    # Initialize PlannerMetric
    train_planner_metric = PlannerMetric()
    val_planner_metric = PlannerMetric()
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        
        # Training loop
        for batch in train_loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            train_track_left = batch["track_left"].to(device)
            train_track_right = batch["track_right"].to(device)    
            train_waypoints = batch["waypoints"].to(device)
            train_waypoints_mask = batch["waypoints_mask"].to(device)

            # Forward pass
            optimizer.zero_grad()   # Zero the gradients    
            outputs = model(train_track_left, train_track_right)
            loss = criterion(outputs, train_waypoints)
            loss.backward()
            optimizer.step()
            loss = loss * train_waypoints_mask[..., None]
            train_loss = loss.sum() 

            # Compute metrics
            train_planner_metric.add(outputs, train_waypoints, train_waypoints_mask)
        
        # Validation loop
        model.eval()
        val_loss = 0.0
        loss = 0.0
        with torch.no_grad():
             for batch in val_loader:
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                val_track_left = batch["track_left"].to(device)
                val_track_right = batch["track_right"].to(device)
                val_waypoints = batch["waypoints"].to(device)
                val_waypoints_mask = batch["waypoints_mask"].to(device)

                outputs = model(val_track_left, val_track_right)
                loss = criterion(outputs, val_waypoints)
                loss = loss * val_waypoints_mask[..., None]
                val_loss = loss.sum()

                # Compute metrics
                val_planner_metric.add(outputs, val_waypoints, val_waypoints_mask)


        val_loss /= len(val_loader)
        train_loss /= len(train_loader)
        # Print training and validation loss at 10th epoch
        if epoch % 5 == 0:
            print(f"Epoch [{epoch}/{num_epochs}], Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(val_loader):.4f}")
            print(f"Train L1 Error: {train_planner_metric.compute()['l1_error']:.4f}")
            print(f"Validation L1 Error: {val_planner_metric.compute()['l1_error']:.4f}")
            print(f"Train Longitudinal Error: {train_planner_metric.compute()['longitudinal_error']:.4f}")
            print(f"Validation Longitudinal Error: {val_planner_metric.compute()['longitudinal_error']:.4f}")
            print(f"Train Lateral Error: {train_planner_metric.compute()['lateral_error']:.4f}")
            print(f"Validation Lateral Error: {val_planner_metric.compute()['lateral_error']:.4f}")
        # Save the model every 10 epochs
        if epoch % 3 == 0:
            save_model(model)
    

 
    
# Example Usage
if __name__ == "__main__":
    # Hyperparameters
    MODEL_NAME = "MLPPlanner"
    n_track = 10
    n_waypoints = 3
    hidden_dim = 128
    batch_size = 2048
    num_epochs = 10
    learning_rate = 0.0001

    # Create dataset and dataloaders
    train_loader = road_dataset.load_data(
            DATA_SPLIT,
            num_workers=1,
            return_dataloader=True,
            batch_size=64,
            shuffle=False,
        )
    
    val_loader = road_dataset.load_data(
            DATA_SPLIT,
            num_workers=1,
            return_dataloader=True,
            batch_size=64,
            shuffle=True,
        )

    # Initialize the model
    #model = load_model(MODEL_NAME, with_weights=True, device=get_device(),n_track=n_track, n_waypoints=n_waypoints)

    model = MLPPlanner(n_track=n_track, n_waypoints=n_waypoints)
    model.device = get_device()  # Use get_device to set the device
    model.to(model.device)

    # Train the model
    train(model, train_loader, val_loader, num_epochs, learning_rate)
