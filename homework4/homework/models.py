from pathlib import Path
from torch.optim.lr_scheduler import StepLR

import torch
import torch.nn as nn
import torch.nn.functional as F

HOMEWORK_DIR = Path(__file__).resolve().parent
INPUT_MEAN = [0.2788, 0.2657, 0.2629]
INPUT_STD = [0.2064, 0.1944, 0.2252]


class MLPPlanner(nn.Module):
    def __init__(
        self,
        n_track: int = 10,
        n_waypoints: int = 3,
    ):
        """
        Args:
            n_track (int): number of points in each side of the track
            n_waypoints (int): number of waypoints to predict
        """
        super().__init__()

        self.n_track = n_track
        self.n_waypoints = n_waypoints
        self.input_dim = 2 * 2 * n_track
        self.output_dim = 2 * n_waypoints
        self.hidden_dim = 128
        self.fc1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.output_dim)
    
    def forward(    
        self,
        track_left: torch.Tensor,
        track_right: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Predicts waypoints from the left and right boundaries of the track.

        During test time, your model will be called with
        model(track_left=..., track_right=...), so keep the function signature as is.

        Args:
            track_left (torch.Tensor): shape (b, n_track, 2)
            track_right (torch.Tensor): shape (b, n_track, 2)

        Returns:
            torch.Tensor: future waypoints with shape (b, n_waypoints, 2)
        """
        # Concatenate the left and right track points
        x = torch.cat((track_left, track_right), dim=1)
        # Flatten the input
        x = x.view(x.size(0), -1)
        # Pass through the first layer
        x = self.fc1(x)
        x = self.relu(x)
        # Pass through the second layer
        x = self.fc2(x)
        #x = self.relu(x)
        # Reshape the output to match the expected output shape
        x = x.view(x.size(0), self.n_waypoints, 2)
        # Return the output
        return x
    
class TransformerPlanner(nn.Module):
    def __init__(
        self,
        n_track: int = 10,
        n_waypoints: int = 3,
        d_model: int = 64,
        nhead: int = 4,  # Reduced number of attention heads
        num_layers: int = 4,  # Reduced number of layers
        dim_feedforward: int = 256,  # Reduced feedforward dimension
        dropout: float = 0.1,  # Added dropout
    ):
        super().__init__()

        self.n_track = n_track
        self.n_waypoints = n_waypoints
        self.d_model = d_model

        self.query_embed = nn.Embedding(n_waypoints, d_model)
        self.input_embed = nn.Linear(2, d_model)
        self.positional_encoding = self._generate_sinusoidal_encoding(n_track * 2, d_model)  # Sinusoidal encoding
        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model, nhead=nhead, batch_first=True, dim_feedforward=dim_feedforward, dropout=dropout
        )
        self.transformer_decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(d_model, 2)

    def _generate_sinusoidal_encoding(self, length: int, d_model: int) -> torch.Tensor:
        """
        Generate sinusoidal positional encoding.
        """
        position = torch.arange(length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(torch.log(torch.tensor(10000.0)) / d_model))
        encoding = torch.zeros(length, d_model)
        encoding[:, 0::2] = torch.sin(position * div_term)
        encoding[:, 1::2] = torch.cos(position * div_term)
        return encoding.unsqueeze(0)  # Shape: (1, length, d_model)

    def forward(
        self,
        track_left: torch.Tensor,
        track_right: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Predicts waypoints from the left and right boundaries of the track.

        During test time, your model will be called with
        model(track_left=..., track_right=...), so keep the function signature as is.

        Args:
            track_left (torch.Tensor): shape (b, n_track, 2)
            track_right (torch.Tensor): shape (b, n_track, 2)

        Returns:
            torch.Tensor: future waypoints with shape (b, n_waypoints, 2)
        """
        b = track_left.size(0)  # Batch size

        # Concatenate left and right track boundaries
        x = torch.cat((track_left, track_right), dim=1)  # Shape: (b, n_track * 2, 2)

        # Embed input and add positional encoding
        memory = self.input_embed(x) + self.positional_encoding.to(x.device)  # Shape: (b, n_track * 2, d_model)

        # Query embeddings for waypoints
        queries = self.query_embed.weight.unsqueeze(0).repeat(b, 1, 1)  # Shape: (b, n_waypoints, d_model)

        # Pass through the transformer decoder
        decoded = self.transformer_decoder(
            tgt=queries,  # Target (queries for waypoints)
            memory=memory,  # Encoded input (track boundaries)
        )  # Shape: (b, n_waypoints, d_model)

        # Project to (x, y) coordinates
        waypoints = self.output_proj(decoded)  # Shape: (b, n_waypoints, 2)

        return waypoints
        


class CNNPlanner(nn.Module):
    def __init__(self, n_waypoints: int = 3):
        """
        CNNPlanner model to predict waypoints from an input image.

        Args:
            n_waypoints (int): Number of waypoints to predict.
        """
        super().__init__()
        self.n_waypoints = n_waypoints
        in_channels = 3  # Input image channels (RGB)

        # Down-sampling layers
        self.down1 = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16)
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32)
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64)
        )

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.ReLU(),
            nn.BatchNorm2d(64)
        )

        # Up-sampling layers
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32)
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16)
        )
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(16, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16)
        )

        # Align channels for skip connections
        self.align_channels_down2 = nn.Conv2d(32, 32, kernel_size=1)
        self.align_channels_down1 = nn.Conv2d(16, 16, kernel_size=1)

        # Waypoint regression head
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))  # (B, C, 1, 1)
        self.waypoint_head = nn.Sequential(
            nn.Flatten(),  # (B, C)
            nn.Linear(16, 64),
            nn.ReLU(),
            nn.Linear(64, n_waypoints * 2)  # output (B, n_waypoints * 2)
        )


    def forward(self, image: torch.FloatTensor) -> torch.Tensor:
        """
        Forward pass of the CNNPlanner.

        Args:
            image (torch.FloatTensor): shape (b, 3, h, w) and vals in [0, 1]

        Returns:
            torch.Tensor: Predicted waypoints of shape (B, n_waypoints, 2).
        """
        # Down-sampling
        down1 = self.down1(image)
        down2 = self.down2(down1)
        down3 = self.down3(down2)

        # Bottleneck
        bottleneck = self.bottleneck(down3)

        # Up-sampling with skip connections
        up1 = self.up1(bottleneck) + self.align_channels_down2(down2)
        up2 = self.up2(up1) + self.align_channels_down1(down1)
        up3 = self.up3(up2)  # (B, 16, H, W)

        # Waypoint prediction
        pooled = self.global_pool(up3)  # (B, 16, 1, 1)
        waypoints = self.waypoint_head(pooled)  # (B, n_waypoints * 2)
        waypoints = waypoints.view(-1, self.n_waypoints, 2)  # (B, n_waypoints, 2)

        return waypoints
    
MODEL_FACTORY = {
    "mlp_planner": MLPPlanner,
    "transformer_planner": TransformerPlanner,
    "cnn_planner": CNNPlanner,
}


def load_model(
    model_name: str,
    with_weights: bool = False,
    **model_kwargs,
) -> torch.nn.Module:
    """
    Called by the grader to load a pre-trained model by name
    """
    m = MODEL_FACTORY[model_name](**model_kwargs)

    if with_weights:
        model_path = HOMEWORK_DIR / f"{model_name}.th"
        assert model_path.exists(), f"{model_path.name} not found"

        try:
            m.load_state_dict(torch.load(model_path, map_location="cpu"))
        except RuntimeError as e:
            raise AssertionError(
                f"Failed to load {model_path.name}, make sure the default model arguments are set correctly"
            ) from e

    # limit model sizes since they will be zipped and submitted
    model_size_mb = calculate_model_size_mb(m)

    if model_size_mb > 20:
        raise AssertionError(f"{model_name} is too large: {model_size_mb:.2f} MB")

    return m


def save_model(model: torch.nn.Module) -> str:
    """
    Use this function to save your model in train.py
    """
    model_name = None

    for n, m in MODEL_FACTORY.items():
        if type(model) is m:
            model_name = n

    if model_name is None:
        raise ValueError(f"Model type '{str(type(model))}' not supported")

    output_path = HOMEWORK_DIR / f"{model_name}.th"
    torch.save(model.state_dict(), output_path)

    return output_path


def calculate_model_size_mb(model: torch.nn.Module) -> float:
    """
    Naive way to estimate model size
    """
    return sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024



