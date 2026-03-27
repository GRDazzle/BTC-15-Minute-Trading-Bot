"""
LSTM model definition for price direction prediction (v4).

Architecture: Conv1D -> BatchNorm -> LSTM -> Attention -> Dense
Inspired by LSTM_AI_Stock_Predictor: Conv1D captures local patterns,
LSTM handles sequence dependencies, attention pools the output.

v4 changes:
- Added Conv1D layer before LSTM (local pattern detection)
- BatchNorm after Conv1D (training stability)
- StandardScaler normalization saved with model (feature scaling)
- Warmup-aware early stopping
"""
import torch
import torch.nn as nn

from ml.lstm_features import LSTM_NUM_FEATURES, LSTM_SEQ_LEN


class PriceLSTM(nn.Module):
    """Conv1D + Bidirectional LSTM for binary price direction prediction."""

    def __init__(
        self,
        input_size: int = LSTM_NUM_FEATURES,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
        num_heads: int = 4,
        mode: str = "classification",
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.mode = mode

        # Conv1D for local temporal pattern detection
        # Input: (batch, seq_len, features) -> transpose to (batch, features, seq_len)
        self.conv1d = nn.Conv1d(input_size, hidden_size, kernel_size=3, padding=1)
        self.batch_norm = nn.BatchNorm1d(hidden_size)
        self.conv_dropout = nn.Dropout(dropout)

        # LSTM on top of conv features
        self.lstm = nn.LSTM(
            hidden_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
        )

        lstm_output_size = hidden_size * 2  # bidirectional

        # Attention pooling
        self.attn_score = nn.Linear(lstm_output_size, 1)

        # Output head
        self.head = nn.Sequential(
            nn.LayerNorm(lstm_output_size),
            nn.Linear(lstm_output_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (batch, seq_len, features) tensor — already normalized by scaler.

        Returns:
            (batch,) tensor of P(BULLISH) probabilities or predicted returns.
        """
        # Conv1D expects (batch, channels, seq_len)
        x = x.transpose(1, 2)  # (batch, features, seq_len)
        x = self.conv1d(x)     # (batch, hidden, seq_len)
        x = self.batch_norm(x)
        x = torch.relu(x)
        x = self.conv_dropout(x)
        x = x.transpose(1, 2)  # (batch, seq_len, hidden)

        # LSTM
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden*2)

        # Attention pooling
        scores = self.attn_score(lstm_out)  # (batch, seq_len, 1)
        weights = torch.softmax(scores, dim=1)
        context = (lstm_out * weights).sum(dim=1)  # (batch, hidden*2)

        out = self.head(context).squeeze(-1)

        if self.mode == "classification":
            out = torch.sigmoid(out)

        return out


def save_model(model: PriceLSTM, path: str, metadata: dict | None = None) -> None:
    """Save model state dict, config, and metadata."""
    payload = {
        "state_dict": model.state_dict(),
        "config": {
            "input_size": model.input_size,
            "hidden_size": model.hidden_size,
            "num_layers": model.num_layers,
            "mode": model.mode,
        },
    }
    if metadata:
        payload["metadata"] = metadata
    torch.save(payload, path)


def load_model(path: str, device: str = "cpu") -> tuple[PriceLSTM, dict]:
    """Load model from .pt file."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    model = PriceLSTM(
        input_size=config["input_size"],
        hidden_size=config["hidden_size"],
        num_layers=config["num_layers"],
        mode=config.get("mode", "classification"),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    metadata = checkpoint.get("metadata", {})
    return model, metadata
