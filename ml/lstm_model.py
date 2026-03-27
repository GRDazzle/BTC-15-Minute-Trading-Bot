"""
LSTM model definition for price prediction (v3).

Architecture: Bidirectional LSTM with attention pooling.
v3: Supports both classification (P(BULLISH)) and regression (price return).
"""
import torch
import torch.nn as nn

from ml.lstm_features import LSTM_NUM_FEATURES, LSTM_SEQ_LEN


class PriceLSTM(nn.Module):
    """Bidirectional LSTM for price prediction (classification or regression)."""

    def __init__(
        self,
        input_size: int = LSTM_NUM_FEATURES,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
        num_heads: int = 4,
        mode: str = "regression",  # "classification" or "regression"
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.mode = mode

        # Input projection
        self.input_norm = nn.LayerNorm(input_size)
        self.input_proj = nn.Linear(input_size, hidden_size)

        self.lstm = nn.LSTM(
            hidden_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
        )

        lstm_output_size = hidden_size * 2  # bidirectional

        # Simple attention pooling
        self.attn_score = nn.Sequential(
            nn.Linear(lstm_output_size, 1),
        )

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
            x: (batch, seq_len, features) tensor.

        Returns:
            (batch,) tensor.
            - regression: predicted price return (unbounded)
            - classification: P(BULLISH) in [0, 1]
        """
        # Normalize and project input
        x = self.input_norm(x)
        x = self.input_proj(x)

        # LSTM
        lstm_out, _ = self.lstm(x)

        # Attention pooling
        scores = self.attn_score(lstm_out)
        weights = torch.softmax(scores, dim=1)
        context = (lstm_out * weights).sum(dim=1)

        out = self.head(context).squeeze(-1)

        if self.mode == "classification":
            out = torch.sigmoid(out)

        return out


def save_model(model: PriceLSTM, path: str, metadata: dict | None = None) -> None:
    """Save model state dict and metadata."""
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
