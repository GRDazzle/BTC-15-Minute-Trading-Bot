"""
Train LSTM model for price prediction (classification or regression).

Walk-forward split: 70% train / 17% val / 13% test (same as XGBoost).
Uses per-asset .npz sequences from generate_lstm_training_data.py.

Usage:
  python scripts/train_lstm.py --asset BTC --mode regression
  python scripts/train_lstm.py --asset BTC,ETH,SOL,XRP --mode regression
  python scripts/train_lstm.py --asset BTC --mode classification
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ml.lstm_model import PriceLSTM, save_model

DATA_DIR = PROJECT_ROOT / "ml" / "training_data"
MODEL_DIR = PROJECT_ROOT / "models"


def train_asset(
    asset: str,
    min_dm: int = 2,
    max_dm: int | None = None,
    model_suffix: str = "",
    mode: str = "regression",
    epochs: int = 100,
    batch_size: int = 512,
    lr: float = 5e-4,
    patience: int = 15,
    hidden_size: int = 64,
    num_layers: int = 2,
    dropout: float = 0.3,
) -> None:
    """Train LSTM for one asset."""
    data_path = DATA_DIR / f"{asset.upper()}_lstm_sequences.npz"
    if not data_path.exists():
        print(f"No LSTM training data for {asset}. Run generate_lstm_training_data.py first.")
        return

    data = np.load(data_path, allow_pickle=True)
    X = data["X"]
    y_cls = data["y"]  # binary labels
    window_starts = data["window_starts"]
    dms = data["dms"]

    # Load regression targets if available
    if "returns" in data:
        y_reg = data["returns"]
    else:
        if mode == "regression":
            print(f"No 'returns' in training data. Regenerate with updated script.")
            return
        y_reg = None

    y = y_reg if mode == "regression" else y_cls

    dm_label = f"dm {min_dm}-{max_dm}" if max_dm is not None else f"dm {min_dm}+"

    print(f"\n{'='*60}")
    print(f"Training LSTM for {asset} [{dm_label}] mode={mode}")
    print(f"{'='*60}")
    print(f"Loaded {len(y)} sequences from {data_path}")

    # Filter by dm range
    mask = dms >= min_dm
    if max_dm is not None:
        mask &= dms <= max_dm

    X = X[mask]
    y = y[mask]
    y_cls_filtered = y_cls[mask]
    window_starts = window_starts[mask]
    dms = dms[mask]
    print(f"After dm filter: {len(y)} sequences")

    # Walk-forward split by unique window starts
    unique_windows = sorted(set(window_starts))
    n_windows = len(unique_windows)
    train_end = int(n_windows * 0.70)
    val_end = int(n_windows * 0.87)

    train_windows = set(unique_windows[:train_end])
    val_windows = set(unique_windows[train_end:val_end])
    test_windows = set(unique_windows[val_end:])

    train_mask = np.array([w in train_windows for w in window_starts])
    val_mask = np.array([w in val_windows for w in window_starts])
    test_mask = np.array([w in test_windows for w in window_starts])

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    y_cls_test = y_cls_filtered[test_mask]

    if mode == "regression":
        print(f"Return stats: mean={y.mean():.6f} std={y.std():.6f} "
              f"min={y.min():.6f} max={y.max():.6f}")
    else:
        bullish = int(y_cls_filtered.sum())
        bearish = len(y_cls_filtered) - bullish
        print(f"Label balance: {bullish} BULLISH ({bullish/len(y)*100:.1f}%) / "
              f"{bearish} BEARISH ({bearish/len(y)*100:.1f}%)")

    print(f"Split: train={len(X_train)} ({len(train_windows)} win) "
          f"val={len(X_val)} ({len(val_windows)} win) "
          f"test={len(X_test)} ({len(test_windows)} win)")

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Create datasets
    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )
    val_ds = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    # Model
    input_size = X.shape[2]
    model = PriceLSTM(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        mode=mode,
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,}")

    # Loss function
    if mode == "regression":
        criterion = nn.MSELoss()  # Let the model learn the full return distribution
    else:
        criterion = nn.BCELoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6,
    )

    # Training loop
    best_val_loss = float("inf")
    best_epoch = 0
    best_state = None
    epochs_no_improve = 0

    t0 = time.time()

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        train_loss = 0.0
        train_total = 0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item() * len(xb)
            train_total += len(xb)

        # Validate
        model.eval()
        val_loss = 0.0
        val_total = 0

        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                loss = criterion(pred, yb)
                val_loss += loss.item() * len(xb)
                val_total += len(xb)

        train_loss /= train_total
        val_loss /= val_total

        scheduler.step(epoch)

        if epoch <= 5 or epoch % 5 == 0 or val_loss < best_val_loss:
            current_lr = optimizer.param_groups[0]["lr"]
            print(f"  Epoch {epoch:3d}: train_loss={train_loss:.6f} "
                  f"val_loss={val_loss:.6f} lr={current_lr:.1e}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"  Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
                break

    train_time = time.time() - t0
    print(f"\nBest epoch: {best_epoch} (val_loss={best_val_loss:.6f})")
    print(f"Training time: {train_time:.1f}s")

    # Load best model
    model.load_state_dict(best_state)
    model.eval()

    # Test evaluation
    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)

    with torch.no_grad():
        test_pred = model(X_test_t).cpu().numpy()

    print(f"\n=== {asset} LSTM Test Results (mode={mode}) ===")

    if mode == "regression":
        # Direction accuracy: does sign of predicted return match actual?
        pred_direction = (test_pred > 0).astype(int)
        actual_direction = y_cls_test.astype(int)
        dir_acc = (pred_direction == actual_direction).mean()
        dir_correct = int((pred_direction == actual_direction).sum())
        print(f"Direction accuracy: {dir_acc:.4f}  ({dir_correct}/{len(y_test)})")

        # Return prediction stats
        print(f"Predicted returns: mean={test_pred.mean():.6f} std={test_pred.std():.6f}")
        print(f"Actual returns:    mean={y_test.mean():.6f} std={y_test.std():.6f}")

        # Correlation
        corr = np.corrcoef(test_pred, y_test)[0, 1]
        print(f"Correlation (pred vs actual): {corr:.4f}")

        # Direction accuracy by magnitude
        print("\nDirection accuracy by predicted magnitude:")
        abs_pred = np.abs(test_pred)
        for thresh_pct in [0.0, 0.01, 0.02, 0.05, 0.10, 0.20]:
            thresh = thresh_pct / 100  # Convert to decimal
            mag_mask = abs_pred >= thresh
            n = int(mag_mask.sum())
            if n > 0:
                mag_acc = (pred_direction[mag_mask] == actual_direction[mag_mask]).mean()
                pct = n / len(y_test) * 100
                print(f"  |pred| >= {thresh_pct:.2f}%: {mag_acc*100:.1f}%  (n={n}, {pct:.0f}% of samples)")

        # Accuracy by dm
        print("\nDirection accuracy by decision minute:")
        test_dms = dms[test_mask]
        for dm_val in sorted(set(test_dms)):
            dm_mask = test_dms == dm_val
            dm_correct = int((pred_direction[dm_mask] == actual_direction[dm_mask]).sum())
            dm_total = int(dm_mask.sum())
            if dm_total > 0:
                print(f"  dm {dm_val}: {dm_correct/dm_total*100:.1f}%  (n={dm_total})")

        test_acc = dir_acc
    else:
        # Classification evaluation
        test_labels = (test_pred > 0.5).astype(int)
        test_acc = (test_labels == y_cls_test).mean()
        test_correct = int((test_labels == y_cls_test).sum())
        print(f"Accuracy:  {test_acc:.4f}  ({test_correct}/{len(y_test)})")

    # Save model
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / f"{asset.upper()}{model_suffix}_lstm.pt"
    save_model(model.cpu(), str(model_path), metadata={
        "asset": asset,
        "mode": mode,
        "dm_range": dm_label,
        "test_accuracy": float(test_acc),
        "best_epoch": best_epoch,
        "train_samples": len(X_train),
        "val_samples": len(X_val),
        "test_samples": len(X_test),
        "seq_len": X.shape[1],
        "num_features": X.shape[2],
    })
    print(f"\nModel saved: {model_path}")


def main():
    parser = argparse.ArgumentParser(description="Train LSTM price prediction model")
    parser.add_argument("--asset", required=True, help="Asset(s), comma-separated")
    parser.add_argument("--mode", default="regression", choices=["regression", "classification"])
    parser.add_argument("--min-dm", type=int, default=2, help="Minimum decision minute")
    parser.add_argument("--max-dm", type=int, default=None, help="Maximum decision minute")
    parser.add_argument("--model-suffix", type=str, default="", help="Model filename suffix")
    parser.add_argument("--epochs", type=int, default=100, help="Max training epochs")
    parser.add_argument("--batch-size", type=int, default=512, help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--hidden-size", type=int, default=64, help="LSTM hidden size")
    parser.add_argument("--num-layers", type=int, default=2, help="LSTM layers")
    args = parser.parse_args()

    assets = [a.strip().upper() for a in args.asset.split(",")]
    for asset in assets:
        train_asset(
            asset,
            min_dm=args.min_dm,
            max_dm=args.max_dm,
            model_suffix=args.model_suffix,
            mode=args.mode,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
        )


if __name__ == "__main__":
    main()
