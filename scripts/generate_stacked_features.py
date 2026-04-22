"""Generate out-of-fold LSTM predictions for XGB stacking.

For each asset:
  1. Load LSTM training data (npz) and XGB training data (csv)
  2. Align by window_start (same tick replay, same windows)
  3. K-fold time-split: train LSTM on K-1 folds, predict on held-out fold
  4. Append lstm_p column to XGB training CSV

The result is an XGB CSV with an additional 'lstm_p' column containing
out-of-fold LSTM predictions. This avoids data leakage since each
LSTM prediction is made by a model that never saw that training row.

Usage:
    python scripts/generate_stacked_features.py --asset BTC,ETH,SOL,XRP
    python scripts/generate_stacked_features.py --asset SOL --n-folds 5
"""
import argparse
import csv
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from loguru import logger

TRAINING_DIR = PROJECT_ROOT / "ml" / "training_data"
MODELS_DIR = PROJECT_ROOT / "models"


def load_lstm_data(asset: str):
    """Load LSTM npz data. Returns X, y, window_starts, dms."""
    path = TRAINING_DIR / f"{asset.upper()}_lstm_sequences.npz"
    if not path.exists():
        raise FileNotFoundError(f"LSTM data not found: {path}")

    data = np.load(path, allow_pickle=True)
    X = data["X"].copy()
    y = data["y"].copy()
    window_starts = data["window_starts"].copy()
    dms = data["dms"].copy()
    data.close()

    logger.info("LSTM data for {}: {} samples, {} unique windows",
                asset, len(X), len(set(window_starts)))
    return X, y, window_starts, dms


def build_lstm_model(n_features: int, seq_len: int = 180):
    """Build a fresh LSTM model (same architecture as train_lstm.py)."""
    from ml.lstm_model import PriceLSTM
    model = PriceLSTM(
        input_size=n_features,
        hidden_size=128,
        num_layers=2,
        dropout=0.3,
    )
    return model


def train_lstm_fold(X_train, y_train, X_val, y_val, n_features, epochs=30, batch_size=128):
    """Train LSTM on one fold, return the trained model."""
    from ml.lstm_model import PriceLSTM

    # Standardize using training data statistics
    mean = X_train.reshape(-1, n_features).mean(axis=0)
    std = X_train.reshape(-1, n_features).std(axis=0)
    std[std == 0] = 1.0

    X_train_norm = (X_train - mean) / std
    X_val_norm = (X_val - mean) / std
    X_train_norm = np.nan_to_num(X_train_norm, nan=0.0, posinf=1e6, neginf=-1e6)
    X_val_norm = np.nan_to_num(X_val_norm, nan=0.0, posinf=1e6, neginf=-1e6)

    model = PriceLSTM(
        input_size=n_features,
        hidden_size=128,
        num_layers=2,
        dropout=0.3,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    criterion = torch.nn.BCELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=3, factor=0.5,
    )

    X_t = torch.tensor(X_train_norm, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.float32)
    # Keep validation data on CPU, batch to GPU during eval
    X_v_cpu = torch.tensor(X_val_norm, dtype=torch.float32)
    y_v_cpu = torch.tensor(y_val, dtype=torch.float32)

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        indices = torch.randperm(len(X_t))
        total_loss = 0
        n_batches = 0

        for start in range(0, len(X_t), batch_size):
            batch_idx = indices[start:start + batch_size]
            xb = X_t[batch_idx].to(device)
            yb = y_t[batch_idx].to(device)

            pred = model(xb).view(-1)
            loss = criterion(pred, yb)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        # Validation — batch to avoid OOM
        model.eval()
        val_preds = []
        with torch.no_grad():
            for vstart in range(0, len(X_v_cpu), batch_size):
                vb = X_v_cpu[vstart:vstart + batch_size].to(device)
                vp = model(vb).view(-1)
                val_preds.append(vp.cpu())
        val_pred = torch.cat(val_preds)
        val_loss = criterion(val_pred, y_v_cpu).item()

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 7:
                break

    if best_state:
        model.load_state_dict(best_state)
    model.eval()

    return model, mean, std


def predict_lstm(model, X, mean, std, n_features, batch_size=256):
    """Run LSTM inference, return P(BULLISH) for each sample."""
    device = next(model.parameters()).device

    X_norm = (X - mean) / std
    X_norm = np.nan_to_num(X_norm, nan=0.0, posinf=1e6, neginf=-1e6)

    X_t = torch.tensor(X_norm, dtype=torch.float32)
    predictions = []

    model.eval()
    with torch.no_grad():
        for start in range(0, len(X_t), batch_size):
            xb = X_t[start:start + batch_size].to(device)
            pred = model(xb).view(-1)
            predictions.extend(pred.cpu().numpy().tolist())

    return np.array(predictions)


def generate_oof_predictions(asset: str, n_folds: int = 5):
    """Generate out-of-fold LSTM predictions for XGB stacking.

    Returns dict mapping window_start -> list of (dm_approx, lstm_p) pairs.
    """
    X_lstm, y_lstm, ws_lstm, dms_lstm = load_lstm_data(asset)
    n_features = X_lstm.shape[2]

    # Sort by window_start for time-based splitting
    unique_dates = sorted(set(ws[:10] for ws in ws_lstm))
    n_dates = len(unique_dates)
    fold_size = n_dates // n_folds

    logger.info("Generating OOF predictions: {} dates, {} folds, fold_size={}",
                n_dates, n_folds, fold_size)

    # Store predictions indexed by (window_start, dm)
    oof_preds = {}

    for fold_idx in range(n_folds):
        test_start = fold_idx * fold_size
        test_end = (fold_idx + 1) * fold_size if fold_idx < n_folds - 1 else n_dates
        test_dates = set(unique_dates[test_start:test_end])
        train_dates = set(unique_dates) - test_dates

        train_mask = np.array([ws[:10] in train_dates for ws in ws_lstm])
        test_mask = np.array([ws[:10] in test_dates for ws in ws_lstm])

        X_train = X_lstm[train_mask]
        y_train = y_lstm[train_mask]
        X_test = X_lstm[test_mask]
        y_test = y_lstm[test_mask]
        ws_test = ws_lstm[test_mask]
        dms_test = dms_lstm[test_mask]

        logger.info("Fold {}/{}: train={}, test={} ({} -> {})",
                     fold_idx + 1, n_folds, len(X_train), len(X_test),
                     sorted(test_dates)[0], sorted(test_dates)[-1])

        if len(X_train) < 100 or len(X_test) < 10:
            logger.warning("Fold {} skipped: insufficient data", fold_idx + 1)
            continue

        # Use 90/10 split within training data for early stopping
        split = int(len(X_train) * 0.9)
        model, mean, std = train_lstm_fold(
            X_train[:split], y_train[:split],
            X_train[split:], y_train[split:],
            n_features,
        )

        # Predict on held-out test fold
        preds = predict_lstm(model, X_test, mean, std, n_features)

        for i in range(len(preds)):
            ws = str(ws_test[i])
            dm = int(dms_test[i])
            oof_preds[(ws, dm)] = float(preds[i])

        # Cleanup
        del model
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    logger.info("OOF predictions generated: {} total", len(oof_preds))
    return oof_preds


def add_lstm_to_xgb_csv(asset: str, oof_preds: dict):
    """Add lstm_p column to the XGB training CSV.

    Reads the existing CSV, matches rows by (window_start, approximate dm),
    adds lstm_p, and writes to a new file with '_stacked' suffix.
    """
    src = TRAINING_DIR / f"{asset.upper()}_features.csv"
    dst = TRAINING_DIR / f"{asset.upper()}_features_stacked.csv"

    matched = 0
    unmatched = 0
    rows_out = []

    with open(src, "r") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames) + ["lstm_p"]

        for row in reader:
            ws = row.get("window_start", "")
            # minute_in_window approximates dm
            dm = int(float(row.get("minute_in_window", 0)))

            lstm_p = oof_preds.get((ws, dm))
            if lstm_p is not None:
                row["lstm_p"] = f"{lstm_p:.6f}"
                matched += 1
            else:
                row["lstm_p"] = "0.5"  # neutral default
                unmatched += 1
            rows_out.append(row)

    with open(dst, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)

    total = matched + unmatched
    logger.info("Stacked CSV written: {} ({} matched, {} unmatched / {} total = {:.1f}% match rate)",
                dst, matched, unmatched, total,
                matched / total * 100 if total else 0)
    return dst


def main():
    parser = argparse.ArgumentParser(description="Generate stacked LSTM features for XGB")
    parser.add_argument("--asset", required=True, help="Assets (comma-separated)")
    parser.add_argument("--n-folds", type=int, default=5, help="OOF folds for LSTM")
    args = parser.parse_args()

    logger.remove()
    logger.add(sys.stderr, level="INFO",
               format="<green>{time:HH:mm:ss}</green> | <level>{level:<7}</level> | {message}")

    assets = [a.strip().upper() for a in args.asset.split(",")]

    for asset in assets:
        print(f"\n{'='*70}")
        print(f"  STACKING: {asset}")
        print(f"{'='*70}")

        # Step 1: Generate OOF LSTM predictions
        oof_preds = generate_oof_predictions(asset, n_folds=args.n_folds)

        # Step 2: Add lstm_p to XGB CSV
        stacked_path = add_lstm_to_xgb_csv(asset, oof_preds)

        print(f"  Output: {stacked_path}")
        print()


if __name__ == "__main__":
    main()
