"""
Train LSTM model for price direction prediction (v4).

Architecture: Conv1D -> BatchNorm -> BiLSTM -> Attention -> Dense
Improvements from LSTM_AI_Stock_Predictor:
- StandardScaler normalization (fitted on train only)
- Conv1D before LSTM for local pattern detection
- Warmup period before early stopping (15 epochs)
- Scaler saved with model for inference consistency

Usage:
  python scripts/train_lstm.py --asset BTC
  python scripts/train_lstm.py --asset BTC,ETH,SOL,XRP
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
    data_suffix: str = "",
    epochs: int = 150,
    batch_size: int = 128,
    lr: float = 1e-3,
    patience: int = 20,
    warmup_epochs: int = 15,
    hidden_size: int = 64,
    num_layers: int = 2,
    dropout: float = 0.3,
    exclude_from_date: str | None = None,
    exclude_to_date: str | None = None,
) -> None:
    """Train LSTM for one asset."""
    # Two accepted formats:
    #   (1) New split format: {ASSET}_lstm_X{suffix}.npy  +  {ASSET}_lstm_meta{suffix}.npz
    #       — X is written via memmap-backed streaming at regen time to avoid
    #         OOM on large assets (1M+ sequences).
    #   (2) Legacy single npz: {ASSET}_lstm_sequences{suffix}.npz
    #       — kept for backward compat with old regen outputs.
    split_X = DATA_DIR / f"{asset.upper()}_lstm_X{data_suffix}.npy"
    split_meta = DATA_DIR / f"{asset.upper()}_lstm_meta{data_suffix}.npz"
    legacy_npz = DATA_DIR / f"{asset.upper()}_lstm_sequences{data_suffix}.npz"

    if split_X.exists() and split_meta.exists():
        data_path = split_X  # for logging
        # np.load() without mmap_mode reads the file fully into a new writable
        # ndarray — no need to .copy(). Saves an extra 19 GB temp for BTC-scale X.
        X = np.load(split_X)
        meta = np.load(split_meta, allow_pickle=True)
        y = meta["y"].copy()
        window_starts = meta["window_starts"].copy()
        dms = meta["dms"].copy()
        meta.close()
    elif legacy_npz.exists():
        data_path = legacy_npz
        data = np.load(legacy_npz, allow_pickle=True)
        X = data["X"].copy()  # Copy to own memory (avoid mmap issues on Windows)
        y = data["y"].copy()
        window_starts = data["window_starts"].copy()
        dms = data["dms"].copy()
        data.close()  # Release the npz file handle
    else:
        print(f"No LSTM training data for {asset}. Run generate_aligned_training_data.py first.")
        return

    dm_label = f"dm {min_dm}-{max_dm}" if max_dm is not None else f"dm {min_dm}+"
    excl_label = ""
    if exclude_from_date or exclude_to_date:
        excl_label = f" [excl {exclude_from_date}..{exclude_to_date}]"

    print(f"\n{'='*60}")
    print(f"Training LSTM v4 for {asset} [{dm_label}]{excl_label}")
    print(f"{'='*60}")
    print(f"Loaded {len(y)} sequences from {data_path}")

    # Walk-forward fold exclusion — drop sequences whose window CLOSE
    # (= window_start + 15 min) is inside [exclude_from_date, exclude_to_date].
    # Filter by close, not start, to match pnl_sweep.py (labels belong to close
    # date — Kalshi settles there). Prevents boundary-window leakage.
    if exclude_from_date or exclude_to_date:
        from datetime import datetime, timedelta, timezone
        def _parse(s):
            return datetime.fromisoformat(s).replace(tzinfo=timezone.utc) if s else None
        lo = _parse(exclude_from_date) if exclude_from_date else None
        hi = None
        if exclude_to_date:
            hi_dt = _parse(exclude_to_date)
            # Inclusive end: next day midnight
            hi = hi_dt + timedelta(days=1)

        def _in_excluded(ws_str: str) -> bool:
            try:
                ws = datetime.fromisoformat(ws_str)
                if ws.tzinfo is None:
                    ws = ws.replace(tzinfo=timezone.utc)
            except Exception:
                return False
            we = ws + timedelta(minutes=15)  # window close_time
            if lo is not None and we < lo:
                return False
            if hi is not None and we >= hi:
                return False
            return True

        before = len(y)
        excl_mask = np.array([_in_excluded(str(w)) for w in window_starts])
        keep = ~excl_mask
        X = X[keep]
        y = y[keep]
        window_starts = window_starts[keep]
        dms = dms[keep]
        print(f"Excluded [{exclude_from_date}..{exclude_to_date}] by close_time: {before} -> {len(y)} sequences")

    # Filter by dm range
    mask = dms >= min_dm
    if max_dm is not None:
        mask &= dms <= max_dm

    X = X[mask]
    y = y[mask]
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
    # Cache shape before freeing X — used later for input_size + model metadata.
    seq_len_actual = X.shape[1]
    n_features = X.shape[2]
    # Memory-critical: for large assets (BTC/ETH ~1.25M seqs × 180 × 21 × 4 =
    # ~19 GB), keeping X alive alongside the 3 split arrays doubles peak RAM.
    # X is no longer needed — split copies are independent — so free it now.
    del X

    bullish = int(y.sum())
    bearish = len(y) - bullish
    print(f"Label balance: {bullish} BULLISH ({bullish/len(y)*100:.1f}%) / "
          f"{bearish} BEARISH ({bearish/len(y)*100:.1f}%)")
    print(f"Split: train={len(X_train)} ({len(train_windows)} win) "
          f"val={len(X_val)} ({len(val_windows)} win) "
          f"test={len(X_test)} ({len(test_windows)} win)")

    # StandardScaler normalization — fit on train only
    X_train_flat = X_train.reshape(-1, n_features)
    scaler_mean = X_train_flat.mean(axis=0).astype(np.float32)
    scaler_std = X_train_flat.std(axis=0).astype(np.float32)
    scaler_std[scaler_std == 0] = 1.0  # avoid div by zero

    # In-place scaling + nan_to_num so we don't materialize intermediate arrays
    # (each would be another 13/3/3 GB for BTC's ~1.25M sequence train split).
    X_train -= scaler_mean
    X_train /= scaler_std
    X_val -= scaler_mean
    X_val /= scaler_std
    X_test -= scaler_mean
    X_test /= scaler_std

    # nan_to_num in-place via copy=False
    np.nan_to_num(X_train, copy=False, nan=0.0, posinf=1e6, neginf=-1e6)
    np.nan_to_num(X_val, copy=False, nan=0.0, posinf=1e6, neginf=-1e6)
    np.nan_to_num(X_test, copy=False, nan=0.0, posinf=1e6, neginf=-1e6)

    print(f"Normalized features (scaler fitted on {len(X_train_flat)} train samples)")

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
    input_size = n_features
    model = PriceLSTM(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        mode="classification",
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,}")

    # Training setup
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10, min_lr=1e-6,
    )

    # Training loop with warmup
    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_epoch = 0
    best_state = None
    epochs_no_improve = 0

    t0 = time.time()

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        train_loss = 0.0
        train_correct = 0
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
            train_correct += ((pred > 0.5).float() == yb).sum().item()
            train_total += len(xb)

        # Validate
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                loss = criterion(pred, yb)
                val_loss += loss.item() * len(xb)
                val_correct += ((pred > 0.5).float() == yb).sum().item()
                val_total += len(xb)

        train_loss /= train_total
        val_loss /= val_total
        train_acc = train_correct / train_total
        val_acc = val_correct / val_total

        scheduler.step(val_loss)

        if epoch <= 5 or epoch % 10 == 0 or val_loss < best_val_loss:
            current_lr = optimizer.param_groups[0]["lr"]
            print(f"  Epoch {epoch:3d}: train_loss={train_loss:.4f} train_acc={train_acc:.3f} "
                  f"val_loss={val_loss:.4f} val_acc={val_acc:.3f} lr={current_lr:.1e}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            # Only start counting patience after warmup
            if epoch >= warmup_epochs:
                epochs_no_improve = 0
        else:
            if epoch >= warmup_epochs:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"  Early stopping at epoch {epoch} (no improvement for {patience} epochs after warmup)")
                    break

    train_time = time.time() - t0
    print(f"\nBest epoch: {best_epoch} (val_loss={best_val_loss:.4f}, val_acc={best_val_acc:.3f})")
    print(f"Training time: {train_time:.1f}s")

    # Load best model
    model.load_state_dict(best_state)
    model.eval()

    # Test evaluation — batched to avoid GPU OOM on large datasets
    y_test_t = torch.tensor(y_test, dtype=torch.float32).to(device)
    eval_batch = 2048
    test_pred_parts = []
    with torch.no_grad():
        for i in range(0, len(X_test), eval_batch):
            xb = torch.tensor(X_test[i:i + eval_batch], dtype=torch.float32).to(device)
            test_pred_parts.append(model(xb))
        test_pred = torch.cat(test_pred_parts)
        test_labels = (test_pred > 0.5).float()
        test_acc = (test_labels == y_test_t).float().mean().item()
        test_correct = int((test_labels == y_test_t).sum().item())

        tp = int(((test_labels == 1) & (y_test_t == 1)).sum().item())
        fp = int(((test_labels == 1) & (y_test_t == 0)).sum().item())
        fn = int(((test_labels == 0) & (y_test_t == 1)).sum().item())
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    print(f"\n=== {asset} LSTM v4 Test Results ===")
    print(f"Accuracy:  {test_acc:.4f}  ({test_correct}/{len(y_test)})")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")

    # Calibration
    test_pred_np = test_pred.cpu().numpy()
    y_test_np = y_test
    bins = [(0, 0.3), (0.3, 0.4), (0.4, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 1.01)]
    print("\nCalibration (predicted prob vs actual):")
    for lo, hi in bins:
        mask = (test_pred_np >= lo) & (test_pred_np < hi)
        n = int(mask.sum())
        if n > 0:
            pred_avg = float(test_pred_np[mask].mean())
            actual = float(y_test_np[mask].mean())
            label = f"[{lo*100:.0f}%-{hi*100:.0f}%)" if hi < 1 else f"[{lo*100:.0f}%-100%]"
            print(f"  {label}: pred_avg={pred_avg:.3f} actual={actual:.3f} n={n}")

    # Accuracy by dm
    print("\nAccuracy by decision minute:")
    test_dms = dms[test_mask]
    for dm_val in sorted(set(test_dms)):
        dm_mask = test_dms == dm_val
        dm_correct = int((test_labels.cpu().numpy()[dm_mask] == y_test_np[dm_mask]).sum())
        dm_total = int(dm_mask.sum())
        if dm_total > 0:
            print(f"  dm {dm_val}: {dm_correct/dm_total*100:.1f}%  (n={dm_total})")

    # High-confidence accuracy
    print("\nHigh-confidence subset accuracy:")
    for conf_thresh in [0.60, 0.65, 0.70, 0.75]:
        conf_mask = (test_pred_np >= conf_thresh) | (test_pred_np <= 1 - conf_thresh)
        n = int(conf_mask.sum())
        if n > 0:
            conf_correct = int((test_labels.cpu().numpy()[conf_mask] == y_test_np[conf_mask]).sum())
            pct = n / len(y_test) * 100
            print(f"  conf >= {conf_thresh*100:.0f}%: {conf_correct/n*100:.1f}%  (n={n}, {pct:.0f}% of samples)")

    # Save model + scaler
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / f"{asset.upper()}{model_suffix}_lstm.pt"
    save_model(model.cpu(), str(model_path), metadata={
        "asset": asset,
        "mode": "classification",
        "dm_range": dm_label,
        "test_accuracy": float(test_acc),
        "test_precision": precision,
        "test_recall": recall,
        "best_epoch": best_epoch,
        "train_samples": len(X_train),
        "val_samples": len(X_val),
        "test_samples": len(X_test),
        "seq_len": seq_len_actual,
        "num_features": n_features,
        "scaler_mean": scaler_mean.tolist(),
        "scaler_std": scaler_std.tolist(),
    })
    print(f"\nModel saved: {model_path}")
    sys_mod = __import__("sys")
    sys_mod.stdout.flush()
    sys_mod.stderr.flush()

    # Force immediate process exit AFTER the model save. PyTorch's CUDA
    # tensor destructors (running during `del model`, normal Python GC, or
    # interpreter shutdown) crash the C runtime on Windows with
    # STATUS_STACK_BUFFER_OVERRUN (0xC0000409 = exit code 3221226505) even
    # AFTER the model file is on disk. os._exit(0) skips ALL Python shutdown
    # paths so the destructors never run. Model save is the only thing that
    # mattered — the orchestrator wants exit 0 to mean "model file exists".
    __import__("os")._exit(0)


def main():
    parser = argparse.ArgumentParser(description="Train LSTM v4 price direction model")
    parser.add_argument("--asset", required=True, help="Asset(s), comma-separated")
    parser.add_argument("--min-dm", type=int, default=2, help="Minimum decision minute")
    parser.add_argument("--max-dm", type=int, default=None, help="Maximum decision minute")
    parser.add_argument("--model-suffix", type=str, default="", help="Model filename suffix")
    parser.add_argument("--epochs", type=int, default=150, help="Max training epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--hidden-size", type=int, default=64, help="Hidden size")
    parser.add_argument("--num-layers", type=int, default=2, help="LSTM layers")
    parser.add_argument("--data-suffix", type=str, default="",
                        help="Suffix for training data file (e.g. '_weekday')")
    parser.add_argument("--exclude-from-date", type=str, default=None,
                        help="YYYY-MM-DD: drop sequences whose window CLOSE is on/after this "
                             "date. Filters by close_time to match pnl_sweep.py.")
    parser.add_argument("--exclude-to-date", type=str, default=None,
                        help="YYYY-MM-DD: drop sequences whose window CLOSE is on/before this "
                             "date (inclusive). Filters by close_time to match pnl_sweep.py.")
    args = parser.parse_args()

    assets = [a.strip().upper() for a in args.asset.split(",")]
    for asset in assets:
        train_asset(
            asset,
            min_dm=args.min_dm,
            max_dm=args.max_dm,
            model_suffix=args.model_suffix,
            data_suffix=args.data_suffix,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            exclude_from_date=args.exclude_from_date,
            exclude_to_date=args.exclude_to_date,
        )

    # Bypass Python's normal shutdown — PyTorch + CUDA finalization on Windows
    # crashes with STATUS_STACK_BUFFER_OVERRUN (0xC0000409 / exit 3221226505)
    # AFTER all models are saved. os._exit(0) skips atexit handlers and the
    # C-level CUDA destructors that trigger the crash. Model files are on
    # disk before this point — process exit is just paperwork.
    import os, sys
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
