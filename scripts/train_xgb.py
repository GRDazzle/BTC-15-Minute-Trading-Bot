"""
Train per-asset XGBoost meta-model.

Walk-forward split: train on first 70%, validate on next 17%, test on last 13%.
Saves model to models/{ASSET}_xgb.json and feature importance chart.

Usage:
  python scripts/train_xgb.py --asset BTC
  python scripts/train_xgb.py --asset BTC,ETH,SOL,XRP
  python scripts/train_xgb.py --asset BTC --tune
  python scripts/train_xgb.py --asset BTC --min-dm 2   # exclude noisy dm 0-1
  python scripts/train_xgb.py --asset XRP --exclude-hours 1,9,10,17
"""
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_score,
    recall_score,
)

from ml.features import FEATURE_NAMES

DATA_DIR = PROJECT_ROOT / "ml" / "training_data"
MODEL_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "output" / "ml"


def load_data(asset: str, min_dm: int = 0, exclude_hours: list[int] | None = None) -> pd.DataFrame:
    """Load training data CSV for an asset."""
    path = DATA_DIR / f"{asset.upper()}_features.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Training data not found: {path}\n"
            f"Run: python scripts/generate_training_data.py --asset {asset}"
        )
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} rows from {path}")

    # Filter out noisy early decision minutes
    if min_dm > 0:
        before = len(df)
        df = df[df["minute_in_window"] >= min_dm].reset_index(drop=True)
        print(f"Filtered dm < {min_dm}: {before} -> {len(df)} rows")

    # Filter out low-accuracy hours
    if exclude_hours:
        before = len(df)
        if "hour_utc" in df.columns:
            df = df[~df["hour_utc"].astype(int).isin(exclude_hours)].reset_index(drop=True)
        else:
            # Recover hour from window_start if hour_utc column not present
            hours = pd.to_datetime(df["window_start"]).dt.hour
            df = df[~hours.isin(exclude_hours)].reset_index(drop=True)
        print(f"Filtered hours {exclude_hours}: {before} -> {len(df)} rows")

    # Drop zero-variance features (dead columns)
    dead = [c for c in FEATURE_NAMES if c in df.columns and df[c].std() == 0]
    if dead:
        print(f"Warning: zero-variance features (will be kept but useless): {dead}")

    # Label balance
    n_bull = (df["label"] == 1).sum()
    n_bear = (df["label"] == 0).sum()
    print(f"Label balance: {n_bull} BULLISH ({n_bull/len(df)*100:.1f}%) / "
          f"{n_bear} BEARISH ({n_bear/len(df)*100:.1f}%)")

    return df


def deduplicate_windows(df: pd.DataFrame, strategy: str = "last") -> pd.DataFrame:
    """Reduce correlated rows: keep only one row per window.

    strategy:
      "last"   - keep the last checkpoint per window (most info, like live first-signal)
      "middle" - keep the median dm checkpoint
      "random" - random sample one per window
    """
    before = len(df)
    if strategy == "last":
        df = df.groupby("window_start").tail(1).reset_index(drop=True)
    elif strategy == "middle":
        df = df.groupby("window_start").apply(
            lambda g: g.iloc[len(g) // 2]
        ).reset_index(drop=True)
    elif strategy == "random":
        df = df.groupby("window_start").sample(n=1, random_state=42).reset_index(drop=True)
    else:
        return df

    print(f"Window dedup ({strategy}): {before} -> {len(df)} rows")
    return df


def walk_forward_split(df: pd.DataFrame):
    """Split data chronologically: 70% train, 17% val, 13% test.

    Splits by unique windows (not rows) to prevent leakage.
    """
    df = df.sort_values("window_start").reset_index(drop=True)

    # Split by unique windows to avoid same-window rows in different splits
    unique_windows = df["window_start"].unique()
    n_win = len(unique_windows)
    train_win_end = int(n_win * 0.70)
    val_win_end = int(n_win * 0.87)

    train_windows = set(unique_windows[:train_win_end])
    val_windows = set(unique_windows[train_win_end:val_win_end])
    test_windows = set(unique_windows[val_win_end:])

    train = df[df["window_start"].isin(train_windows)].reset_index(drop=True)
    val = df[df["window_start"].isin(val_windows)].reset_index(drop=True)
    test = df[df["window_start"].isin(test_windows)].reset_index(drop=True)

    print(f"Split: train={len(train)} ({len(train_windows)} win) "
          f"val={len(val)} ({len(val_windows)} win) "
          f"test={len(test)} ({len(test_windows)} win)")

    for name, split in [("Train", train), ("Val", val), ("Test", test)]:
        ws = split["window_start"]
        print(f"  {name}: {ws.iloc[0]} -> {ws.iloc[-1]}")

    return train, val, test


def train_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    tune: bool = False,
) -> xgb.XGBClassifier:
    """Train XGBoost classifier with early stopping and optional tuning."""
    X_train = train_df[FEATURE_NAMES].values
    y_train = train_df["label"].values
    X_val = val_df[FEATURE_NAMES].values
    y_val = val_df["label"].values

    # Class balancing: scale_pos_weight = n_negative / n_positive
    n_pos = (y_train == 1).sum()
    n_neg = (y_train == 0).sum()
    spw = n_neg / n_pos if n_pos > 0 else 1.0
    print(f"scale_pos_weight: {spw:.3f}")

    # Early stopping rounds
    early_stop = 20

    if tune:
        best_acc = 0.0
        best_params = {}
        param_grid = {
            "max_depth": [3, 4, 5],
            "learning_rate": [0.01, 0.05, 0.1],
            "n_estimators": [200, 400, 600],
            "min_child_weight": [10, 20, 50],
        }
        total = 1
        for v in param_grid.values():
            total *= len(v)
        print(f"Tuning: {total} parameter combinations")
        tried = 0

        for md in param_grid["max_depth"]:
            for lr in param_grid["learning_rate"]:
                for ne in param_grid["n_estimators"]:
                    for mcw in param_grid["min_child_weight"]:
                        tried += 1
                        model = xgb.XGBClassifier(
                            objective="binary:logistic",
                            n_estimators=ne,
                            max_depth=md,
                            learning_rate=lr,
                            subsample=0.7,
                            colsample_bytree=0.7,
                            min_child_weight=mcw,
                            scale_pos_weight=spw,
                            eval_metric="logloss",
                            early_stopping_rounds=early_stop,
                            verbosity=0,
                        )
                        model.fit(
                            X_train, y_train,
                            eval_set=[(X_val, y_val)],
                            verbose=False,
                        )
                        y_pred = model.predict(X_val)
                        acc = accuracy_score(y_val, y_pred)
                        if acc > best_acc:
                            best_acc = acc
                            best_params = {
                                "max_depth": md,
                                "learning_rate": lr,
                                "n_estimators": ne,
                                "min_child_weight": mcw,
                            }
                        if tried % 10 == 0:
                            print(f"  [{tried}/{total}] current best: acc={best_acc:.4f}")

        print(f"\nBest params: {best_params} (val acc={best_acc:.4f})")

        # Retrain with best params
        model = xgb.XGBClassifier(
            objective="binary:logistic",
            subsample=0.7,
            colsample_bytree=0.7,
            scale_pos_weight=spw,
            eval_metric="logloss",
            early_stopping_rounds=early_stop,
            verbosity=0,
            **best_params,
        )
    else:
        model = xgb.XGBClassifier(
            objective="binary:logistic",
            n_estimators=400,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.7,
            colsample_bytree=0.7,
            min_child_weight=20,
            scale_pos_weight=spw,
            eval_metric="logloss",
            early_stopping_rounds=early_stop,
            verbosity=0,
        )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    # Report best iteration (early stopping)
    if hasattr(model, "best_iteration"):
        print(f"Best iteration: {model.best_iteration} / {model.n_estimators}")

    y_val_pred = model.predict(X_val)
    val_acc = accuracy_score(y_val, y_val_pred)
    print(f"Validation accuracy: {val_acc:.4f}")

    return model


def evaluate_model(model: xgb.XGBClassifier, test_df: pd.DataFrame, asset: str) -> None:
    """Evaluate model on test set and print report."""
    X_test = test_df[FEATURE_NAMES].values
    y_test = test_df["label"].values

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)

    print(f"\n=== {asset} Test Results ===")
    print(f"Accuracy:  {acc:.4f}  ({sum(y_pred == y_test)}/{len(y_test)})")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print()
    print(classification_report(
        y_test, y_pred,
        target_names=["BEARISH", "BULLISH"],
        zero_division=0,
    ))

    # Calibration
    print("Calibration (predicted prob vs actual):")
    bins = [(0.0, 0.3), (0.3, 0.4), (0.4, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 1.01)]
    for lo, hi in bins:
        mask = (y_prob >= lo) & (y_prob < hi)
        if mask.sum() == 0:
            continue
        actual_rate = y_test[mask].mean()
        count = mask.sum()
        label = f"[{lo:.0%}-{hi:.0%})" if hi < 1.0 else f"[{lo:.0%}-100%]"
        print(f"  {label}: pred_avg={y_prob[mask].mean():.3f} actual={actual_rate:.3f} n={count}")
    print()

    # Accuracy by hour
    if "hour_utc" in test_df.columns:
        print("Accuracy by hour (UTC):")
        hours = test_df["hour_utc"].values
        for h in sorted(set(hours.astype(int))):
            mask = hours.astype(int) == h
            if mask.sum() < 10:
                continue
            h_acc = accuracy_score(y_test[mask], y_pred[mask])
            print(f"  {h:02d}:00  {h_acc:.1%}  (n={mask.sum()})")
        print()

    # Accuracy by decision minute
    if "minute_in_window" in test_df.columns:
        print("Accuracy by decision minute:")
        dms = test_df["minute_in_window"].values
        for dm in sorted(set(dms.astype(int))):
            mask = dms.astype(int) == dm
            if mask.sum() < 10:
                continue
            dm_acc = accuracy_score(y_test[mask], y_pred[mask])
            print(f"  dm {dm}: {dm_acc:.1%}  (n={mask.sum()})")
        print()

    # High-confidence subset accuracy
    print("High-confidence subset accuracy:")
    for thresh in [0.60, 0.65, 0.70, 0.75]:
        high_mask = (y_prob >= thresh) | (y_prob <= 1.0 - thresh)
        if high_mask.sum() < 10:
            continue
        hc_acc = accuracy_score(y_test[high_mask], y_pred[high_mask])
        pct = high_mask.sum() / len(y_test) * 100
        print(f"  conf >= {thresh:.0%}: {hc_acc:.1%}  (n={high_mask.sum()}, {pct:.0f}% of windows)")
    print()


def save_feature_importance(model: xgb.XGBClassifier, asset: str) -> None:
    """Save feature importance chart as PNG."""
    importance = model.feature_importances_
    pairs = sorted(zip(FEATURE_NAMES, importance), key=lambda x: x[1], reverse=True)

    # Always print text importance
    print(f"\n{asset} Feature Importance:")
    for name, imp in pairs:
        bar = "#" * int(imp * 200)
        print(f"  {name:25s} {imp:.4f}  {bar}")
    print()

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        indices = np.argsort(importance)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.barh(range(len(indices)), importance[indices])
        ax.set_yticks(range(len(indices)))
        ax.set_yticklabels([FEATURE_NAMES[i] for i in indices])
        ax.set_xlabel("Feature Importance (gain)")
        ax.set_title(f"{asset} XGBoost Feature Importance")
        plt.tight_layout()

        out_path = OUTPUT_DIR / f"{asset}_feature_importance.png"
        fig.savefig(str(out_path), dpi=150)
        plt.close(fig)
        print(f"Feature importance chart: {out_path}")
    except ImportError:
        pass


def train_asset(
    asset: str,
    tune: bool = False,
    min_dm: int = 0,
    dedup: str = "none",
    exclude_hours: list[int] | None = None,
) -> None:
    """Full training pipeline for one asset."""
    print(f"\n{'='*60}")
    print(f"Training XGBoost for {asset}")
    print(f"{'='*60}")

    df = load_data(asset, min_dm=min_dm, exclude_hours=exclude_hours)

    # Optional window deduplication
    if dedup != "none":
        df = deduplicate_windows(df, strategy=dedup)

    train_df, val_df, test_df = walk_forward_split(df)

    model = train_model(train_df, val_df, tune=tune)

    evaluate_model(model, test_df, asset)

    # Save model
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / f"{asset.upper()}_xgb.json"
    model.save_model(str(model_path))
    print(f"Model saved: {model_path}")

    save_feature_importance(model, asset)


def main():
    parser = argparse.ArgumentParser(description="Train per-asset XGBoost meta-model")
    parser.add_argument(
        "--asset", required=True,
        help="Asset(s) to train, comma-separated (e.g. BTC or BTC,ETH,SOL,XRP)",
    )
    parser.add_argument(
        "--tune", action="store_true",
        help="Run hyperparameter grid search",
    )
    parser.add_argument(
        "--min-dm", type=int, default=2,
        help="Exclude decision minutes below this (default: 2, excludes noisy dm 0-1)",
    )
    parser.add_argument(
        "--dedup", choices=["none", "last", "middle", "random"], default="none",
        help="Window dedup strategy: keep one row per window (default: none = all rows)",
    )
    parser.add_argument(
        "--exclude-hours", type=str, default=None,
        help="Comma-separated UTC hours to exclude (e.g. '1,9,10,17'). "
             "Removes rows from hours with low historical accuracy.",
    )
    args = parser.parse_args()

    # Parse exclude hours
    exclude_hours = None
    if args.exclude_hours:
        exclude_hours = [int(h.strip()) for h in args.exclude_hours.split(",")]

    assets = [a.strip().upper() for a in args.asset.split(",")]
    for asset in assets:
        train_asset(
            asset,
            tune=args.tune,
            min_dm=args.min_dm,
            dedup=args.dedup,
            exclude_hours=exclude_hours,
        )


if __name__ == "__main__":
    main()
