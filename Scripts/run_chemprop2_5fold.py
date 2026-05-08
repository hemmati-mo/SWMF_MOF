#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import subprocess
from pathlib import Path

import pandas as pd

# ===================== Paths & settings =====================
DATA_ROOT = Path("../data/splits/ip_data_cleaned")

MODEL_ROOT = Path("../Models/ip_data_cleaned/classification/Chemprop")
PRED_ROOT  = Path("../predictions/ip_data_cleaned/classification/chemprop")

SMILES_COL = "Canonical SMILES"
TARGET_COL = "Category"
SPLIT_COL  = "split"

FOLDS = [0, 1, 2, 3, 4]

EPOCHS = 50
METRIC = "multiclass-mcc"

# If chemprop isn’t on PATH, set CHEMPROP_BIN to its full path, else leave as "chemprop"
CHEMPROP_BIN = "chemprop"


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def run_cmd(cmd, cwd=None):
    print("\n[CMD]", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=cwd)


def main():
    for fold in FOLDS:
        print(f"\n================ Fold {fold} ================")

        split_csv = DATA_ROOT / f"fold{fold}/split.csv"
        if not split_csv.exists():
            raise FileNotFoundError(f"Missing split.csv for fold {fold}: {split_csv}")

        # Output dirs
        model_dir = MODEL_ROOT / f"fold_{fold}"
        pred_dir  = PRED_ROOT  / f"fold_{fold}"
        ensure_dir(model_dir)
        ensure_dir(pred_dir)

        pred_csv = pred_dir / "pred.csv"

        # Skip if predictions already exist
        if pred_csv.exists():
            print(f"[SKIP] Fold {fold}: {pred_csv} already exists.")
            continue

        # ===================== 1) Train Chemprop model for this fold =====================
        print(f"[INFO] Training Chemprop model for fold {fold}...")

        train_cmd = [
            CHEMPROP_BIN, "train",
            "--data-path", str(split_csv),
            "--task-type", "multiclass",
            "--splits-column", SPLIT_COL,
            "--epochs", str(EPOCHS),
            "--output-dir", str(model_dir),
            "--smiles-columns", SMILES_COL,
            "--target-columns", TARGET_COL,
            "--metrics", METRIC,
        ]
        run_cmd(train_cmd)

        # ===================== 2) Build test-only CSV for this fold =====================
        print(f"[INFO] Preparing test subset for fold {fold}...")
        df_all = pd.read_csv(split_csv)
        df_test = df_all[df_all[SPLIT_COL] == "test"].copy()
        if df_test.empty:
            raise RuntimeError(f"Fold {fold}: no test rows found with {SPLIT_COL} == 'test'.")

        test_csv = pred_dir / "test_only.csv"
        df_test.to_csv(test_csv, index=False)
        print(f"[INFO] Saved test-only CSV: {test_csv}")

        # ===================== 3) Run chemprop predict on test-only CSV =====================
        raw_pred_csv = pred_dir / "raw_predictions.csv"
        print(f"[INFO] Running chemprop predict for fold {fold}...")

        predict_cmd = [
            CHEMPROP_BIN, "predict",
            "--test-path", str(test_csv),
            "--smiles-columns", SMILES_COL,
            "--model-paths", str(model_dir),
            "--preds-path", str(raw_pred_csv),
        ]
        run_cmd(predict_cmd)

        if not raw_pred_csv.exists():
            raise FileNotFoundError(f"Chemprop did not produce predictions file: {raw_pred_csv}")

        # ===================== 4) Merge probabilities back into original test rows =====================
        print(f"[INFO] Merging Chemprop probabilities into test dataframe for fold {fold}...")

        df_pred = pd.read_csv(raw_pred_csv)
        if len(df_pred) != len(df_test):
            raise RuntimeError(
                f"Row count mismatch in fold {fold}: "
                f"test rows={len(df_test)}, pred rows={len(df_pred)}"
            )

        # Identify prediction columns produced by Chemprop:
        # take any columns that are NOT in the original test dataframe
        extra_cols = [c for c in df_pred.columns if c not in df_test.columns]
        if not extra_cols:
            raise RuntimeError(
                f"Could not find prediction columns in {raw_pred_csv}. "
                f"Columns: {list(df_pred.columns)}"
            )

        # Sort them to get a stable order, then rename as prob_0, prob_1, ...
        extra_cols_sorted = sorted(extra_cols)
        num_classes = len(extra_cols_sorted)
        print(f"[INFO] Detected {num_classes} prediction columns: {extra_cols_sorted}")

        for idx, col in enumerate(extra_cols_sorted):
            prob_col = f"prob_{idx}"
            df_test[prob_col] = df_pred[col].values

        # ===================== 5) Save final pred.csv with original columns + prob_* =====================
        df_test.to_csv(pred_csv, index=False)
        print(f"[DONE] Fold {fold}: saved merged predictions to {pred_csv}")

    print("\nAll folds processed.")


if __name__ == "__main__":
    main()
