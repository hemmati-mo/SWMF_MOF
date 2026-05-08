#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ================= NumPy <-> legacy aliases shim =================
import numpy as np
if not hasattr(np, "float"):
    np.float  = float   # noqa: F401
    np.int    = int     # noqa: F401
    np.bool   = bool    # noqa: F401
    np.object = object  # noqa: F401
    np.str    = str     # noqa: F401

# ================= Imports =================
import os
import pandas as pd
import joblib
from tqdm import tqdm

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs import ConvertToNumpyArray

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score,
    recall_score, f1_score, matthews_corrcoef, roc_auc_score
)

# ================= Paths/Config =================
splits_dir = "../data/splits/ip_data_cleaned"

model_root = "../Models/ip_data_cleaned/classification/RF_Morgan"
pred_root  = "../predictions/ip_data_cleaned/classification/RF_Morgan"
os.makedirs(model_root, exist_ok=True)
os.makedirs(pred_root, exist_ok=True)

target_col = "Category"
fp_size    = 2048
print(f"[Config] Morgan fingerprint size: {fp_size}")

# ================= Featurization =================
def smiles_to_morgan(s, radius=2, nBits=2048):
    try:
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            return np.zeros((nBits,), dtype=np.float32)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
        arr = np.zeros((nBits,), dtype=np.float32)
        ConvertToNumpyArray(fp, arr)
        return arr
    except Exception:
        return np.zeros((nBits,), dtype=np.float32)

# ================= Metrics (same style as your other scripts) =================
def compute_metrics(y_true, y_pred, y_scores):
    metrics = {
        "accuracy":           accuracy_score(y_true, y_pred),
        "balanced_accuracy":  balanced_accuracy_score(y_true, y_pred),
        "precision_macro":    precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro":       recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro":           f1_score(y_true, y_pred, average="macro", zero_division=0),
        "mcc":                matthews_corrcoef(y_true, y_pred),
        "roc_auc_macro":      np.nan,
        "roc_auc_weighted":   np.nan,
    }

    try:
        scores = np.asarray(y_scores)
        mask = ~np.isnan(scores) if scores.ndim == 1 else ~np.isnan(scores).any(axis=1)
        yt = y_true[mask]
        sc = scores[mask]

        uniq = np.unique(yt)
        if sc.ndim == 1 and len(uniq) == 2:
            auc = roc_auc_score(yt, sc)
            metrics["roc_auc_macro"] = float(auc)
            metrics["roc_auc_weighted"] = float(auc)
        else:
            metrics["roc_auc_macro"] = roc_auc_score(yt, sc, multi_class="ovr", average="macro")
            metrics["roc_auc_weighted"] = roc_auc_score(yt, sc, multi_class="ovr", average="weighted")
    except Exception as e:
        print(f"[Warning] ROC AUC computation failed: {e}")

    return metrics

# ================= Train/Test per fold =================
all_metrics = []

for fold in range(5):
    print(f"\n=== Fold {fold} ===")
    path = os.path.join(splits_dir, f"fold{fold}", "split.csv")
    print(f"[Load] {path}")
    df = pd.read_csv(path)

    # Output dirs for this fold
    fold_model_dir = os.path.join(model_root, f"fold{fold}")
    fold_pred_dir  = os.path.join(pred_root,  f"fold{fold}")
    os.makedirs(fold_model_dir, exist_ok=True)
    os.makedirs(fold_pred_dir, exist_ok=True)

    pred_csv_path = os.path.join(fold_pred_dir, "pred.csv")
    if os.path.exists(pred_csv_path):
        print(f"[SKIP] Fold {fold}: predictions already exist at {pred_csv_path}")
        continue

    print("[Featurize] Computing Morgan fingerprints...")
    X_cpu = np.vstack([
        smiles_to_morgan(s, nBits=fp_size)
        for s in tqdm(df["Canonical SMILES"], desc="Morgan FPs")
    ]).astype(np.float32)
    y_cpu = df[target_col].to_numpy()
    split = df["split"].to_numpy()
    print(f"[Shape] Features: {X_cpu.shape}, Labels: {y_cpu.shape}")

    # Train/val/test masks
    tr, va, te = split == "train", split == "val", split == "test"
    Xtr, ytr = X_cpu[tr], y_cpu[tr]
    Xte, yte = X_cpu[te], y_cpu[te]

    print("[Train] RandomForestClassifier on CPU...")
    rf = RandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        n_jobs=-1,
        class_weight="balanced",
        random_state=42,
    )
    rf.fit(Xtr, ytr)

    print("[Eval] Predicting on test set...")
    y_pred = rf.predict(Xte)

    # Probabilities for ROC-AUC + saving
    if hasattr(rf, "predict_proba"):
        y_probs = rf.predict_proba(Xte)  # shape (n_test, n_classes)
    else:
        # Fallback: use dummy scores from decision_function if it exists
        if hasattr(rf, "decision_function"):
            scores = rf.decision_function(Xte)
            if scores.ndim == 1:
                # binary: convert to 2D probability-like scores via logistic
                from scipy.special import expit
                p1 = expit(scores)
                y_probs = np.vstack([1 - p1, p1]).T
            else:
                # multiclass: softmax
                s = scores - scores.max(axis=1, keepdims=True)
                e = np.exp(s)
                y_probs = e / e.sum(axis=1, keepdims=True)
        else:
            # last resort: no proper scores
            y_probs = np.zeros((len(y_pred), len(np.unique(ytr))), dtype=float)

    metrics = compute_metrics(yte, y_pred, y_probs)
    metrics["fold"] = fold
    all_metrics.append(metrics)
    print("[Metrics]", metrics)

    # Save model
    model_path = os.path.join(fold_model_dir, "rf_morgan.pkl")
    joblib.dump(rf, model_path)
    print(f"[Save] Model → {model_path}")

    # ================= Save test predictions with probabilities =================
    df_test = df[te].copy()
    if df_test.shape[0] != y_probs.shape[0]:
        raise RuntimeError(
            f"Fold {fold}: mismatch between test rows ({df_test.shape[0]}) "
            f"and prob rows ({y_probs.shape[0]})"
        )

    # Map RF class order to 0..K-1 prob_* columns
    # rf.classes_ gives the actual labels order in columns
    classes = rf.classes_
    n_classes = y_probs.shape[1]
    print(f"[Info] RF classes for fold {fold}: {classes.tolist()}")

    # Initialize prob_* columns with NaNs
    for k in range(n_classes):
        df_test[f"prob_{k}"] = np.nan

    # Fill according to RF's class order
    for col_idx, cls in enumerate(classes):
        prob_col_name = f"prob_{int(cls)}"  # align prob_<label> with label value
        df_test[prob_col_name] = y_probs[:, col_idx]

    df_test.to_csv(pred_csv_path, index=False)
    print(f"[Save] Test predictions with probabilities → {pred_csv_path}")

# Save per-fold metrics
metrics_path = os.path.join(model_root, "metrics.csv")
if all_metrics:
    pd.DataFrame(all_metrics).to_csv(metrics_path, index=False)
    print(f"\n[Done] Metrics CSV → {metrics_path}")
else:
    print("\n[Done] No new folds were trained (all had existing predictions).")
