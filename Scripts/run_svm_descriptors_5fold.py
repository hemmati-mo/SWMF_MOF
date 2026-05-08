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

# ================= CPU-side imports first (avoid CUDA in forks) ==============
import os, json
from pathlib import Path

import pandas as pd
import joblib
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score,
    recall_score, f1_score, matthews_corrcoef, roc_auc_score
)

# ================= Paths/Config =================
# Splits (same as ChemBERTa / Chemprop)
splits_dir = "../data/splits/ip_data_cleaned"

# Where models are stored
model_root = "../Models/ip_data_cleaned/classification/SVM_descriptor"
os.makedirs(model_root, exist_ok=True)

# Where predictions will be stored (as requested)
pred_root = "../predictions/ip_data_cleaned/classification/SVM_descriptor"
os.makedirs(pred_root, exist_ok=True)

target_col = "Category"

# Parallel settings (cap to avoid process thrash)
DEFAULT_PROCS = min(max(cpu_count() - 1, 1), 64)
N_PROCS    = int(os.environ.get("RDESC_NPROCS", DEFAULT_PROCS))
CHUNK_SIZE = int(os.environ.get("RDESC_CHUNK", 256))

# Silence RDKit warnings
RDLogger.DisableLog("rdApp.warning")

# ================= RDKit descriptors =================
desc_names = [d[0] for d in Descriptors._descList]
calc = MoleculeDescriptors.MolecularDescriptorCalculator(desc_names)
print(f"[Config] Number of RDKit descriptors: {len(desc_names)}")
print(f"[Config] Parallel featurization with {N_PROCS} processes, chunk={CHUNK_SIZE}")

# Save descriptor names & order at the root for reproducibility
descriptor_names_path = os.path.join(model_root, "descriptor_names.json")
with open(descriptor_names_path, "w") as f:
    json.dump({"descriptor_names": desc_names}, f, indent=2)
print(f"[Save] Descriptor names → {descriptor_names_path}")


def smiles_to_desc_row(s: str):
    """Return descriptor vector; NaNs on failure; finite floats only."""
    try:
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            return [np.nan] * len(desc_names)
        vals = list(calc.CalcDescriptors(mol))
        out = []
        for v in vals:
            try:
                vf = float(v)
                out.append(vf if np.isfinite(vf) else np.nan)
            except Exception:
                out.append(np.nan)
        return out
    except Exception:
        return [np.nan] * len(desc_names)


def featurize_descriptors_parallel_ordered(smiles_list, n_procs=N_PROCS, chunk_size=CHUNK_SIZE):
    """Ordered parallel featurization (imap preserves order)."""
    rows = []
    with Pool(processes=n_procs) as pool:
        it = pool.imap(smiles_to_desc_row, smiles_list, chunksize=chunk_size)
        for row in tqdm(it, total=len(smiles_list), desc="RDKit descriptors (parallel)"):
            rows.append(row)
    return pd.DataFrame(rows, dtype=np.float64)  # float64 for safe cleaning


# ================= Cleaning helpers =================
def clean_descriptor_df_with_mask(X_df: pd.DataFrame, winsorize: bool = True, p: float = 0.999):
    """
    - Replace ±inf with NaN
    - (Optional) Winsorize per-column at p-quantile to bound extreme values
    - Build a keep-mask (row-wise finiteness) then drop rows accordingly
    - Return cleaned_df_float64 and the boolean mask (len == original rows)
    """
    X = X_df.replace([np.inf, -np.inf], np.nan)

    if winsorize:
        q = X.quantile(p, axis=0, numeric_only=True)
        X = X.clip(lower=None, upper=q, axis=1)

    keep_mask = ~X.isna().any(axis=1).to_numpy()
    dropped = int((~keep_mask).sum())
    if dropped:
        print(f"[Clean] Dropped {dropped} rows with NaN/inf or extreme values")

    X = X.loc[keep_mask]
    if not np.isfinite(X.to_numpy(dtype=np.float64)).all():
        raise ValueError("Non-finite values remain after cleaning")
    return X, keep_mask


# ================= Utils =================
def row_softmax(scores: np.ndarray) -> np.ndarray:
    """Row-wise softmax to convert decision scores to pseudo-probabilities."""
    s = scores - np.max(scores, axis=1, keepdims=True)
    e = np.exp(s)
    denom = e.sum(axis=1, keepdims=True)
    denom[denom == 0] = 1.0
    return e / denom


# ================= Metrics =================
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


# ================= cuML OvR trainer/predictor =================
def fit_ovr_cuml_rbf(Xtr_cp, ytr_cp, classes_np, C=1.0, gamma="scale"):
    """
    Train one cuML SVC (RBF) per class in a One-vs-Rest fashion.
    Returns: list of fitted models (aligned to classes_np order).
    """
    from cuml.svm import SVC as cuSVC  # import after forking
    models = []
    for c in classes_np:
        y_bin = (ytr_cp == c).astype(ytr_cp.dtype)  # {0,1}
        clf = cuSVC(kernel="rbf", C=C, gamma=gamma, probability=False)
        clf.fit(Xtr_cp, y_bin)
        models.append(clf)
    return models


def ovr_decision_matrix(models, X_cp):
    """
    For each sample, collect decision_function score from each binary model.
    Returns (n_samples, n_classes) NumPy array.
    """
    import cupy as cp
    cols = []
    for clf in models:
        d = clf.decision_function(X_cp)      # cupy array (n_samples,)
        cols.append(cp.asnumpy(d).reshape(-1))
    return np.vstack(cols).T  # shape (n_samples, n_classes)


# ================= Train/Test per fold =================
def main():
    all_metrics = []

    for fold in range(5):
        print(f"\n=== Fold {fold} ===")
        split_path = os.path.join(splits_dir, f"fold{fold}", "split.csv")
        print(f"[Load] {split_path}")
        if not os.path.exists(split_path):
            raise FileNotFoundError(f"Missing split.csv for fold {fold}: {split_path}")

        df = pd.read_csv(split_path)

        smiles = df["Canonical SMILES"].tolist()
        y_all = df[target_col].to_numpy()
        split_all = df["split"].to_numpy()

        # Featurize RDKit descriptors for all rows
        print("[Featurize] Extracting RDKit descriptors in parallel...")
        X_df_full = featurize_descriptors_parallel_ordered(smiles)

        # Clean + build keep_mask
        print("[Clean] Replacing inf, winsorizing 99.9%, dropping NaNs...")
        X_df, keep_mask = clean_descriptor_df_with_mask(X_df_full, winsorize=True, p=0.999)

        # Align labels/splits using the keep_mask
        y_cpu = y_all[keep_mask]
        split = split_all[keep_mask]
        print(f"[Shape] Features: {X_df.shape}, Labels: {y_cpu.shape}")

        # Define fold-specific directories
        fold_model_dir = os.path.join(model_root, f"fold{fold}")
        os.makedirs(fold_model_dir, exist_ok=True)

        fold_pred_dir = os.path.join(pred_root, f"fold{fold}")
        os.makedirs(fold_pred_dir, exist_ok=True)
        pred_csv = os.path.join(fold_pred_dir, "pred.csv")

        classes_json = os.path.join(fold_model_dir, "classes.json")
        scaler_path  = os.path.join(fold_model_dir, "scaler.pkl")

        # Decide whether models already exist (skip retraining in that case)
        models_exist = os.path.exists(classes_json) and os.path.exists(scaler_path)

        # ===== Scaling (fit only if training) =====
        if models_exist:
            print("[Info] Existing scaler & classes found. Loading instead of refitting.")
            scaler = joblib.load(scaler_path)
            X_scaled = scaler.transform(X_df.to_numpy(dtype=np.float64)).astype(np.float32, copy=False)

            with open(classes_json, "r") as f:
                class_info = json.load(f)
            classes = np.array(class_info["classes"])
        else:
            print("[Scale] Standardizing features on CPU (fit scaler)...")
            scaler = StandardScaler(with_mean=True, with_std=True)
            X_scaled = scaler.fit_transform(X_df.to_numpy(dtype=np.float64)).astype(np.float32, copy=False)
            joblib.dump(scaler, scaler_path)

            # Classes from the cleaned data
            classes = np.unique(y_cpu)
            with open(classes_json, "w") as f:
                json.dump({"classes": [int(c) for c in classes]}, f, indent=2)

            # Also store descriptor names for this fold
            with open(os.path.join(fold_model_dir, "descriptor_names.json"), "w") as f:
                json.dump({"descriptor_names": desc_names}, f, indent=2)

        print(f"[Classes] {classes.tolist()}")

        # Prepare splits
        tr = split == "train"
        va = split == "val"
        te = split == "test"

        Xtr_np, ytr_np = X_scaled[tr], y_cpu[tr]
        Xte_np, yte_np = X_scaled[te], y_cpu[te]

        # Move to GPU for cuML training/prediction
        import cupy as cp
        Xtr_cp = cp.asarray(Xtr_np)
        ytr_cp = cp.asarray(ytr_np)
        Xte_cp = cp.asarray(Xte_np)

        # ===== Train or load models =====
        models = []
        model_paths = [os.path.join(fold_model_dir, f"ovr_class_{int(c)}.pkl") for c in classes]
        if models_exist and all(os.path.exists(p) for p in model_paths):
            print("[Info] Existing OvR models found. Loading instead of retraining.")
            for p in model_paths:
                clf = joblib.load(p)
                models.append(clf)
        else:
            print(f"[Train] cuML SVC (RBF) manual OvR over classes: {classes.tolist()}")
            models = fit_ovr_cuml_rbf(Xtr_cp, ytr_cp, classes, C=1.0, gamma="scale")

            # Save models
            for c, clf in zip(classes, models):
                path_c = os.path.join(fold_model_dir, f"ovr_class_{int(c)}.pkl")
                joblib.dump(clf, path_c)
            print(f"[Save] Models & scaler & classes → {fold_model_dir}")

        # ===== Evaluate on test (always done, even if models were loaded) =====
        print("[Eval] Getting OvR decision scores and predictions on test set...")
        dec_mat = ovr_decision_matrix(models, Xte_cp)  # (n_test, n_classes)
        y_pred = classes[np.argmax(dec_mat, axis=1)]
        y_prob = row_softmax(dec_mat)                  # rows sum to 1 (for ROC-AUC)

        # Metrics
        metrics = compute_metrics(yte_np, y_pred, y_prob)
        metrics["fold"] = fold
        all_metrics.append(metrics)
        print("[Metrics]", metrics)

        # ===== Save test predictions like ChemBERTa =====
        # We need the original test rows aligned with the cleaned/kept mask
        df_kept = df.loc[keep_mask].reset_index(drop=True)
        df_test = df_kept[split == "test"].copy()

        if df_test.shape[0] != y_prob.shape[0]:
            raise RuntimeError(
                f"Row mismatch on fold {fold}: "
                f"test rows={df_test.shape[0]} vs prob rows={y_prob.shape[0]}"
            )

        # Add probability columns per class
        for idx, cls in enumerate(classes):
            df_test[f"prob_{int(cls)}"] = y_prob[:, idx]

        df_test.to_csv(pred_csv, index=False)
        print(f"[Save] Test predictions → {pred_csv}")

    # ===== Save metrics summary over all folds =====
    metrics_path = os.path.join(model_root, "metrics.csv")
    pd.DataFrame(all_metrics).to_csv(metrics_path, index=False)
    print(f"\n[Done] Metrics CSV → {metrics_path}")


if __name__ == "__main__":
    main()
