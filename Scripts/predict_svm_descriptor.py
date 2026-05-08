#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path

import cupy as cp
import joblib
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from tqdm import tqdm


RDLogger.DisableLog("rdApp.warning")

MODEL_ROOT = Path("../Models/ip_data_cleaned/classification/SVM_descriptor")
DEFAULT_FOLDS = [0, 1, 2, 3, 4]
DEFAULT_SMILES_COL = "smiles"
N_CLASSES = 3

DESC_NAMES = [d[0] for d in Descriptors._descList]
CALC = MoleculeDescriptors.MolecularDescriptorCalculator(DESC_NAMES)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Predict acute toxicity classes with the SVM descriptor ensemble."
    )
    parser.add_argument("--input_csv", required=True, help="Input CSV containing SMILES.")
    parser.add_argument("--output_csv", required=True, help="Output CSV for predictions.")
    parser.add_argument("--smiles_col", default=DEFAULT_SMILES_COL, help="SMILES column name.")
    parser.add_argument("--model_root", default=str(MODEL_ROOT), help="SVM descriptor model directory.")
    parser.add_argument("--folds", nargs="+", type=int, default=DEFAULT_FOLDS, help="Fold IDs to use.")
    return parser.parse_args()


def smiles_to_desc_row(smiles: str):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return [np.nan] * len(DESC_NAMES)
        vals = list(CALC.CalcDescriptors(mol))
        out = []
        for val in vals:
            try:
                val = float(val)
                out.append(val if np.isfinite(val) else np.nan)
            except Exception:
                out.append(np.nan)
        return out
    except Exception:
        return [np.nan] * len(DESC_NAMES)


def clean_descriptor_df_with_mask(x_df: pd.DataFrame, winsorize: bool = True, p: float = 0.999):
    x = x_df.replace([np.inf, -np.inf], np.nan)
    if winsorize:
        q = x.quantile(p, axis=0, numeric_only=True)
        x = x.clip(lower=None, upper=q, axis=1)
    keep_mask = ~x.isna().any(axis=1).to_numpy()
    return x.loc[keep_mask], keep_mask


def row_softmax(scores: np.ndarray) -> np.ndarray:
    shifted = scores - np.max(scores, axis=1, keepdims=True)
    exp_scores = np.exp(shifted)
    denom = exp_scores.sum(axis=1, keepdims=True)
    denom[denom == 0] = 1.0
    return exp_scores / denom


def ovr_decision_matrix(models, x_cp):
    cols = []
    for clf in models:
        decision = clf.decision_function(x_cp)
        cols.append(cp.asnumpy(decision).reshape(-1))
    return np.vstack(cols).T


def main():
    args = parse_args()
    input_csv = Path(args.input_csv)
    output_csv = Path(args.output_csv)
    model_root = Path(args.model_root)

    print(f"[Load] {input_csv}")
    df = pd.read_csv(input_csv)
    if args.smiles_col not in df.columns:
        raise ValueError(f"Expected SMILES column '{args.smiles_col}' in {input_csv}")

    smiles = df[args.smiles_col].astype(str).tolist()
    n_rows = len(df)
    print(f"[Info] Number of molecules: {n_rows}")
    print(f"[Config] Number of RDKit descriptors: {len(DESC_NAMES)}")

    print("[Featurize] Computing RDKit descriptors...")
    rows = [smiles_to_desc_row(s) for s in tqdm(smiles, desc="RDKit descriptors")]
    x_df_full = pd.DataFrame(rows, columns=DESC_NAMES, dtype=np.float64)

    fold_probs = []
    for fold in args.folds:
        print(f"\n=== SVM Descriptor Fold {fold} ===")
        fold_dir = model_root / f"fold{fold}"
        scaler_path = fold_dir / "scaler.pkl"
        classes_path = fold_dir / "classes.json"
        if not scaler_path.exists() or not classes_path.exists():
            raise FileNotFoundError(f"Missing scaler/classes in {fold_dir}")

        scaler = joblib.load(scaler_path)
        with classes_path.open() as f:
            classes = np.array(json.load(f)["classes"], dtype=int)

        models = []
        for cls in classes:
            clf_path = fold_dir / f"ovr_class_{int(cls)}.pkl"
            if not clf_path.exists():
                raise FileNotFoundError(f"Missing classifier: {clf_path}")
            models.append(joblib.load(clf_path))

        x_clean, keep_mask = clean_descriptor_df_with_mask(x_df_full, winsorize=True, p=0.999)
        print(f"[Clean] Kept {keep_mask.sum()} / {n_rows} rows after NaN/inf filtering")
        x_scaled = scaler.transform(x_clean.to_numpy(dtype=np.float64)).astype(np.float32, copy=False)
        x_cp = cp.asarray(x_scaled)

        print("[Predict] Getting decision scores for kept rows...")
        probs_kept = row_softmax(ovr_decision_matrix(models, x_cp))
        probs_fold = np.full((n_rows, N_CLASSES), np.nan, dtype=float)
        kept_idx = np.where(keep_mask)[0]
        for j, cls in enumerate(classes):
            if 0 <= cls < N_CLASSES:
                probs_fold[kept_idx, int(cls)] = probs_kept[:, j]
        fold_probs.append(probs_fold)

    print("\n[Aggregate] Averaging SVM descriptor probabilities across folds...")
    mean_probs = np.nanmean(np.stack(fold_probs, axis=0), axis=0)

    df_out = df.copy()
    for cls in range(N_CLASSES):
        df_out[f"prob_{cls}"] = mean_probs[:, cls]

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    print(f"[Save] {output_csv}")
    df_out.to_csv(output_csv, index=False)
    print("[Done] SVM descriptor predictions saved.")


if __name__ == "__main__":
    main()
