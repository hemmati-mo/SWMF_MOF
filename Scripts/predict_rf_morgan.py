#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs import ConvertToNumpyArray
from tqdm import tqdm


MODEL_ROOT = Path("../Models/ip_data_cleaned/classification/RF_Morgan")
DEFAULT_FOLDS = [0, 1, 2, 3, 4]
DEFAULT_SMILES_COL = "smiles"
FP_SIZE = 2048
N_CLASSES = 3


def parse_args():
    parser = argparse.ArgumentParser(
        description="Predict acute toxicity classes with the RF Morgan ensemble."
    )
    parser.add_argument("--input_csv", required=True, help="Input CSV containing SMILES.")
    parser.add_argument("--output_csv", required=True, help="Output CSV for predictions.")
    parser.add_argument("--smiles_col", default=DEFAULT_SMILES_COL, help="SMILES column name.")
    parser.add_argument("--model_root", default=str(MODEL_ROOT), help="RF Morgan model directory.")
    parser.add_argument("--folds", nargs="+", type=int, default=DEFAULT_FOLDS, help="Fold IDs to use.")
    return parser.parse_args()


def smiles_to_morgan(smiles, radius=2, n_bits=FP_SIZE):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.zeros((n_bits,), dtype=np.float32)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        arr = np.zeros((n_bits,), dtype=np.float32)
        ConvertToNumpyArray(fp, arr)
        return arr
    except Exception:
        return np.zeros((n_bits,), dtype=np.float32)


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
    print(f"[Config] Morgan fingerprint size: {FP_SIZE}")

    print("[Featurize] Computing Morgan fingerprints...")
    x = np.vstack(
        [smiles_to_morgan(s) for s in tqdm(smiles, desc="Morgan FPs")]
    ).astype(np.float32)

    fold_probs = []
    for fold in args.folds:
        print(f"\n=== RF Morgan Fold {fold} ===")
        fold_dir = model_root / f"fold{fold}"
        model_path = fold_dir / "rf_morgan.pkl"
        if not model_path.exists():
            raise FileNotFoundError(f"Missing RF model at {model_path}")

        rf = joblib.load(model_path)
        if not hasattr(rf, "predict_proba"):
            raise RuntimeError("RF model has no predict_proba; was it trained as a classifier?")

        classes = rf.classes_
        probs = rf.predict_proba(x)
        probs_fold = np.full((n_rows, N_CLASSES), np.nan, dtype=float)
        for j, cls in enumerate(classes):
            if 0 <= cls < N_CLASSES:
                probs_fold[:, int(cls)] = probs[:, j]
        fold_probs.append(probs_fold)

    print("\n[Aggregate] Averaging RF Morgan probabilities across folds...")
    mean_probs = np.nanmean(np.stack(fold_probs, axis=0), axis=0)

    df_out = df.copy()
    for cls in range(N_CLASSES):
        df_out[f"prob_{cls}"] = mean_probs[:, cls]

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    print(f"[Save] {output_csv}")
    df_out.to_csv(output_csv, index=False)
    print("[Done] RF Morgan predictions saved.")


if __name__ == "__main__":
    main()
