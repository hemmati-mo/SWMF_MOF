#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import subprocess
from pathlib import Path


MODEL_ROOT = Path("../Models/ip_data_cleaned/classification/Chemprop")
DEFAULT_FOLDS = [0, 1, 2, 3, 4]
DEFAULT_SMILES_COL = "smiles"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Predict acute toxicity classes with the Chemprop ensemble."
    )
    parser.add_argument("--input_csv", required=True, help="Input CSV containing SMILES.")
    parser.add_argument("--output_csv", required=True, help="Output CSV for predictions.")
    parser.add_argument("--smiles_col", default=DEFAULT_SMILES_COL, help="SMILES column name.")
    parser.add_argument("--model_root", default=str(MODEL_ROOT), help="Chemprop model directory.")
    parser.add_argument("--folds", nargs="+", type=int, default=DEFAULT_FOLDS, help="Fold IDs to use.")
    parser.add_argument("--chemprop_bin", default="chemprop", help="Chemprop executable.")
    return parser.parse_args()


def run_cmd(cmd):
    print("\n[CMD]", " ".join(map(str, cmd)))
    subprocess.run(cmd, check=True)


def main():
    args = parse_args()
    input_csv = Path(args.input_csv)
    output_csv = Path(args.output_csv)
    model_root = Path(args.model_root)

    print(f"[Load] {input_csv}")
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    model_dirs = []
    for fold in args.folds:
        model_dir = model_root / f"fold_{fold}"
        if not model_dir.exists():
            raise FileNotFoundError(f"Missing Chemprop fold dir: {model_dir}")
        model_dirs.append(str(model_dir))

    print("[Info] Using model paths:")
    for model_dir in model_dirs:
        print("   ", model_dir)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        args.chemprop_bin,
        "predict",
        "--test-path",
        str(input_csv),
        "--smiles-columns",
        args.smiles_col,
        "--model-paths",
        *model_dirs,
        "--preds-path",
        str(output_csv),
    ]
    run_cmd(cmd)
    print(f"[Done] Chemprop predictions saved to: {output_csv}")


if __name__ == "__main__":
    main()
