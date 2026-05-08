#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from scipy.special import softmax
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments


MODEL_ROOT = Path("../Models/ip_data_cleaned/classification/Chemberta")
MODEL_NAME = "DeepChem/ChemBERTa-77M-MLM"
DEFAULT_FOLDS = [0, 1, 2, 3, 4]
DEFAULT_SMILES_COL = "smiles"
N_CLASSES = 3


def parse_args():
    parser = argparse.ArgumentParser(
        description="Predict acute toxicity classes with the ChemBERTa ensemble."
    )
    parser.add_argument("--input_csv", required=True, help="Input CSV containing SMILES.")
    parser.add_argument("--output_csv", required=True, help="Output CSV for predictions.")
    parser.add_argument("--smiles_col", default=DEFAULT_SMILES_COL, help="SMILES column name.")
    parser.add_argument("--model_root", default=str(MODEL_ROOT), help="ChemBERTa model directory.")
    parser.add_argument("--model_name", default=MODEL_NAME, help="Tokenizer base model name or path.")
    parser.add_argument("--folds", nargs="+", type=int, default=DEFAULT_FOLDS, help="Fold IDs to use.")
    parser.add_argument("--batch_size", type=int, default=64, help="Evaluation batch size.")
    return parser.parse_args()


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

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Device] {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    ds = Dataset.from_dict({"smiles": smiles})
    ds = ds.map(lambda batch: tokenizer(batch["smiles"], truncation=True), batched=True)

    fold_probs = []
    for fold in args.folds:
        print(f"\n=== ChemBERTa Fold {fold} ===")
        fold_dir = model_root / f"fold_{fold}"
        if not fold_dir.exists():
            raise FileNotFoundError(f"Missing fold directory: {fold_dir}")

        print(f"[Load model] {fold_dir}")
        model = AutoModelForSequenceClassification.from_pretrained(str(fold_dir))
        model.to(device)

        training_args = TrainingArguments(
            output_dir=str(fold_dir / "inference_tmp"),
            per_device_eval_batch_size=args.batch_size,
            dataloader_num_workers=4,
            fp16=(device == "cuda"),
            report_to=[],
        )
        trainer = Trainer(model=model, args=training_args, tokenizer=tokenizer)

        print("[Predict] Running model on all molecules...")
        logits = trainer.predict(ds).predictions
        probs = softmax(logits, axis=1)
        if probs.shape[1] != N_CLASSES:
            raise RuntimeError(f"Expected {N_CLASSES} classes, got shape {probs.shape}")
        fold_probs.append(probs)

    print("\n[Aggregate] Averaging ChemBERTa probabilities across folds...")
    mean_probs = np.stack(fold_probs, axis=0).mean(axis=0)

    df_out = df.copy()
    for cls in range(N_CLASSES):
        df_out[f"prob_{cls}"] = mean_probs[:, cls]

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    print(f"[Save] {output_csv}")
    df_out.to_csv(output_csv, index=False)
    print("[Done] ChemBERTa predictions saved.")


if __name__ == "__main__":
    main()
