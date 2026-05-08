#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# --- Optional: sanitize weird HF headers env that can crash huggingface_hub ---
import os as _os
for _k in ("HF_HUB_HEADERS", "HF_HEADERS", "HUGGINGFACE_HEADERS"):
    if _k in _os.environ:
        _os.environ.pop(_k, None)

# NO offline flags here; behave like your old script.

import os
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)

from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, roc_auc_score,
    balanced_accuracy_score, f1_score, matthews_corrcoef
)
from scipy.special import softmax

# ===================== Paths & config =====================
DATA_ROOT = Path("../data/splits/ip_data_cleaned")
MODEL_ROOT = Path("../Models/ip_data_cleaned/classification/Chemberta")
PRED_ROOT  = Path("../predictions/ip_data_cleaned/classification/chemberta")

SMILES_COL = "Canonical SMILES"
TARGET_COL = "Category"     # 3-class target: {0,1,2}
SPLIT_COL  = "split"

FOLDS = [0, 1, 2, 3, 4]

MODEL_NAME = "DeepChem/ChemBERTa-77M-MLM"   # exactly as in your working script

EPOCHS = 5
TRAIN_BS = 32
EVAL_BS  = 64
FP16 = True

# ===================== Metrics =====================
def compute_metrics_builder(num_labels: int):
    def compute_metrics(eval_pred: Tuple[np.ndarray, np.ndarray]) -> Dict[str, Any]:
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=1)

        acc = accuracy_score(labels, preds)
        bal_acc = balanced_accuracy_score(labels, preds)
        prec, rec, f1_w, _ = precision_recall_fscore_support(labels, preds, average="weighted")
        f1_bal = f1_score(labels, preds, average="macro")
        mcc = matthews_corrcoef(labels, preds)

        auc = np.nan
        try:
            if num_labels > 2:
                auc = roc_auc_score(labels, logits, multi_class="ovr")
            else:
                auc = roc_auc_score(labels, logits[:, 1])
        except Exception:
            pass

        return {
            "accuracy": acc,
            "auc": float(auc) if auc == auc else np.nan,
            "precision": prec,
            "recall": rec,
            "f1": f1_w,
            "balanced_f1": f1_bal,
            "balanced_accuracy": bal_acc,
            "mcc": mcc,
        }
    return compute_metrics

# ===================== Data utils =====================
def make_datasets(tokenizer, df: pd.DataFrame):
    df_local = df.rename(columns={SMILES_COL: "smiles"}).copy()
    df_local["labels"] = df_local[TARGET_COL].astype(int)

    dataset = Dataset.from_pandas(df_local[["smiles", "labels", SPLIT_COL]])
    dataset = dataset.map(lambda b: tokenizer(b["smiles"], truncation=True), batched=True)

    train_ds = dataset.filter(lambda x: x[SPLIT_COL] == "train")
    val_ds   = dataset.filter(lambda x: x[SPLIT_COL] == "val")
    test_ds  = dataset.filter(lambda x: x[SPLIT_COL] == "test")
    return train_ds, val_ds, test_ds

def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

# ===================== Main CV loop =====================
def main():
    device = get_device()
    print(f"Using device: {device}")

    # behave like your old script: global tokenizer from MODEL_NAME
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    for fold in FOLDS:
        print(f"\n================ Fold {fold} ================")
        split_csv = DATA_ROOT / f"fold{fold}/split.csv"
        if not split_csv.exists():
            raise FileNotFoundError(f"Missing CSV for fold {fold}: {split_csv}")

        df = pd.read_csv(split_csv)

        unique_labels = sorted(df[TARGET_COL].dropna().astype(int).unique().tolist())
        num_labels = len(unique_labels)
        if num_labels != 3:
            print(f"Warning: expected 3 labels; found {unique_labels}. Proceeding with num_labels={num_labels}.")

        # Datasets
        train_ds, val_ds, test_ds = make_datasets(tokenizer, df)

        # === Model config + model (same style as your previous script) ===
        config = AutoConfig.from_pretrained(
            MODEL_NAME,
            num_labels=num_labels,
            problem_type="single_label_classification",
        )
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, config=config)
        model.to(device)

        # === Output dirs ===
        model_out = MODEL_ROOT / f"fold_{fold}"
        ensure_dir(model_out)
        logs_out = model_out / "logs"
        ensure_dir(logs_out)

        # === Training args ===
        training_args = TrainingArguments(
            output_dir=str(model_out),
            overwrite_output_dir=True,
            num_train_epochs=EPOCHS,
            learning_rate=2e-5,
            weight_decay=0.0,
            warmup_ratio=0.06,
            per_device_train_batch_size=TRAIN_BS,
            per_device_eval_batch_size=EVAL_BS,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="mcc",
            greater_is_better=True,
            fp16=(FP16 and device == "cuda"),
            logging_dir=str(logs_out),
            logging_steps=50,
            report_to=[],
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            tokenizer=tokenizer,
            data_collator=DataCollatorWithPadding(tokenizer),
            compute_metrics=compute_metrics_builder(num_labels),
        )

        print(f"Training on: {split_csv}")
        trainer.train()

        # === Predict probabilities on test split ===
        print("Predicting on test split...")
        pred_output = trainer.predict(test_ds)
        logits = pred_output.predictions
        probs = softmax(logits, axis=1) if logits.ndim == 2 else np.zeros((len(logits), num_labels))

        # Map back to test rows of original CSV and write pred.csv
        df_test = df[df[SPLIT_COL] == "test"].copy()
        if len(df_test) != probs.shape[0]:
            raise RuntimeError(f"Row mismatch: test rows={len(df_test)} vs prob rows={probs.shape[0]}")

        for k in range(num_labels):
            df_test[f"prob_{k}"] = probs[:, k]

        pred_dir = PRED_ROOT / f"fold_{fold}"
        ensure_dir(pred_dir)
        pred_csv = pred_dir / "pred.csv"
        df_test.to_csv(pred_csv, index=False)
        print(f"Saved probabilities: {pred_csv}")

        # Save the final model snapshot
        trainer.save_model(str(model_out))
        print(f"Saved model to: {model_out}")

if __name__ == "__main__":
    main()
