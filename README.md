# Ensemble Machine Learning for Interpretable Prediction of Acute Toxicity in Metal-Organic Framework Linkers

This repository contains the code for the paper [Ensemble Machine Learning for Interpretable Prediction of Acute Toxicity in Metal-Organic Framework Linkers](https://pubs.acs.org/doi/10.1021/acs.jcim.6c00227).

It includes scripts used to train, evaluate, interpret, and apply ensemble machine-learning models for acute toxicity classification of organic linkers.

The workflow combines four model families:

- Random forest on Morgan fingerprints
- SVM on RDKit molecular descriptors
- ChemBERTa sequence classification
- Chemprop message-passing neural networks

Trained model weights and generated prediction CSV files are intentionally excluded from version control. Recreate them locally with the training scripts, or place locally trained artifacts under `Models/` using the directory layout expected by the prediction scripts.

## Repository Layout

```text
Scripts/
  run_rf_morgan_5fold.py          Train/evaluate RF Morgan models
  run_svm_descriptors_5fold.py    Train/evaluate SVM descriptor models
  run_chemberta2_5fold.py         Train/evaluate ChemBERTa models
  run_chemprop2_5fold.py          Train/evaluate Chemprop models
  predict_rf_morgan.py            Generic RF Morgan CSV predictor
  predict_svm_descriptor.py       Generic SVM descriptor CSV predictor
  predict_chemberta.py            Generic ChemBERTa CSV predictor
  predict_chemprop.py             Generic Chemprop CSV predictor
  interpret_chemprop_shap.py      Chemprop SHAP and SWMF construction

data/
  ip_data_cleaned_mapped.csv      Curated acute toxicity data
  splits/                         Five-fold train/validation/test splits

Figures/
  Figure3_ip_data_training_and_performance.*
  Figure4_chemical_space_AD.*
  Figure5_swmf.*
```

## Environment

The scripts were written for Python 3 and use common scientific Python tooling plus chemistry and deep-learning packages. Install versions compatible with your CUDA and Chemprop setup.

Core packages:

- `numpy`
- `pandas`
- `scikit-learn`
- `scipy`
- `rdkit`
- `joblib`
- `tqdm`
- `torch`
- `transformers`
- `datasets`
- `chemprop`
- `cupy` and `cuml` for the GPU SVM workflow

Example:

```bash
conda create -n linker-tox python=3.10
conda activate linker-tox
conda install -c conda-forge rdkit numpy pandas scikit-learn scipy joblib tqdm
pip install torch transformers datasets chemprop
```

Install `cupy`/`cuml` according to the CUDA version available on your machine.

## Training

The five-fold training scripts expect split files under:

```text
data/splits/ip_data_cleaned/fold0/split.csv
data/splits/ip_data_cleaned/fold1/split.csv
...
data/splits/ip_data_cleaned/fold4/split.csv
```

Each split CSV should contain:

- `Canonical SMILES`
- `Category`
- `split`, with values `train`, `val`, or `test`

Run models from the repository root or from `Scripts/` with paths adjusted as needed:

```bash
cd Scripts
python run_rf_morgan_5fold.py
python run_svm_descriptors_5fold.py
python run_chemberta2_5fold.py
python run_chemprop2_5fold.py
```

Training writes local model artifacts to `Models/` and fold-level prediction files to `predictions/`. Both directories are ignored by Git.

## Generic Prediction

The prediction scripts accept any CSV containing a SMILES column and write a copy of the input with `prob_0`, `prob_1`, and `prob_2` columns appended.

By default, predictors look for a column named `smiles`. Use `--smiles_col` for another column name.

```bash
python Scripts/predict_rf_morgan.py \
  --input_csv path/to/input.csv \
  --output_csv path/to/rf_predictions.csv \
  --smiles_col smiles

python Scripts/predict_svm_descriptor.py \
  --input_csv path/to/input.csv \
  --output_csv path/to/svm_predictions.csv \
  --smiles_col smiles

python Scripts/predict_chemberta.py \
  --input_csv path/to/input.csv \
  --output_csv path/to/chemberta_predictions.csv \
  --smiles_col smiles

python Scripts/predict_chemprop.py \
  --input_csv path/to/input.csv \
  --output_csv path/to/chemprop_predictions.csv \
  --smiles_col smiles
```

Use `--model_root` and `--folds` if your local model directory differs from the default layout.

## Outputs

Prediction columns represent class probabilities for the three acute toxicity categories used during training:

- `prob_0`
- `prob_1`
- `prob_2`

Generated outputs should remain local unless a specific result table is intentionally prepared for release.

## Interpretability and SWMF

`Scripts/interpret_chemprop_shap.py` runs Chemprop atom/bond masking with SHAP for a selected set of molecules, then aggregates the local attributions into SHAP-weighted molecular fragments (SWMF).

Example:

```bash
python Scripts/interpret_chemprop_shap.py \
  --checkpoint Models/ip_data_cleaned/classification/Chemprop/fold_0/model_0/checkpoints/last.ckpt \
  --input_csv data/ip_data_cleaned_mapped.csv \
  --output_dir interpretability/chemprop_shap \
  --smiles_col "Canonical SMILES" \
  --group_col Category \
  --groups 0 1 2 \
  --n_per_group 10 \
  --target_class 2 \
  --max_evals 200 \
  --fragment_radius 1
```

The script writes:

- `atom_shap.csv`: atom-level SHAP values and atom-environment fragments
- `bond_shap.csv`: bond-level SHAP values and bond fragments
- `feature_shap.csv`: combined atom/bond attribution table
- `swmf_fragments.csv`: SWMF table grouped by class, group, feature type, and fragment

Use `swmf_fragments.csv` to rank fragments by `swmf_abs_score` for overall importance or by signed `swmf_score` to identify fragments that increase or decrease the target-class probability.
