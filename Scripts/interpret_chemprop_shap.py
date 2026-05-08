#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
from copy import deepcopy
from pathlib import Path
from typing import List, Optional

import lightning.pytorch as pl
import numpy as np
import pandas as pd
import shap
import torch
from rdkit import Chem

from chemprop import data, models
from chemprop.featurizers import (
    CustomMultiHotAtomFeaturizer,
    CustomMultiHotBondFeaturizer,
    CustomSimpleMoleculeMolGraphFeaturizer,
)


ATOM_FEATURIZER = CustomMultiHotAtomFeaturizer.v2()
BOND_FEATURIZER = CustomMultiHotBondFeaturizer()


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run Chemprop atom/bond SHAP interpretation and aggregate "
            "SHAP-weighted molecular fragments."
        )
    )
    parser.add_argument("--checkpoint", required=True, help="Path to a Chemprop .ckpt checkpoint.")
    parser.add_argument("--input_csv", required=True, help="Input CSV containing molecules to explain.")
    parser.add_argument("--output_dir", default="interpretability/chemprop_shap", help="Output directory.")
    parser.add_argument("--smiles_col", default="smiles", help="SMILES column name.")
    parser.add_argument("--group_col", default=None, help="Optional column for stratified sampling.")
    parser.add_argument("--groups", nargs="+", default=None, help="Optional group values to explain.")
    parser.add_argument("--n_per_group", type=int, default=10, help="Molecules to sample per group.")
    parser.add_argument("--target_class", type=int, default=2, help="Class index to explain.")
    parser.add_argument("--max_evals", type=int, default=200, help="SHAP evaluation budget.")
    parser.add_argument("--fragment_radius", type=int, default=1, help="Atom-environment radius for SWMF.")
    parser.add_argument("--random_state", type=int, default=42, help="Sampling seed.")
    return parser.parse_args()


def load_chemprop_model(checkpoint: Path) -> models.MPNN:
    print(f"[MODEL] Loading Chemprop MPNN from {checkpoint}")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint}")
    mpnn = models.MPNN.load_from_checkpoint(checkpoint)
    mpnn.eval()
    return mpnn


def get_prediction(
    mpnn: models.MPNN,
    smiles: str,
    keep_atoms: Optional[List[bool]],
    keep_bonds: Optional[List[bool]],
    target_class: int,
) -> float:
    featurizer = CustomSimpleMoleculeMolGraphFeaturizer(
        atom_featurizer=ATOM_FEATURIZER,
        bond_featurizer=BOND_FEATURIZER,
        keep_atoms=keep_atoms,
        keep_bonds=keep_bonds,
    )
    datapoint = data.MoleculeDatapoint.from_smi(smiles)
    dset = data.MoleculeDataset([datapoint], featurizer=featurizer)
    loader = data.build_dataloader(dset, shuffle=False, batch_size=1)

    trainer = pl.Trainer(
        logger=False,
        enable_progress_bar=False,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
    )

    with torch.inference_mode():
        preds_list = trainer.predict(mpnn, loader)

    logits = preds_list[0]
    probs = torch.softmax(logits, dim=-1)[0]
    return float(probs[target_class].item())


class AtomBondMaskWrapper:
    def __init__(
        self,
        mpnn: models.MPNN,
        smiles: str,
        n_atoms: int,
        n_bonds: int,
        target_class: int,
    ):
        self.mpnn = mpnn
        self.smiles = smiles
        self.n_atoms = n_atoms
        self.n_bonds = n_bonds
        self.target_class = target_class

    def __call__(self, x: np.ndarray) -> np.ndarray:
        preds = []
        for mask_vec in x:
            keep_atoms = mask_vec[: self.n_atoms].astype(bool).tolist()
            keep_bonds = mask_vec[self.n_atoms : self.n_atoms + self.n_bonds].astype(bool).tolist()
            pred = get_prediction(
                self.mpnn,
                self.smiles,
                keep_atoms=keep_atoms,
                keep_bonds=keep_bonds,
                target_class=self.target_class,
            )
            preds.append([pred])
        return np.array(preds)


def binary_masker(binary_mask: np.ndarray, x: np.ndarray) -> np.ndarray:
    masked_x = deepcopy(x)
    masked_x[binary_mask == 0] = 0
    return np.array([masked_x])


def atom_fragment_label(mol: Chem.Mol, atom_idx: int, radius: int) -> str:
    if radius <= 0:
        return mol.GetAtomWithIdx(atom_idx).GetSymbol()
    bond_ids = list(Chem.FindAtomEnvironmentOfRadiusN(mol, radius, atom_idx))
    atom_ids = {atom_idx}
    for bond_id in bond_ids:
        bond = mol.GetBondWithIdx(bond_id)
        atom_ids.add(bond.GetBeginAtomIdx())
        atom_ids.add(bond.GetEndAtomIdx())
    return Chem.MolFragmentToSmiles(
        mol,
        atomsToUse=sorted(atom_ids),
        bondsToUse=sorted(bond_ids),
        rootedAtAtom=atom_idx,
        canonical=True,
    )


def bond_fragment_label(mol: Chem.Mol, bond_idx: int) -> str:
    bond = mol.GetBondWithIdx(bond_idx)
    atom_ids = sorted([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    return Chem.MolFragmentToSmiles(
        mol,
        atomsToUse=atom_ids,
        bondsToUse=[bond_idx],
        canonical=True,
    )


def run_shap_for_molecule(
    mpnn: models.MPNN,
    smiles: str,
    molecule_id: str,
    group_name: str,
    target_class: int,
    max_evals: int,
    fragment_radius: int,
):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"[WARN] RDKit failed on SMILES, skipping: {smiles}")
        return [], []

    n_atoms = mol.GetNumAtoms()
    n_bonds = mol.GetNumBonds()
    if n_atoms == 0:
        print(f"[WARN] 0-atom molecule, skipping: {smiles}")
        return [], []

    print(
        f"[SHAP] group={group_name}, molecule_id={molecule_id}, "
        f"atoms={n_atoms}, bonds={n_bonds}"
    )

    base_features = np.ones(n_atoms + n_bonds, dtype=int)
    explainer = shap.PermutationExplainer(
        AtomBondMaskWrapper(mpnn, smiles, n_atoms, n_bonds, target_class),
        masker=binary_masker,
    )
    explanation = explainer(np.array([base_features]), max_evals=max_evals)
    shap_vals = explanation.values[0]

    atom_rows = []
    for atom_idx in range(n_atoms):
        atom = mol.GetAtomWithIdx(atom_idx)
        shap_value = float(shap_vals[atom_idx])
        atom_rows.append(
            {
                "molecule_id": molecule_id,
                "group": group_name,
                "smiles": smiles,
                "target_class": target_class,
                "feature_type": "atom",
                "feature_index": atom_idx,
                "atom_symbol": atom.GetSymbol(),
                "fragment": atom_fragment_label(mol, atom_idx, fragment_radius),
                "shap_value": shap_value,
                "abs_shap_value": abs(shap_value),
            }
        )

    bond_rows = []
    for bond_idx in range(n_bonds):
        bond = mol.GetBondWithIdx(bond_idx)
        shap_value = float(shap_vals[n_atoms + bond_idx])
        bond_rows.append(
            {
                "molecule_id": molecule_id,
                "group": group_name,
                "smiles": smiles,
                "target_class": target_class,
                "feature_type": "bond",
                "feature_index": bond_idx,
                "begin_atom_idx": bond.GetBeginAtomIdx(),
                "end_atom_idx": bond.GetEndAtomIdx(),
                "bond_type": str(bond.GetBondType()),
                "fragment": bond_fragment_label(mol, bond_idx),
                "shap_value": shap_value,
                "abs_shap_value": abs(shap_value),
            }
        )

    return atom_rows, bond_rows


def select_rows(df: pd.DataFrame, args) -> pd.DataFrame:
    if args.group_col is None:
        n_take = min(args.n_per_group, len(df))
        sample = df.sample(n=n_take, random_state=args.random_state)
        sample = sample.copy()
        sample["_interpret_group"] = "all"
        return sample

    if args.group_col not in df.columns:
        raise ValueError(f"Expected group column '{args.group_col}' in {args.input_csv}")

    group_values = args.groups or sorted(df[args.group_col].dropna().astype(str).unique().tolist())
    samples = []
    for group_value in group_values:
        sub = df[df[args.group_col].astype(str) == str(group_value)]
        if sub.empty:
            print(f"[WARN] Group '{group_value}' has 0 molecules, skipping.")
            continue
        n_take = min(args.n_per_group, len(sub))
        sample = sub.sample(n=n_take, random_state=args.random_state).copy()
        sample["_interpret_group"] = str(group_value)
        samples.append(sample)
        print(f"[GROUP] {group_value}: {len(sub)} total, taking {n_take}.")

    if not samples:
        raise RuntimeError("No molecules selected for interpretation.")
    return pd.concat(samples, axis=0)


def build_swmf_table(feature_df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["target_class", "group", "feature_type", "fragment"]
    out = (
        feature_df.groupby(group_cols, dropna=False)
        .agg(
            n_occurrences=("shap_value", "size"),
            mean_shap=("shap_value", "mean"),
            mean_abs_shap=("abs_shap_value", "mean"),
            sum_shap=("shap_value", "sum"),
            sum_abs_shap=("abs_shap_value", "sum"),
        )
        .reset_index()
    )
    out["swmf_score"] = out["mean_shap"]
    out["swmf_abs_score"] = out["mean_abs_shap"]
    return out.sort_values(["swmf_abs_score", "n_occurrences"], ascending=[False, False])


def main():
    args = parse_args()
    input_csv = Path(args.input_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[LOAD] {input_csv}")
    df = pd.read_csv(input_csv)
    if args.smiles_col not in df.columns:
        raise ValueError(f"Expected SMILES column '{args.smiles_col}' in {input_csv}")

    selected = select_rows(df, args)
    selected = selected.reset_index(drop=False).rename(columns={"index": "source_row_index"})

    mpnn = load_chemprop_model(Path(args.checkpoint))

    all_atom_rows = []
    all_bond_rows = []
    for row_idx, row in selected.iterrows():
        molecule_id = str(row.get("source_row_index", row_idx))
        smiles = str(row[args.smiles_col])
        group_name = str(row["_interpret_group"])
        atom_rows, bond_rows = run_shap_for_molecule(
            mpnn=mpnn,
            smiles=smiles,
            molecule_id=molecule_id,
            group_name=group_name,
            target_class=args.target_class,
            max_evals=args.max_evals,
            fragment_radius=args.fragment_radius,
        )
        all_atom_rows.extend(atom_rows)
        all_bond_rows.extend(bond_rows)

    atom_df = pd.DataFrame(all_atom_rows)
    bond_df = pd.DataFrame(all_bond_rows)
    feature_df = pd.concat([atom_df, bond_df], axis=0, ignore_index=True)
    swmf_df = build_swmf_table(feature_df)

    atom_path = output_dir / "atom_shap.csv"
    bond_path = output_dir / "bond_shap.csv"
    feature_path = output_dir / "feature_shap.csv"
    swmf_path = output_dir / "swmf_fragments.csv"

    atom_df.to_csv(atom_path, index=False)
    bond_df.to_csv(bond_path, index=False)
    feature_df.to_csv(feature_path, index=False)
    swmf_df.to_csv(swmf_path, index=False)

    print(f"[SAVE] Atom SHAP: {atom_path}")
    print(f"[SAVE] Bond SHAP: {bond_path}")
    print(f"[SAVE] Feature SHAP: {feature_path}")
    print(f"[SAVE] SWMF table: {swmf_path}")
    print("[DONE] Chemprop SHAP interpretation complete.")


if __name__ == "__main__":
    main()
