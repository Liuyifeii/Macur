import os
import json
from typing import List, Tuple

import pandas as pd
from rdkit import Chem
import yaml


def load_smiles(csv_path: str, smiles_col: str = "smiles") -> List[str]:
    df = pd.read_csv(csv_path)
    if smiles_col not in df.columns:
        raise ValueError(f"Column '{smiles_col}' not found in {csv_path}")
    smiles_list = df[smiles_col].dropna().astype(str).tolist()
    return smiles_list


def filter_valid_smiles(smiles_list: List[str]) -> List[str]:
    valid = []
    for s in smiles_list:
        try:
            if "." in s or "*" in s or "'" in s:
                continue
            mol = Chem.MolFromSmiles(s)
            if mol is not None:
                valid.append(s)
        except Exception:
            continue
    return valid


def build_char_vocab(smiles_list: List[str]) -> Tuple[dict, dict]:
    charset = set()
    for s in smiles_list:
        for ch in s:
            charset.add(ch)
    chars = sorted(list(charset))
    for forbidden in [".", "'", "*"]:
        if forbidden in chars:
            chars.remove(forbidden)
    token2id = {ch: idx for idx, ch in enumerate(chars)}
    id2token = {idx: ch for ch, idx in token2id.items()}
    return token2id, id2token


def smiles_to_selfies_list(smiles_list: List[str]) -> List[str]:
    import selfies as sf
    selfies_list: List[str] = []
    for s in smiles_list:
        try:
            x = sf.encoder(s)
            if isinstance(x, str) and x:
                selfies_list.append(x)
        except Exception:
            continue
    return selfies_list


def build_selfies_vocab(selfies_list: List[str]) -> Tuple[dict, dict]:
    tokens = set()
    for seq in selfies_list:
        i = 0
        n = len(seq)
        while i < n:
            if seq[i] == "[":
                j = seq.find("]", i + 1)
                if j == -1:
                    break
                tokens.add(seq[i:j+1])
                i = j + 1
            else:
                tokens.add(seq[i])
                i += 1
    toks = sorted(list(tokens))
    token2id = {t: idx for idx, t in enumerate(toks)}
    id2token = {idx: t for t, idx in token2id.items()}
    return token2id, id2token


def save_vocab(token2id: dict, save_path: str, mode: str = "char"):
    dir_name = os.path.dirname(save_path)
    if dir_name:  # Only attempt to create directory if dir_name is not empty
        os.makedirs(dir_name, exist_ok=True)
    payload = dict(token2id)
    payload["__meta__"] = {"mode": mode}
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="YAML config file to override CLI arguments")
    parser.add_argument("--csv", type=str, default="data/raw/smiles_train.csv")
    parser.add_argument("--smiles_col", type=str, default="smiles")
    parser.add_argument("--out_vocab", type=str, default="data/artifacts/vocab_smiles.json")
    parser.add_argument("--out_clean", type=str, default="data/processed/clean_smiles.txt")
    parser.add_argument("--use_selfies", action="store_true", help="Convert SMILES to SELFIES and build the corresponding vocabulary")
    args = parser.parse_args()

    if args.config and os.path.exists(args.config):
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        for k, v in cfg.items():
            if hasattr(args, k):
                setattr(args, k, v)

    os.makedirs(os.path.dirname(args.out_vocab), exist_ok=True)
    os.makedirs(os.path.dirname(args.out_clean), exist_ok=True)

    smiles = load_smiles(args.csv, args.smiles_col)
    smiles = filter_valid_smiles(smiles)

    if args.use_selfies:
        selfies_list = smiles_to_selfies_list(smiles)
        token2id, _ = build_selfies_vocab(selfies_list)
        save_vocab(token2id, args.out_vocab, mode="selfies")
        with open(args.out_clean, "w", encoding="utf-8") as f:
            for s in selfies_list:
                f.write(s + "\n")
        print(f"Saved SELFIES vocab to {args.out_vocab}, size={len(token2id)}")
        print(f"Saved cleaned SELFIES to {args.out_clean}, n={len(selfies_list)}")
    else:
        token2id, _ = build_char_vocab(smiles)
        save_vocab(token2id, args.out_vocab, mode="char")
        with open(args.out_clean, "w", encoding="utf-8") as f:
            for s in smiles:
                f.write(s + "\n")
        print(f"Saved vocab to {args.out_vocab}, size={len(token2id)}")
        print(f"Saved cleaned SMILES to {args.out_clean}, n={len(smiles)}")


if __name__ == "__main__":
    main()


