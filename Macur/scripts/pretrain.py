import os
from typing import List

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoConfig, GPT2Config, AutoModelForCausalLM, Trainer, TrainingArguments
import yaml

# Ensure src/ is on sys.path so 'macur' can be imported without installation
import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from macur.tokenization import WrappedTokenizer


class SmilesDataset(Dataset):
    def __init__(self, smiles_paths: List[str], tokenizer: WrappedTokenizer, block_size: int = 128):
        self.samples = []
        for p in smiles_paths:
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    s = line.strip()
                    if not s:
                        continue
                    ids = tokenizer.encode(s, add_special_tokens=True)
                    if len(ids) > block_size:
                        ids = ids[:block_size - 1] + [tokenizer.eos_token_id]
                    self.samples.append(torch.tensor(ids, dtype=torch.long))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x = self.samples[idx]
        return {"input_ids": x, "labels": x.clone()}


def build_model(tokenizer: WrappedTokenizer, n_layer: int = 8, n_head: int = 8, n_embd: int = 512):
    config = GPT2Config(
        vocab_size=tokenizer.vocab_size,
        n_positions=512,
        n_ctx=512,
        n_embd=n_embd,
        n_layer=n_layer,
        n_head=n_head,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        n_inner=4 * n_embd,
        pad_token_id=tokenizer.pad_token_id,
    )
    model = AutoModelForCausalLM.from_config(config)
    return model


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="YAML config file to override CLI arguments")
    parser.add_argument("--vocab", type=str, default="data/artifacts/vocab_smiles.json")
    parser.add_argument("--clean_smiles", type=str, default="data/processed/clean_smiles.txt")
    parser.add_argument("--save_dir", type=str, default="outputs/checkpoints")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--block_size", type=int, default=128)
    parser.add_argument("--save_total_limit", type=int, default=10)
    args = parser.parse_args()

    if args.config and os.path.exists(args.config):
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        for k, v in cfg.items():
            if hasattr(args, k):
                setattr(args, k, v)
    try:
        args.epochs = int(args.epochs)
    except Exception:
        args.epochs = int(float(args.epochs))
    try:
        args.batch = int(args.batch)
    except Exception:
        args.batch = int(float(args.batch))
    try:
        args.lr = float(args.lr)
    except Exception:
        args.lr = float(str(args.lr).replace(" ", ""))
    try:
        args.block_size = int(args.block_size)
    except Exception:
        args.block_size = int(float(args.block_size))
    try:
        args.save_total_limit = int(args.save_total_limit)
    except Exception:
        args.save_total_limit = int(float(args.save_total_limit))

    os.makedirs(args.save_dir, exist_ok=True)

    tokenizer = WrappedTokenizer(args.vocab)

    dataset = SmilesDataset([args.clean_smiles], tokenizer, block_size=args.block_size)

    model = build_model(tokenizer)

    def data_collator(features):
        max_len = max(len(f["input_ids"]) for f in features)
        input_ids = []
        labels = []
        attention_masks = []
        for f in features:
            ids = f["input_ids"]
            pad_len = max_len - len(ids)
            if pad_len > 0:
                ids = torch.cat([ids, torch.full((pad_len,), tokenizer.pad_token_id, dtype=torch.long)])
            input_ids.append(ids)
            labels.append(ids.clone())
            attention_masks.append(torch.ones_like(ids, dtype=torch.long))
        input_ids = torch.stack(input_ids)
        labels = torch.stack(labels)
        labels[labels == tokenizer.pad_token_id] = -100
        return {"input_ids": input_ids, "labels": labels, "attention_mask": (input_ids != tokenizer.pad_token_id).long()}

    training_args = TrainingArguments(
        output_dir=args.save_dir,
        per_device_train_batch_size=args.batch,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        optim="adamw_torch",
        logging_steps=50,
        save_steps=500,
        save_total_limit=args.save_total_limit,
        evaluation_strategy="no",
        remove_unused_columns=False,
        fp16=torch.cuda.is_available(),
        dataloader_pin_memory=True,
        gradient_checkpointing=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    trainer.train()

    last_dir = os.path.join(args.save_dir, "final")
    os.makedirs(last_dir, exist_ok=True)
    trainer.save_model(last_dir)
    tokenizer.save_vocabulary(last_dir)


if __name__ == "__main__":
    main()


