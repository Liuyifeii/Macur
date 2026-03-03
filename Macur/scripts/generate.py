# generate.py
import os
import json
import torch
import pandas as pd
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import QED
from rdkit.Chem import rdMolDescriptors

from transformers import GPT2LMHeadModel
from trl import AutoModelForCausalLMWithValueHead
from transformers import LogitsProcessor, LogitsProcessorList
# Ensure src/ is on sys.path so 'macur' can be imported without installation
import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
from macur.tokenization import WrappedTokenizer
from macur import sascorer
import yaml

class StaticBanProcessor(LogitsProcessor):
    def __init__(self, banned_token_ids):
        self.banned = set(int(t) for t in banned_token_ids)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if not self.banned:
            return scores
        for tid in self.banned:
            scores[:, tid] = -float("inf")
        return scores


class MaxTokenRepeatProcessor(LogitsProcessor):
    def __init__(self, max_repeat: int = 4):
        self.max_repeat = int(max_repeat)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if input_ids.size(1) == 0:
            return scores
        batch_size = input_ids.size(0)
        for b in range(batch_size):
            seq = input_ids[b]
            last_id = seq[-1].item()
            run_len = 1
            for t in range(seq.size(0) - 2, -1, -1):
                if seq[t].item() == last_id:
                    run_len += 1
                    if run_len > self.max_repeat:
                        break
                else:
                    break
            if run_len > self.max_repeat:
                scores[b, last_id] = -float("inf")
        return scores


class SmilesSyntaxProcessor(LogitsProcessor):
    def __init__(self, tokenizer: WrappedTokenizer, stage_name: str):
        self.tokenizer = tokenizer
        self.stage_name = stage_name
        try:
            self.vocab = tokenizer.get_vocab()
        except Exception:
            self.vocab = getattr(tokenizer, "token2id", {}) or {}
        self.id_to_token = getattr(tokenizer, "id2token", None)
        self.char2id = {ch: int(self.vocab[ch]) for ch in self.vocab if isinstance(ch, str)}

        self.right_paren_id = self.char2id.get(")")
        self.left_paren_id = self.char2id.get("(")
        self.right_bracket_id = self.char2id.get("]")
        self.left_bracket_id = self.char2id.get("[")

        self.stage1_forbidden_ids = set()
        if stage_name == "stage_1":
            for ch in ["#"] + [str(d) for d in range(1, 10)]:
                tid = self.char2id.get(ch)
                if tid is not None:
                    self.stage1_forbidden_ids.add(tid)
            self.allowed_in_bracket = set()
            for ch in ["B","C","N","O","F","P","S","I","H","l","r","@","+","-","0","1","2","3","4","5","6","7","8","9"]:
                tid = self.char2id.get(ch)
                if tid is not None:
                    self.allowed_in_bracket.add(tid)
            if self.right_bracket_id is not None:
                self.allowed_in_bracket.add(self.right_bracket_id)

    def _count_unmatched(self, seq_ids: torch.LongTensor) -> tuple:
        left_paren = 0
        left_bracket = 0
        if self.id_to_token is not None:
            for i in seq_ids.tolist():
                if i == self.left_paren_id:
                    left_paren += 1
                elif i == self.right_paren_id and left_paren > 0:
                    left_paren -= 1
                if i == self.left_bracket_id:
                    left_bracket += 1
                elif i == self.right_bracket_id and left_bracket > 0:
                    left_bracket -= 1
        return left_paren, left_bracket

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        batch_size = input_ids.size(0)
        for b in range(batch_size):
            seq = input_ids[b]
            left_paren, left_bracket = self._count_unmatched(seq)

            if left_paren == 0 and self.right_paren_id is not None:
                scores[b, self.right_paren_id] = -float("inf")
            if left_bracket == 0 and self.right_bracket_id is not None:
                scores[b, self.right_bracket_id] = -float("inf")

            if self.stage_name == "stage_1":
                for tid in self.stage1_forbidden_ids:
                    scores[b, tid] = -float("inf")
                if left_bracket > 0 and hasattr(self, "allowed_in_bracket") and self.allowed_in_bracket:
                    scores_row = scores[b]
                    allowed = torch.full_like(scores_row, -float("inf"))
                    allowed[list(self.allowed_in_bracket)] = scores_row[list(self.allowed_in_bracket)]
                    scores[b] = allowed
        return scores


def build_logits_processor(tokenizer: WrappedTokenizer, stage_name: str) -> LogitsProcessorList:
    processors = []
    processors.append(MaxTokenRepeatProcessor(max_repeat=4))
    processors.append(SmilesSyntaxProcessor(tokenizer, stage_name))
    class FallbackLogitsProcessor(LogitsProcessor):
        def __init__(self, eos_id: int):
            self.eos_id = int(eos_id)
        def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
            scores = scores.clone()
            scores[torch.isnan(scores)] = -1e9
            pos_inf = torch.isinf(scores) & (scores > 0)
            neg_inf = torch.isinf(scores) & (scores < 0)
            scores[pos_inf] = 1e9
            scores[neg_inf] = -1e9
            row_max, _ = scores.max(dim=-1)
            need_fallback = ~torch.isfinite(row_max) | (row_max <= -1e8)
            if need_fallback.any():
                for b in torch.nonzero(need_fallback, as_tuple=False).flatten().tolist():
                    scores[b, :] = -1e9
                    scores[b, self.eos_id] = 0.0
            return scores
    processors.append(FallbackLogitsProcessor(tokenizer.eos_token_id))
    return LogitsProcessorList(processors)


def qed_score(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        return QED.qed(mol) if mol else 0.0
    except:
        return 0.0

def sa_score(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return 0.0
        score = sascorer.calculateScore(mol)
        norm_score = (10. - score) / 9.
        return max(0.0, min(norm_score, 1.0))
    except:
        return 0.0

def is_macrocycle(smiles, min_ring_size=12):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return 0.0
        ri = mol.GetRingInfo()
        for ring in ri.AtomRings():
            if len(ring) >= min_ring_size:
                return 1.0
        return 0.0
    except:
        return 0.0

def max_ring_size(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return 0
        ri = mol.GetRingInfo()
        if not ri.AtomRings():
            return 0
        return max(len(r) for r in ri.AtomRings())
    except:
        return 0

def macrocycle_flexibility_score(smiles, min_ring_size=12):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return 0.0
        ri = mol.GetRingInfo()
        rings = [ring for ring in ri.AtomRings() if len(ring) >= min_ring_size]
        if not rings:
            return 0.0
        largest = max(rings, key=len)
        ring_atom_set = set(largest)

        ring_bonds = []
        for bond in mol.GetBonds():
            a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            if a in ring_atom_set and b in ring_atom_set:
                ring_bonds.append(bond)

        if not ring_bonds:
            return 0.0

        flexible, total = 0, 0
        for bond in ring_bonds:
            total += 1
            if bond.GetBondType() != Chem.BondType.SINGLE:
                continue
            if bond.GetIsAromatic():
                continue
            if bond.GetIsConjugated():
                continue
            flexible += 1
        score = flexible / float(total) if total > 0 else 0.0
        return max(0.0, min(score, 1.0))
    except:
        return 0.0


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="YAML config file, can override command-line arguments")
    parser.add_argument("--vocab", type=str, default="data/artifacts/vocab_smiles.json")
    parser.add_argument("--model_dir", type=str, default="outputs/ppo_curriculum_checkpoints/stage_5")
    parser.add_argument("--stage", type=str, default="stage_5")
    parser.add_argument("--num", type=int, default=1000)
    parser.add_argument("--max_length", type=int, default=200)
    parser.add_argument("--out_dir", type=str, default="outputs/samples")
    args = parser.parse_args()

    if args.config and os.path.exists(args.config):
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        for k, v in cfg.items():
            if hasattr(args, k):
                setattr(args, k, v)

    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = WrappedTokenizer(args.vocab)
    try:
        model = AutoModelForCausalLMWithValueHead.from_pretrained(args.model_dir).to(device)
    except Exception:
        from transformers import GPT2Config, AutoModelForCausalLM
        cfg_path = os.path.join(args.model_dir, "config.json")
        if not os.path.exists(cfg_path):
            raise FileNotFoundError(f"Missing config file: {cfg_path}")
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg_dict = json.load(f)
        config = GPT2Config.from_dict(cfg_dict)
        base_model = AutoModelForCausalLM.from_config(config)
        state_path = os.path.join(args.model_dir, "pytorch_model_state_dict.bin")
        if not os.path.exists(state_path):
            raise FileNotFoundError(f"Missing weight file: {state_path}")
        sd = torch.load(state_path, map_location="cpu")
        base_model.load_state_dict(sd, strict=False)
        model = AutoModelForCausalLMWithValueHead.from_pretrained(args.model_dir, local_files_only=True, ignore_mismatched_sizes=True)
        try:
            model.pretrained_model.load_state_dict(base_model.state_dict(), strict=False)
        except Exception:
            pass
        model = model.to(device)
    model.eval()

    num_samples = args.num
    records = []
    seen_smiles = set()

    print(f"Generating {num_samples} valid SMILES...")

    pbar = tqdm(total=num_samples)
    while len(records) < num_samples:
        input_ids = torch.tensor([[tokenizer.bos_token_id]], dtype=torch.long).to(device)
        with torch.no_grad():
            try:
                vocab = tokenizer.get_vocab()
            except Exception:
                vocab = getattr(tokenizer, "token2id", {}) or {}
            bw = []
            for c in [".", "'", "*"]:
                if c in vocab:
                    bw.append([int(vocab[c])])
            if getattr(tokenizer, "unk_token_id", None) is not None:
                bw.append([int(tokenizer.unk_token_id)])
            bad_words_ids = bw if len(bw) > 0 else None
            output = model.generate(
                input_ids,
                max_length=args.max_length,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                temperature=1.0,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                bos_token_id=tokenizer.bos_token_id,
                repetition_penalty=1.2,
                no_repeat_ngram_size=3,
                renormalize_logits=True,
                remove_invalid_values=True,
                bad_words_ids=bad_words_ids,
                logits_processor=build_logits_processor(tokenizer, args.stage)
            )
        gen_ids = output[0].tolist()
        if tokenizer.bos_token_id in gen_ids:
            gen_ids = gen_ids[1:]
        if tokenizer.eos_token_id in gen_ids:
            eos_pos = gen_ids.index(tokenizer.eos_token_id)
            gen_ids = gen_ids[:eos_pos]
        smi = tokenizer.decode_to_smiles(gen_ids)

        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        try:
            can_smi = Chem.MolToSmiles(mol, isomericSmiles=True)
        except Exception:
            continue
        if can_smi in seen_smiles:
            continue
        seen_smiles.add(can_smi)

        rec = {
            "smiles": can_smi,
            "qed": qed_score(smi),
            "sa": sa_score(smi),
            "macro": is_macrocycle(smi),
            "max_ring": max_ring_size(smi),
            "flex": macrocycle_flexibility_score(smi)
        }
        records.append(rec)
        pbar.update(1)

    pbar.close()

    df = pd.DataFrame(records)
    out_csv = os.path.join(args.out_dir, f"out-{num_samples}.csv")
    df.to_csv(out_csv, index=False)

    print(f"Generation complete, saved to {out_csv}")
    print("Average QED:", df["qed"].mean())
    print("Average SA:", df["sa"].mean())
    print("Average Macro:", df["macro"].mean())
    print("Average Max Ring:", df["max_ring"].mean())
    print("Average Flex:", df["flex"].mean())


if __name__ == "__main__":
    main()


