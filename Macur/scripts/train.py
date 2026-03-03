import os
import json
from collections import deque
from typing import List, Dict

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn.functional as F
from transformers import PreTrainedTokenizer
from transformers import LogitsProcessor, LogitsProcessorList
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from rdkit import Chem
import yaml

# Ensure src/ is on sys.path so 'macur' can be imported without installation
import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from macur.tokenization import WrappedTokenizer
from macur.reward import reward_vector, pareto_reward, is_valid_molecule, max_ring_size, soft_constraints_score


VOCAB_FILE = "data/artifacts/vocab_smiles.json"
PRETRAIN_CHECKPOINT = "outputs/checkpoints/final"
CSV_FILE = "data/raw/smiles_train.csv"
SAVE_DIR_ROOT = "outputs/ppo_curriculum_checkpoints"

KL_COEF_INIT = 0.3
TARGET_KL = 0.02

GEN_MAX_LENGTH = 128
TOP_K = 50
TOP_P = 0.95
BASE_TEMPERATURE = 1.2
STEPS_PER_STAGE = 1000


def get_base_lm(m):
    for attr in ("pretrained_model", "model", "base_model", "transformer", "lm_model", "gpt2"):
        if hasattr(m, attr):
            return getattr(m, attr)
    return m


@torch.no_grad()
def compute_logp_entropy(base_lm, input_ids, attention_mask, action_ids):
    out = base_lm(input_ids=input_ids, attention_mask=attention_mask)
    logits = out.logits[:, :-1, :]
    logprobs = F.log_softmax(logits, dim=-1)
    probs = logprobs.exp()
    logp_taken = logprobs.gather(dim=-1, index=action_ids.unsqueeze(-1)).squeeze(-1)
    entropy_t = -(probs * logprobs).sum(dim=-1)
    entropy = entropy_t.mean(dim=-1)
    return logp_taken, entropy


def adaptive_temperature(stage_idx: int):
    return max(0.9, BASE_TEMPERATURE - 0.1 * stage_idx)


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
            for ch in ["#", "@", "/", "\\"]:
                tid = self.char2id.get(ch)
                if tid is not None:
                    self.stage1_forbidden_ids.add(tid)
            self.allowed_in_bracket = set()
            for ch in ["B","C","N","O","F","P","S","Cl","Br","I","H","l","r","+","-","0"]:
                tid = self.char2id.get(ch)
                if tid is not None:
                    self.allowed_in_bracket.add(tid)
            if self.right_bracket_id is not None:
                self.allowed_in_bracket.add(self.right_bracket_id)
        elif stage_name == "stage_2":
            self.soft_penalize_ids = []
            for ch in ["@", "/", "\\"]:
                tid = self.char2id.get(ch)
                if tid is not None:
                    self.soft_penalize_ids.append(tid)
        else:
            self.soft_penalize_ids = []

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
        else:
            pass
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
                if left_bracket > 0 and self.allowed_in_bracket:
                    scores_row = scores[b]
                    allowed = torch.full_like(scores_row, -float("inf"))
                    allowed[list(self.allowed_in_bracket)] = scores_row[list(self.allowed_in_bracket)]
                    scores[b] = allowed
            elif self.stage_name == "stage_2" and getattr(self, "soft_penalize_ids", None):
                for tid in self.soft_penalize_ids:
                    scores[b, tid] = scores[b, tid] - 1.0

        return scores


def build_logits_processor(tokenizer: WrappedTokenizer, stage_name: str) -> LogitsProcessorList:
    processors = []
    processors.append(MaxTokenRepeatProcessor(max_repeat=4))

    if stage_name == "stage_1":
        processors.append(SmilesSyntaxProcessor(tokenizer, stage_name))
    else:
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


def sample_and_score(model, tokenizer, batch_size: int, generate_kwargs: Dict):
    device = next(model.parameters()).device
    new_queries, new_responses, gen_texts = [], [], []
    empty_count = 0
    attempts = 0
    max_attempts = batch_size * 6
    while len(new_queries) < batch_size and attempts < max_attempts:
        attempts += 1
        input_ids = torch.tensor([[tokenizer.bos_token_id]], device=device)
        try:
            with torch.no_grad():
                gen_output = model.generate(input_ids=input_ids, min_new_tokens=1, **generate_kwargs)
        except Exception:
            empty_count += 1
            continue
        gen_ids = gen_output[0].tolist()
        response_ids = gen_ids[1:]
        if tokenizer.eos_token_id in response_ids:
            eos_pos = response_ids.index(tokenizer.eos_token_id)
            response_ids = response_ids[:eos_pos]
        if len(response_ids) == 0:
            empty_count += 1
            continue
        gen_text = tokenizer.decode_to_smiles(response_ids)
        if not gen_text:
            empty_count += 1
            continue
        new_queries.append(torch.tensor([tokenizer.bos_token_id], dtype=torch.long))
        new_responses.append(torch.tensor(response_ids, dtype=torch.long))
        gen_texts.append(gen_text)
    if empty_count > 0 and len(new_queries) == 0:
        new_queries = [torch.tensor([tokenizer.bos_token_id], dtype=torch.long)]
        new_responses = [torch.tensor([tokenizer.eos_token_id], dtype=torch.long)]
        gen_texts = [""]
    return new_queries, new_responses, gen_texts


def run_stage(stage_name: str, epochs: int, ppo_trainer: PPOTrainer, tokenizer: WrappedTokenizer, baseline_vectors_holder: List[np.ndarray], novelty_memory: List[set]):
    device = next(ppo_trainer.model.parameters()).device
    vocab = {}
    try:
        vocab = tokenizer.get_vocab()
    except Exception:
        vocab = getattr(tokenizer, "token2id", {}) or {}
    bad_ids = []
    for ch in [".", "'", "*"]:
        if ch in vocab:
            bad_ids.append(vocab[ch])
    if hasattr(tokenizer, "unk_token_id") and tokenizer.unk_token_id is not None:
        bad_ids.append(int(tokenizer.unk_token_id))
    bad_words_ids = [[int(tid)] for tid in sorted(set(bad_ids))] if bad_ids else None
    generate_kwargs = {
        "max_length": GEN_MAX_LENGTH,
        "do_sample": True,
        "top_k": TOP_K,
        "top_p": TOP_P,
        "temperature": BASE_TEMPERATURE,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
        "bos_token_id": tokenizer.bos_token_id,
        "num_return_sequences": 1,
        "repetition_penalty": 1.2,
        "no_repeat_ngram_size": 3,
        "renormalize_logits": True,
        "remove_invalid_values": True,
        **({"bad_words_ids": bad_words_ids} if bad_words_ids is not None else {}),
        "logits_processor": build_logits_processor(tokenizer, stage_name),
    }

    if stage_name == "stage_1":
        stage_min_ring = 6
    elif stage_name == "stage_2":
        stage_min_ring = 10
    elif stage_name == "stage_3":
        stage_min_ring = 12
    else:
        stage_min_ring = 12

    novelty_weight = 0.2
    diversity_weight = 0.05

    stage_seen = set()

    for ep in range(epochs):
        generate_kwargs["temperature"] = adaptive_temperature(ep)
        if stage_name in ("stage_2", "stage_3"):
            generate_kwargs["temperature"] = max(generate_kwargs.get("temperature", 1.0), 1.3)
            generate_kwargs["top_k"] = max(int(generate_kwargs.get("top_k", TOP_K)), 100)
            generate_kwargs["no_repeat_ngram_size"] = max(int(generate_kwargs.get("no_repeat_ngram_size", 3)), 4)
        running = {"reward": [], "qed": [], "sa": [], "macro": [], "flex": [], "valid": [], "unique": []}
        epoch_vectors = []
        epoch_smiles = []
        epoch_unique_set = set()

        for _ in tqdm(range(STEPS_PER_STAGE)):
            new_queries, new_responses, gen_texts = sample_and_score(ppo_trainer.model, tokenizer, ppo_trainer.config.batch_size, generate_kwargs)
            
            vectors = [reward_vector(s, min_ring_size=stage_min_ring) for s in gen_texts]

            novelty_bonus = []
            for s in gen_texts:
                try:
                    mol = Chem.MolFromSmiles(s)
                    can = Chem.MolToSmiles(mol, isomericSmiles=True) if mol else ""
                except Exception:
                    can = ""
                if can and (can not in stage_seen):
                    novelty_bonus.append(1.0)
                    epoch_unique_set.add(can)
                    stage_seen.add(can)
                else:
                    novelty_bonus.append(0.0)
            N = len(gen_texts)
            diversity_bonus = [0.0] * N
            fps = []
            idx_map = []
            for i, s in enumerate(gen_texts):
                try:
                    mol = Chem.MolFromSmiles(s)
                    if mol is None:
                        continue
                    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
                    fps.append(fp)
                    idx_map.append(i)
                except Exception:
                    continue
            if len(fps) >= 2:
                for a in range(len(fps)):
                    i = idx_map[a]
                    sim_sum = 0.0
                    cnt = 0
                    for b in range(len(fps)):
                        if a == b:
                            continue
                        try:
                            sim = DataStructs.TanimotoSimilarity(fps[a], fps[b])
                            sim_sum += (1.0 - sim)
                            cnt += 1
                        except Exception:
                            continue
                    diversity_bonus[i] = (sim_sum / cnt) if cnt > 0 else 0.0

            if stage_name == "stage_1":
                shaped = np.array([1.0 if is_valid_molecule(s) else 0.0 for s in gen_texts], dtype=float)
            elif stage_name == "stage_2":
                reward_vec_np = np.stack(vectors)
                macro = reward_vec_np[:, 2]
                flex = reward_vec_np[:, 3]
                qed = reward_vec_np[:, 0]
                sa = reward_vec_np[:, 1]
                ring_sizes = []
                for s in gen_texts:
                    try:
                        rs = max_ring_size(s)
                    except Exception:
                        rs = 0
                    ring_sizes.append(min(max(rs, 0), 16) / 16.0)
                ring_sizes = np.array(ring_sizes, dtype=float)
                shaped = 1.2 * macro + 0.6 * flex + 0.1 * qed + 0.1 * sa + 0.8 * ring_sizes
            elif stage_name == "stage_3":
                reward_vec_np = np.stack(vectors)
                ring_sizes = []
                for s in gen_texts:
                    try:
                        rs = max_ring_size(s)
                    except Exception:
                        rs = 0
                    ring_sizes.append(min(max(rs, 0), 16) / 16.0)
                ring_sizes = np.array(ring_sizes, dtype=float)
                shaped = pareto_reward(reward_vec_np, baseline_vectors_holder[0])
                shaped = shaped + 0.6 * reward_vec_np[:, 2] + 0.2 * reward_vec_np[:, 3] + 0.4 * ring_sizes
                # Soft downstream constraints (keep the effect small; do not dominate Pareto shaping)
                constraint_w = 0.05
                c_scores = np.array(
                    [soft_constraints_score(s, min_ring_size=stage_min_ring) for s in gen_texts],
                    dtype=float,
                )
                shaped = shaped + constraint_w * c_scores
            elif stage_name == "stage_4":
                reward_vec_np = np.stack(vectors)
                ring_sizes = []
                for s in gen_texts:
                    try:
                        rs = max_ring_size(s)
                    except Exception:
                        rs = 0
                    ring_sizes.append(min(max(rs, 0), 16) / 16.0)
                ring_sizes = np.array(ring_sizes, dtype=float)
                shaped = pareto_reward(reward_vec_np, baseline_vectors_holder[0])
                shaped = shaped + 0.4 * reward_vec_np[:, 2] + 0.15 * reward_vec_np[:, 3] + 0.3 * ring_sizes
            elif stage_name == "stage_5":
                vectors = [v if is_valid_molecule(s) else np.array([-1, -1, -1, -1]) for v, s in zip(vectors, gen_texts)]
                reward_vec_np = np.stack(vectors)
                shaped = pareto_reward(reward_vec_np, baseline_vectors_holder[0])
                macro_col = reward_vec_np[:, 2]
                ring_sizes = []
                for s in gen_texts:
                    try:
                        rs = max_ring_size(s)
                    except Exception:
                        rs = 0
                    ring_sizes.append(min(max(rs, 0), 16) / 16.0)
                ring_sizes = np.array(ring_sizes, dtype=float)
                macro_scale = 0.1 + 0.9 * macro_col
                shaped = shaped * macro_scale + 0.3 * ring_sizes
            else:
                reward_vec_np = np.stack(vectors)
                shaped = pareto_reward(reward_vec_np, baseline_vectors_holder[0])
            if stage_name == "stage_1":
                shaped = shaped + 0.05 * np.array(novelty_bonus, dtype=float) + 0.02 * np.array(diversity_bonus, dtype=float)
            else:
                shaped = shaped + 0.2 * np.array(novelty_bonus, dtype=float) + 0.10 * np.array(diversity_bonus, dtype=float)
            rewards = [torch.tensor(r, dtype=torch.float32, device=device) for r in shaped]

            try:
                ppo_trainer.step(new_queries, new_responses, rewards)
            except Exception as e:
                print("PPO step error:", e)
                continue

            for s, v, r in zip(gen_texts, vectors, shaped):
                valid = is_valid_molecule(s)
                running["valid"].append(1 if valid else 0)
                if valid:
                    epoch_smiles.append(s)
                    epoch_vectors.append(v)
                running["reward"].append(r)
                running["qed"].append(v[0])
                running["sa"].append(v[1])
                running["macro"].append(v[2])
                running["flex"].append(v[3])

        baseline_vectors_holder[0].extend(epoch_vectors)
        if len(baseline_vectors_holder[0]) > 10000:
            baseline_vectors_holder[0] = baseline_vectors_holder[0][-10000:]

        novelty_memory[0].update(epoch_unique_set)

        if running["reward"]:
            valid_ratio = (np.sum(running["valid"]) / len(running["valid"])) if running["valid"] else 0.0
            unique_count = len(epoch_unique_set)
            print(f"[{stage_name}] Ep Stats: Reward {np.mean(running['reward']):.4f} | QED {np.mean(running['qed']):.4f} | SA {np.mean(running['sa']):.4f} | Macro {np.mean(running['macro']):.4f} | Flex {np.mean(running['flex']):.4f} | Valid {valid_ratio*100:.1f}% | Unique {unique_count}")

            if valid_ratio < 0.2:
                generate_kwargs["temperature"] = max(generate_kwargs.get("temperature", 1.0), 1.0)
                generate_kwargs["no_repeat_ngram_size"] = min(generate_kwargs.get("no_repeat_ngram_size", 3) + 1, 5)
                generate_kwargs.pop("bad_words_ids", None)
                if hasattr(ppo_trainer, "config"):
                    ppo_trainer.config.init_kl_coef = float(ppo_trainer.config.init_kl_coef) * 1.5


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="YAML config file to override default hyperparameters")
    args_cli = parser.parse_args()
    global STEPS_PER_STAGE, GEN_MAX_LENGTH, TOP_K, TOP_P, BASE_TEMPERATURE, SAVE_DIR_ROOT
    if args_cli.config and os.path.exists(args_cli.config):
        with open(args_cli.config, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    else:
        cfg = {}

    vocab_file = cfg.get("vocab_file", VOCAB_FILE)
    pretrain_ckpt = cfg.get("pretrain_checkpoint", PRETRAIN_CHECKPOINT)
    save_dir_root = cfg.get("save_dir_root", SAVE_DIR_ROOT)
    steps_per_stage_cfg = int(cfg.get("steps_per_stage", STEPS_PER_STAGE))
    batch_size_cfg = int(cfg.get("batch_size", 8))
    mini_batch_size_cfg = int(cfg.get("mini_batch_size", 4))
    learning_rate_cfg = float(cfg.get("learning_rate", 1e-5))
    init_kl_coef_cfg = float(cfg.get("init_kl_coef", KL_COEF_INIT))
    target_kl_cfg = float(cfg.get("target_kl", TARGET_KL))
    gen_max_length_cfg = int(cfg.get("gen_max_length", GEN_MAX_LENGTH))
    top_k_cfg = int(cfg.get("top_k", TOP_K))
    top_p_cfg = float(cfg.get("top_p", TOP_P))
    base_temperature_cfg = float(cfg.get("base_temperature", BASE_TEMPERATURE))

    STEPS_PER_STAGE = steps_per_stage_cfg
    GEN_MAX_LENGTH = gen_max_length_cfg
    TOP_K = top_k_cfg
    TOP_P = top_p_cfg
    BASE_TEMPERATURE = base_temperature_cfg
    SAVE_DIR_ROOT = save_dir_root

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    tokenizer = WrappedTokenizer(vocab_file, model_max_length=512)
    print("Vocab size:", tokenizer.vocab_size)

    model = AutoModelForCausalLMWithValueHead.from_pretrained(pretrain_ckpt)
    if hasattr(model, "pretrained_model") and hasattr(model.pretrained_model, "resize_token_embeddings"):
        model.pretrained_model.resize_token_embeddings(tokenizer.vocab_size)
    model.to(device)

    ppo_config = PPOConfig(
        model_name=pretrain_ckpt,
        learning_rate=learning_rate_cfg,
        batch_size=batch_size_cfg,
        mini_batch_size=mini_batch_size_cfg,
        gradient_accumulation_steps=1,
        init_kl_coef=init_kl_coef_cfg,
        target_kl=target_kl_cfg,
    )

    trainer = PPOTrainer(
        config=ppo_config,
        model=model,
        tokenizer=tokenizer,
    )

    baseline_vectors_holder = [[]]
    novelty_memory = [set()]

    curriculum = [
        ("stage_1", 1),
        ("stage_2", 1),
        ("stage_3", 1),
        ("stage_4", 1),
        ("stage_5", 1),
    ]

    os.makedirs(SAVE_DIR_ROOT, exist_ok=True)

    for stage_name, epochs in curriculum:
        print(f"=== Running {stage_name} for {epochs} epochs ===")
        run_stage(stage_name, epochs, trainer, tokenizer, baseline_vectors_holder, novelty_memory)

        save_dir = os.path.join(SAVE_DIR_ROOT, stage_name)
        os.makedirs(save_dir, exist_ok=True)
        try:
            trainer.model.save_pretrained(save_dir)
            tokenizer.save_vocabulary(save_dir)
        except Exception:
            torch.save(trainer.model.state_dict(), os.path.join(save_dir, "pytorch_model_state_dict.bin"))
            tokenizer.save_vocabulary(save_dir)

    print("Training finished with curriculum.")


if __name__ == "__main__":
    main()


