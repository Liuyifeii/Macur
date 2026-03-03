#!/bin/bash
#SBATCH --job-name=train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=15-00:00:00
#SBATCH --output=log_%j.out
#SBATCH --error=err_%j.err
 
source ~/.bashrc
set -euo pipefail

# Minimal, robust setup: use absolute paths (adjust /home/liuyifei/Macur if different)
export PYTHONUNBUFFERED=1
export PYTHONPATH="/home/liuyifei/Macur/src:${PYTHONPATH:-}"
#python scripts/generate_from_pretrain.py --model_dir outputs/checkpoints/final --num 5000 --min_ring_size 12 --out_dir outputs/samples
#python3 scripts/generate.py --config configs/generate.yaml
#python3 scripts/ppo_train_curriculum.py --config configs/ppo.yaml
python scripts/generate_maro.py --config configs/generate.yaml --min_ring_size 12
#python scripts/generate.py --config configs/generate.yaml --novelty_guided --pool_size 64 --select_k 8 --novelty_weight 1.0 --diversity_weight 0.5 --macro_weight 0.5 --min_ring_size 12