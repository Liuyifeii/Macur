### Setup

```bash
pip install -r requirements.txt
```

### Quickstart

1) Curriculum training:

```bash
python scripts/train.py --config configs/train.yaml
```

2) Sample & evaluate:

```bash
python scripts/generate.py \
  --config configs/generate.yaml \
  --vocab data/artifacts/vocab_smiles.json \
  --model_dir outputs/ppo_curriculum_checkpoints/final \
  --num 1000 --max_length 200 --out_dir outputs/samples
```
3) Continue training from `stage_5` with a custom reward
Users can build upon the final trained model from Stage 5 and customize the reward function for downstream optimization (e.g., incorporating docking scores).
