# GR00T N1.6 Fine-tuning for AgileX Piper Robot

This folder contains everything needed to fine-tune NVIDIA GR00T N1.6 on your collected pick-and-place demonstrations.

## Quick Start

```bash
# 1. Setup environment (run once)
./setup_gr00t.sh

# 2. Train the model
./train_gr00t.sh

# 3. Evaluate on physical robot
python evaluate_gr00t.py --checkpoint checkpoints/piper_finetune
```

## Files

| File | Description |
|------|-------------|
| `setup_gr00t.sh` | One-time environment setup |
| `train_gr00t.sh` | Fine-tuning script |
| `evaluate_gr00t.py` | Run model on physical robot |
| `piper_modality.json` | Robot state/action config |
| `Isaac-GR00T/` | NVIDIA gr00t repository |

## Requirements

- **GPU**: RTX 4090, A6000, or better (24GB+ VRAM recommended)
- **CUDA**: 12.4 recommended
- **Python**: 3.10

## Dataset

Training uses your collected demonstrations at:
```
../datasets/pick_the_white_cup_and_place_it_on_the_red_cup/
```

## Training Time

- ~2000 steps recommended
- ~1-2 hours on RTX 4090
- Checkpoints saved every 500 steps

## Evaluation

After training, run on the physical robot:
```bash
python evaluate_gr00t.py --checkpoint checkpoints/piper_finetune --task "pick the white cup and place it on the red cup"
```
