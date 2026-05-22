# GPU Computing Guide: Running Reasoning on a Manifold

This guide covers every realistic option for running this project with GPU access, from free tiers to institutional resources.

---

## What You Actually Need

The 1.5B model is small. Your GPU requirements are modest:

| Phase | GPU Memory | Time (1.5B) | Can use free tier? |
|-------|-----------|-------------|-------------------|
| Generate 1000 chains | ~4 GB | ~2-3 hours | Yes |
| Extract activations | ~5 GB | ~1-2 hours | Yes |
| PCA analysis | CPU only | ~5 minutes | Yes (no GPU) |
| Steered generation | ~4 GB | ~3-4 hours | Yes |
| **Total** | **~5 GB peak** | **~8 hours** | **Yes** |

The 7B model needs ~16 GB VRAM (A100/L4 territory). Start with 1.5B.

---

## Option 1: Google Colab (Free — Recommended Starting Point)

**What you get:** Free T4 GPU (16 GB VRAM), 12-hour session limit, ~80 GB disk.

**Constraints:** Sessions disconnect after ~90 minutes of inactivity. The 12-hour limit means you can run the full 1.5B pipeline in one session if you don't idle. Save checkpoints to Google Drive.

### Setup

1. Go to [colab.research.google.com](https://colab.research.google.com)
2. Create a new notebook
3. Runtime → Change runtime type → T4 GPU
4. Run the setup cells below

### Colab Cells

**Cell 1 — Install dependencies and clone project:**
```python
# Mount Google Drive for persistent storage
from google.colab import drive
drive.mount('/content/drive')

# Create project workspace
!mkdir -p /content/drive/MyDrive/reasoning-on-manifold

# Install dependencies (Colab already has torch + transformers)
!pip install -q accelerate bitsandbytes einops openai datasets tqdm wandb

# Clone Venhoff's repo for reference
!git clone -q https://github.com/cvenhoff/steering-thinking-llms.git /content/venhoff

# Upload and unzip the project files (upload reasoning-on-manifold-phase1.zip first)
# OR copy from Drive if you've saved it there:
# !cp /content/drive/MyDrive/reasoning-on-manifold-phase1.zip /content/
# !unzip -q reasoning-on-manifold-phase1.zip

# Verify GPU
import torch
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
```

**Cell 2 — Load model:**
```python
import sys
sys.path.insert(0, '/content/reasoning-on-manifold')

from src.utils.model_utils import load_model_and_tokenizer, print_model_info

model, tokenizer = load_model_and_tokenizer(
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    dtype="float16",
)
print_model_info(model, tokenizer)
```

**Cell 3 — Generate chains (with Drive checkpointing):**
```python
from pathlib import Path
from src.data.generate_chains import generate_chains_for_tasks

# Load or create tasks
import json
tasks_path = Path("/content/drive/MyDrive/reasoning-on-manifold/all_tasks.json")

if tasks_path.exists():
    with open(tasks_path) as f:
        tasks = json.load(f)
    print(f"Loaded {len(tasks)} existing tasks")
else:
    # You'll need to generate tasks first (see Cell 3a below)
    pass

# Generate chains — checkpoints to Drive so you survive disconnections
chains = generate_chains_for_tasks(
    model=model,
    tokenizer=tokenizer,
    tasks=tasks,
    max_tokens=1000,
    temperature=0.0,
    save_path=Path("/content/drive/MyDrive/reasoning-on-manifold/chains_R1-1.5B.json"),
    checkpoint_every=25,  # Save frequently on Colab
)
```

**Cell 3a — Generate tasks (requires OpenAI API key):**
```python
import os
os.environ["OPENAI_API_KEY"] = "sk-..."  # Your key here

from src.data.generate_tasks import generate_all_tasks
from src.utils.config import load_config

config = load_config("/content/reasoning-on-manifold/configs/experiment_config.yaml")

tasks = generate_all_tasks(
    categories=config["data"]["categories"],
    n_per_category=config["data"]["tasks_per_category"],
    save_dir=Path("/content/drive/MyDrive/reasoning-on-manifold/"),
)
```

**Cell 4 — Annotate chains:**
```python
import os
os.environ["OPENAI_API_KEY"] = "sk-..."  # If not already set

from src.data.annotate_chains import annotate_chains

annotated = annotate_chains(
    chains=chains,
    save_path=Path("/content/drive/MyDrive/reasoning-on-manifold/annotated_R1-1.5B.json"),
    annotation_model="gpt-4o",
    checkpoint_every=10,  # Frequent checkpoints for Colab
)
```

### Colab Pro ($12/month)

Upgrades you to: A100 (40 GB) or L4 (24 GB), 24-hour sessions, priority GPU access. Worth it if free-tier T4 queues are long. The A100 lets you run the 7B model.

---

## Option 2: Kaggle Notebooks (Free)

**What you get:** 2× T4 GPUs (16 GB each), 30 hours/week GPU quota, persistent storage.

**Advantages over Colab:** More generous weekly GPU hours (30 vs ~12 per session). Persistent datasets between sessions. No random disconnections.

**Constraints:** Notebooks are slower to start. Internet access must be enabled manually. 20 GB disk per notebook output.

### Setup

1. Go to [kaggle.com/code](https://www.kaggle.com/code)
2. New Notebook → Settings → Accelerator → GPU T4 x2
3. Settings → Internet → On

```python
# Install missing dependencies
!pip install -q openai accelerate bitsandbytes einops

# Upload your project zip as a Kaggle dataset, then:
!cp -r /kaggle/input/reasoning-on-manifold/* /kaggle/working/
```

The rest of the code is identical to the Colab cells above, just change paths from `/content/drive/MyDrive/...` to `/kaggle/working/...`.

---

## Option 3: Lightning AI Studios (Free Tier)

**What you get:** Free GPU credits (~22 hours of A10G per month), persistent filesystem, proper terminal access (not just notebooks).

**Advantages:** Feels like a real development environment. VS Code in the browser. Persistent files. You can run `setup.sh` directly.

### Setup

1. Go to [lightning.ai](https://lightning.ai) → Studios → New Studio
2. Select GPU: A10G (24 GB) or L4
3. Open terminal and run:

```bash
git clone <your-repo-url> reasoning-on-manifold
cd reasoning-on-manifold
pip install torch transformers accelerate bitsandbytes openai datasets tqdm wandb einops scikit-learn matplotlib seaborn
bash setup.sh  # Or just run directly
python scripts/run_phase1.py
```

This is the most "local machine-like" free option.

---

## Option 4: RunPod / Lambda Labs / Vast.ai (Pay-per-hour)

For serious work or the 7B model. All three let you rent GPUs by the hour with full SSH access.

| Provider | GPU | Cost/hr | Best for |
|----------|-----|---------|----------|
| RunPod | RTX 4090 (24 GB) | ~$0.44 | Budget 1.5B runs |
| RunPod | A100 (40 GB) | ~$1.10 | 7B model |
| Lambda Labs | A100 (40 GB) | ~$1.10 | 7B model, reliable |
| Vast.ai | RTX 3090 (24 GB) | ~$0.25 | Cheapest option |

### RunPod Quickstart

1. Create account at [runpod.io](https://www.runpod.io)
2. Deploy → GPU Pod → Select RTX 4090 → PyTorch template
3. Connect via SSH or web terminal
4. Clone your project and run `setup.sh`

**Total cost estimate:** The full 1.5B pipeline takes ~8 hours. At $0.44/hr (RTX 4090 on RunPod), that's about **$3.50** total.

---

## Option 5: University / Institutional HPC

If you have access through LSE, this is the best option. Most university clusters have SLURM job scheduling.

### Typical SLURM Job Script

```bash
#!/bin/bash
#SBATCH --job-name=reasoning-manifold
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%j.out

module load python/3.10 cuda/12.1

cd /home/$USER/reasoning-on-manifold
conda activate reasoning_manifold

# Run Phase 1
python scripts/run_phase1.py --model primary
```

Ask your department's IT about GPU access. LSE likely has shared compute or credits for MSc dissertation work. UCL also has HPC resources if you still have access through alumni channels.

---

## Option 6: Your Own Machine

If you have a gaming PC with an RTX 3060 (12 GB) or better:

```bash
# Confirm GPU
nvidia-smi

# Clone and set up
git clone <repo> reasoning-on-manifold
cd reasoning-on-manifold
bash setup.sh
conda activate reasoning_manifold
export OPENAI_API_KEY='sk-...'
python scripts/run_phase1.py
```

For the 1.5B model, even an RTX 3060 (12 GB) works fine.

---

## Recommended Workflow

Given that you're an MSc student likely without dedicated GPU hardware:

1. **Start with Google Colab Free** for development and debugging. Run a small pilot: 50 tasks instead of 1000, verify the pipeline works end-to-end. This costs nothing and takes ~1 hour.

2. **Scale to Kaggle** for the full 1000-task run on 1.5B. Kaggle's 30 hours/week is enough for the complete Phase 1 pipeline.

3. **Use RunPod A100** ($1.10/hr) for the 7B model replication in Phase 5. Budget ~$15-20 for a full 7B run.

4. **PCA analysis and plotting** (Phase 3-4 analysis portions) can run on **CPU only** — no GPU needed. Do this on your laptop or in a free Colab without GPU.

---

## Handling Colab/Kaggle Disconnections

The code checkpoints at regular intervals. The key pattern:

```python
# All save paths point to persistent storage (Drive / Kaggle output)
# If your session disconnects, just re-run the cell — it will
# detect the checkpoint and resume from where it left off.

chains = generate_chains_for_tasks(
    model=model,
    tokenizer=tokenizer,
    tasks=tasks,
    save_path=Path("/content/drive/MyDrive/reasoning-on-manifold/chains.json"),
    checkpoint_every=25,  # Saves every 25 chains
)
# If you re-run this after a disconnect, it loads the checkpoint
# and continues from chain 26 (or wherever it stopped).
```

The same resume logic is built into `annotate_chains` and `run_phase1.py`.

---

## Cost Summary

| Approach | GPU Cost | API Cost (GPT-4o) | Total |
|----------|---------|-------------------|-------|
| Colab Free + 1.5B | $0 | ~$10-15 | ~$10-15 |
| Kaggle Free + 1.5B | $0 | ~$10-15 | ~$10-15 |
| RunPod 4090 + 1.5B | ~$3.50 | ~$10-15 | ~$15-20 |
| RunPod A100 + 7B | ~$15-20 | ~$10-15 | ~$25-35 |
| Full project (both models) | ~$20 | ~$30-40 | ~$50-60 |
