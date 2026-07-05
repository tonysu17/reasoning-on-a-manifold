# Safety post-training spillover — runbook

Implements the *post-training as intervention* extension
(`../../../post_training_spillover_extension.md`). We apply a safety
post-training step to **R1-1.5B** and measure how the geometry of its **generic,
non-safety** reasoning behaviours shifts (PH1 existence, PH2 selectivity). The
same run produces a safety-aligned R1-1.5B whose safety-reasoning geometry feeds
the safety extension's **S4 recipe arm** (`../../../safety_reasoning_extension.md`).

**Data = an LLM-generated contrastive dataset of harmful / non-harmful prompts.**
Harmful *requests* are paired with *refusals* (deliberative-alignment style
`<think>` deliberation that cites a policy area, then declines); benign
look-alikes are paired with helpful answers. The harmful/benign split is also the
basis for the safety direction used in extraction. Defensive scope: no
operational harmful content is ever generated; CSAM-class categories are excluded.

## Components

| File | Role |
|------|------|
| `src/safety_posttrain/contrastive.py` | dataset generation (Bedrock + offline `mock_*`), SFT formatting (prompt-masked) |
| `src/safety_posttrain/sft.py` | LoRA SFT (completion-only loss), dose-response, merge |
| `src/safety_posttrain/spillover.py` | numpy before/after geometry diff (principal angles, d_eff, drift) |
| `pt01_generate_contrastive.py` | Step 1 — build the dataset |
| `pt02_train_safety_lora.py` | Step 2 — LoRA SFT (safety + control) |
| `pt03_measure_spillover.py` | Step 3 — before/after geometry diff + PH2 ranking |

## Dependencies

Generation needs only `requests` + the proxy env (`CLAUDE_PROXY_URL`,
`CLAUDE_PROXY_KEY`). Training needs `pip install .[gpu] peft` (and `trl` is
optional — the trainer uses plain `transformers.Trainer`). Target hardware: the
DGX Spark (CUDA, bf16). A 1.5B LoRA run on ~1-2k examples is minutes, single-GPU.

## End-to-end (on the Spark)

```bash
# 1. generate the contrastive dataset (proxy creds in env)
python pt01_generate_contrastive.py --n-pairs 250 --out data/safety_contrastive.json

# 1b. size-matched NON-SAFETY control (isolates "safety" from "any SFT").
#     Build it from the model's own generic chains (benign reasoning, same count):
python - <<'PY'
import json, random
from src.safety_posttrain import contrastive as C
chains = json.load(open("data/chains_R1-1.5B.json"))
random.seed(42); random.shuffle(chains)
recs = [{"id": f"ctrl_{i:05d}", "label": "benign", "category": "generic",
         "prompt": c["instruction"], "reasoning": "",
         "answer": c["chain"], "refusal": False, "source": "control:self-chain"}
        for i, c in enumerate(chains) if c.get("chain")][:500]
C.save_dataset(recs, "data/control_generic_sft.json")
print("wrote", len(recs), "control records")
PY

# 2. LoRA SFT — safety (dose-response) and control, with merged checkpoints
python pt02_train_safety_lora.py --data data/safety_contrastive.json \
    --dose 100,300,all --merge --out-dir checkpoints/r1_1.5b_safety
python pt02_train_safety_lora.py --data data/control_generic_sft.json \
    --dose all --merge --out-dir checkpoints/r1_1.5b_control

# 3. measure spillover on the GENERIC annotated chains (base vs safety, base vs control)
python pt03_measure_spillover.py \
    --base-model-id deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
    --post-model-id checkpoints/r1_1.5b_safety/dose_all/merged \
    --annotated data/annotated_R1-1.5B.json \
    --out results/safety_posttrain/spillover_safety.json
python pt03_measure_spillover.py \
    --base-model-id deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
    --post-model-id checkpoints/r1_1.5b_control/dose_all/merged \
    --annotated data/annotated_R1-1.5B.json \
    --out results/safety_posttrain/spillover_control.json
```

**Read the result as:** safety-minus-control per-behaviour movement. If safety
shifts some behaviours (e.g. uncertainty-estimation, backtracking) *more than the
control does*, that is selective spillover (PH2). The `spillover.py` metrics are
**descriptive** — the causal claim additionally needs label-permutation nulls and
the in-/out-of-sample discipline noted in the safety extension (F2). For a clean
zero-training cross-check, also diff the public pair
`DeepSeek-R1-Distill-Qwen-1.5B` → `UCSC-VLAA/STAR1-R1-Distill-1.5B` with `pt03`.

## Offline smoke (no GPU, no proxy — verifies the plumbing)

```bash
python pt01_generate_contrastive.py --mock --n-pairs 8 --out data/safety_contrastive_mock.json
pytest tests/test_safety_posttrain.py -q
```
