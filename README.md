# Reasoning on a Manifold

**Do individual reasoning behaviours in thinking LLMs have manifold structure?**

This project synthesises two lines of work:

- **Huang et al. (NeurIPS 2025)** — *Mitigating Overthinking in Large Reasoning Models via Manifold Steering*: showed that the composite phenomenon of overthinking lives on a low-dimensional manifold in activation space, and that projecting steering vectors onto this manifold dramatically improves intervention effectiveness.

- **Venhoff et al. (ICLR 2025 Workshop)** — *Understanding Reasoning in Thinking Language Models via Steering Vectors*: identified several distinct reasoning behaviours in thinking LLMs (backtracking, uncertainty estimation, example testing, knowledge augmentation) and showed each can be controlled via a single linear steering vector.

**The gap:** Venhoff assumes each behaviour is captured by a single direction. Huang only studies the composite overthinking phenomenon. Nobody has checked whether *individual* reasoning behaviours have richer geometric structure (multi-dimensional manifolds rather than single directions).

**The hypothesis:** Behaviours like backtracking come in multiple flavours (e.g., "arithmetic re-checking" vs "strategy-level pivoting"). If so, they should occupy a multi-dimensional subspace, and manifold-projected steering per behaviour should enable finer-grained control than a single vector.

## Quick Start

```bash
# 1. Clone and set up
git clone <this-repo>
cd reasoning-on-manifold
bash setup.sh

# 2. Activate environment
conda activate reasoning_manifold

# 3. Set OpenAI API key (for task generation and annotation)
export OPENAI_API_KEY='sk-...'

# 4. Run Phase 1 (data generation + annotation)
python scripts/run_phase1.py

# 5. Verify everything worked
python scripts/verify_setup.py
```

## Project Structure

```
reasoning-on-manifold/
├── configs/experiment_config.yaml   # All experiment parameters
├── src/
│   ├── data/                        # Task gen, chain gen, annotation
│   ├── extraction/                  # Activation extraction hooks
│   ├── analysis/                    # PCA, clustering, sub-type discovery
│   ├── steering/                    # Manifold projection, steered inference
│   ├── evaluation/                  # Benchmarks, behaviour fractions
│   └── utils/                       # Config, model loading, plotting
├── scripts/
│   ├── setup.sh                     # One-time environment bootstrap
│   ├── run_phase1.py                # Phase 1: data generation + annotation
│   └── verify_setup.py              # Check everything is configured
├── external/
│   └── steering-thinking-llms/      # Venhoff et al.'s codebase (cloned)
├── data/                            # Generated data (gitignored)
├── results/                         # Outputs (gitignored)
└── notebooks/                       # Analysis notebooks
```

## Requirements

- **GPU:** NVIDIA RTX 3090/4090 (24 GB) for 1.5B model; A100 (40 GB) for 7B
- **RAM:** 32 GB system memory
- **Disk:** ~100 GB (model weights + cached activations)
- **API:** OpenAI API key (~$50-100 for full experiment)
- **Time:** ~1 day for complete pipeline on 1.5B model

## References

- Huang et al., "Mitigating Overthinking in Large Reasoning Models via Manifold Steering," NeurIPS 2025.
- Venhoff et al., "Understanding Reasoning in Thinking Language Models via Steering Vectors," ICLR 2025 Workshop.
