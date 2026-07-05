#!/bin/bash
# One-time environment setup on a fresh RunPod pod (PyTorch template).
# Run from the repo directory after it's been rsync'd onto the pod.
set -e
cd "$(dirname "$0")"
echo "=== repo: $(pwd) ==="
echo "=== installing deps ([gpu] extra: torch already in the PyTorch image) ==="
pip install -q -e ".[gpu]"
echo "=== GPU + deps sanity ==="
python -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU')"
python -c "import transformers, sklearn, numpy, scipy, tqdm, yaml; print('deps OK')"
echo "=== disk on the working volume ==="
df -h . | tail -1
echo "=== setup complete ==="
