#!/usr/bin/env python3
"""Merge a LoRA adapter into its base model and save a plain checkpoint
(loadable by 04_extract_activations.py --model-path). Usage:
   python3 merge_adapter.py <adapter_dir> <out_dir> [--model 1.5b]
"""
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("adapter_dir")
ap.add_argument("out_dir")
ap.add_argument("--model", default="1.5b")
a = ap.parse_args()

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.config import model_tuple

model_id, _, _ = model_tuple(a.model)
base = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, trust_remote_code=True)
merged = PeftModel.from_pretrained(base, a.adapter_dir).merge_and_unload()
merged.save_pretrained(a.out_dir)
AutoTokenizer.from_pretrained(model_id, trust_remote_code=True).save_pretrained(a.out_dir)
print(f"merged {a.adapter_dir} -> {a.out_dir}")
