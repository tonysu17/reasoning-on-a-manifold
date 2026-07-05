#!/usr/bin/env python3
"""Print the lab proxy's available models (GET /model-api/models)."""
import os
import requests

url = os.environ["CLAUDE_PROXY_URL"].replace("/invoke", "/models")
r = requests.get(url, headers={"X-Api-Key": os.environ["CLAUDE_PROXY_KEY"]}, timeout=30)
models = r.json()
print(f"{len(models)} models available:\n")
for m in models:
    mid = m.get("modelId", "")
    prov = m.get("provider", "")
    cat = m.get("category", "")
    print(f"  {mid:50s} | {prov:12s} | {cat}")
