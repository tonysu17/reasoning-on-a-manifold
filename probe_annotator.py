#!/usr/bin/env python3
"""Probe the lab Bedrock proxy for which model ids work, to pick a NON-builder
annotator (builder = Sonnet-4.5). Reads CLAUDE_PROXY_URL / CLAUDE_PROXY_KEY from
env (no secrets in this file). Prints status + a sample reply per candidate."""
import os
import requests

URL = os.environ["CLAUDE_PROXY_URL"]
KEY = os.environ["CLAUDE_PROXY_KEY"]

# Try many id formats — the proxy accepted the eu. Sonnet profile, so probe
# Nova/Claude/Llama/Qwen/gpt-oss across prefix variants. Print the 400 body.
CANDIDATES = [
    "eu.anthropic.claude-sonnet-4-5-20250929-v1:0",   # BUILDER (sanity only)
    "eu.anthropic.claude-3-7-sonnet-20250219-v1:0",
    "eu.anthropic.claude-3-5-haiku-20241022-v1:0",
    "anthropic.claude-3-5-haiku-20241022-v1:0",
    "eu.amazon.nova-pro-v1:0",
    "amazon.nova-pro-v1:0",
    "us.amazon.nova-pro-v1:0",
    "eu.meta.llama3-3-70b-instruct-v1:0",
    "eu.meta.llama3-1-70b-instruct-v1:0",
    "eu.mistral.mistral-large-2407-v1:0",
    "qwen3-235b", "qwen3-235b-a22b", "Qwen/Qwen3-235B-A22B",
    "nova-pro", "gpt-oss-120b", "openai.gpt-oss-120b-1:0",
]


def try_list_endpoint():
    """Some proxies expose a model list. Try a couple of likely routes."""
    import re
    base = re.sub(r"/invoke/?$", "", URL)
    for path in ("/models", "/model-api/models", ""):
        for full in ({base + path, URL.replace("/invoke", path)} if path else {base}):
            for meth in ("GET",):
                try:
                    r = requests.request(meth, full, headers={"X-Api-Key": KEY}, timeout=20)
                    print(f"  LIST {meth} {full} -> {r.status_code} {r.text[:160]!r}")
                except Exception as e:
                    print(f"  LIST {meth} {full} -> ERR {str(e)[:50]}")


def _extract(payload):
    c = payload.get("content")
    if isinstance(c, list) and c and isinstance(c[0], dict):
        return c[0].get("text", "") or ""
    return c if isinstance(c, str) else ""


for m in CANDIDATES:
    try:
        r = requests.post(
            URL,
            json={"model": m,
                  "messages": [{"role": "user", "content": "Reply with the single word OK."}],
                  "max_tokens": 16, "temperature": 0.0},
            headers={"X-Api-Key": KEY, "Content-Type": "application/json"},
            timeout=60,
        )
        if r.status_code == 200:
            txt = _extract(r.json())
            status = "WORKS" if txt.strip() else "EMPTY"
            print(f"{m:45s} {status:8s} {txt[:40]!r}")
        else:
            print(f"{m:45s} HTTP{r.status_code}  body={r.text[:120]!r}")
    except Exception as e:
        print(f"{m:45s} ERR      {type(e).__name__}: {str(e)[:70]}")

print("\n--- model-list endpoint probe ---")
try_list_endpoint()
