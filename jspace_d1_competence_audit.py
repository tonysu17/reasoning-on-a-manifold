#!/usr/bin/env python
"""D1 — behavioural competence audit of R1-1.5B on the three J-lens eval suites.

Implements §3 of results/prereg/JSPACE_R1_DIAGNOSTIC_PROTOCOL_2026-08-16.md
(sealed 2026-08-16, commit 752dee7). Diagnostic-only; non-gating; cannot
reclassify Phase 1. Deterministic scoring; greedy decoding; resume-safe.
"""
import datetime
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import time
import urllib.request
import uuid

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------- sealed constants (protocol §2–§3) ----------------
SEED = 20260816
MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
MODEL_REV = "ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562"
JLENS_COMMIT = "581d398613e5602a5af361e1c34d3a92ea82ba8e"
ELIG_MANIFEST = "results/prereg/jspace_r1_eval_eligibility_manifest.json"
ELIG_SHA256 = "a193ce15ff18d1852a870703dba104763a01b06740c99ec8fd955e3a15927f7c"
RAW_BASE = f"https://raw.githubusercontent.com/anthropics/jacobian-lens/{JLENS_COMMIT}/data/evaluations"
SUITE_FILES = {
    "association": "lens-eval-association.json",
    "typo": "lens-eval-typo.json",
    "multihop": "lens-eval-multihop.json",
}
# wrappers frozen verbatim in the sealed protocol
WRAP_MH_COT = 'Complete this sentence with the correct final word or number: "{prompt}"'
WRAP_AS_COT = 'Read this passage and name, in one word, the unstated concept it points to: "{prompt}"'
SUFFIX_AS_IMM = '\nThe single word that best describes what is left unstated here is "'
SUFFIX_TY_IMM = '\nThe misspelled word in the sentence above, spelled correctly, is "'
MAX_NEW = {"MH-imm": 8, "MH-cot": 512, "AS-imm": 4, "AS-cot": 512, "TY-imm": 4}
THRESH_WITHIN = 0.50
THRESH_OUTSIDE = 0.20


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def norm_ws(s):
    return re.sub(r"\s+", " ", s)


def match_primary(text, target):
    """Sealed primary rule: case-insensitive \b-bounded whole-string match on
    whitespace-normalized text; multi-word targets matched whole."""
    words = target.strip().split()
    pat = r"\b" + r"\s+".join(re.escape(w) for w in words) + r"\b"
    return re.search(pat, norm_ws(text), re.IGNORECASE) is not None


def match_substring(text, target):
    """Sealed secondary sensitivity count: case-insensitive substring."""
    return norm_ws(target.strip()).lower() in norm_ws(text).lower()


def main():
    repo = os.path.dirname(os.path.abspath(__file__))
    os.chdir(repo)
    run_uuid = "d1-" + datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:12]
    # resume into an existing run dir if one is marked RESUME
    outroot = "results/jspace_r1_pilot/diagnostics/d1"
    os.makedirs(outroot, exist_ok=True)
    existing = sorted(d for d in os.listdir(outroot) if d.startswith("d1-") and
                      os.path.exists(os.path.join(outroot, d, "RESUME")))
    if existing:
        run_uuid = existing[-1]
        print(f"[d1] resuming into {run_uuid}", flush=True)
    outdir = os.path.join(outroot, run_uuid)
    os.makedirs(outdir, exist_ok=True)
    open(os.path.join(outdir, "RESUME"), "w").write("resume marker; removed on success\n")
    gen_path = os.path.join(outdir, "generations.jsonl")

    # ---------- integrity: eligibility manifest ----------
    got = sha256_file(ELIG_MANIFEST)
    assert got == ELIG_SHA256, f"eligibility manifest hash mismatch: {got}"
    elig = json.load(open(ELIG_MANIFEST))

    # ---------- fetch repo eval files at pinned commit (D6: hash at fetch) ----------
    repo_suites, fetch_hashes = {}, {}
    for suite, fn in SUITE_FILES.items():
        cache = os.path.join(outdir, fn)
        if os.path.exists(cache):
            raw = open(cache, "rb").read()
        else:
            raw = urllib.request.urlopen(f"{RAW_BASE}/{fn}", timeout=60).read()
            open(cache, "wb").write(raw)
        fetch_hashes[fn] = sha256_bytes(raw)
        repo_suites[suite] = {it["name"]: it for it in json.loads(raw)["items"]}

    # ---------- build task list from eligible subsets ----------
    key_map = {"association": "lens-eval-association", "typo": "lens-eval-typo",
               "multihop": "lens-eval-multihop"}
    tasks = []  # dicts: suite, arm, name, mode(raw|chat), input_text, endpoints
    counts = {}
    for suite in ["multihop", "association", "typo"]:
        items = [it for it in elig["evaluations"][key_map[suite]]["items"]
                 if it.get("item_eligible") and it.get("n_eligible_labels", 0) > 0]
        counts[suite] = len(items)
        for it in items:
            name = it["name"]
            rit = repo_suites[suite].get(name)
            assert rit is not None, f"{suite}/{name} missing from repo file"
            assert rit["prompt"] == it["prompt"], f"{suite}/{name} prompt mismatch repo vs manifest"
            labels = [l["label"] for l in it["labels"] if l.get("eligible")]
            prompt = it["prompt"]
            if suite == "multihop":
                target = rit["target"]
                tasks.append(dict(suite=suite, arm="MH-imm", name=name, mode="raw",
                                  input_text=prompt, endpoints=dict(target=[target])))
                tasks.append(dict(suite=suite, arm="MH-cot", name=name, mode="chat",
                                  input_text=WRAP_MH_COT.format(prompt=prompt),
                                  endpoints=dict(target=[target], bridge=labels)))
            elif suite == "association":
                tasks.append(dict(suite=suite, arm="AS-imm", name=name, mode="raw",
                                  input_text=prompt + SUFFIX_AS_IMM, endpoints=dict(label=labels)))
                tasks.append(dict(suite=suite, arm="AS-cot", name=name, mode="chat",
                                  input_text=WRAP_AS_COT.format(prompt=prompt), endpoints=dict(label=labels)))
            else:  # typo, descriptive, immediate only
                tasks.append(dict(suite=suite, arm="TY-imm", name=name, mode="raw",
                                  input_text=prompt + SUFFIX_TY_IMM, endpoints=dict(label=labels)))
    assert counts == {"multihop": 81, "association": 98, "typo": 96}, f"population mismatch: {counts}"
    print(f"[d1] tasks: {len(tasks)} across {counts}", flush=True)

    # ---------- resume bookkeeping ----------
    done = set()
    if os.path.exists(gen_path):
        for line in open(gen_path):
            try:
                r = json.loads(line)
                done.add((r["arm"], r["name"]))
            except Exception:
                pass
        print(f"[d1] resume: {len(done)} generations already present", flush=True)

    # ---------- model ----------
    torch.manual_seed(SEED)
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REV)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, revision=MODEL_REV, dtype=torch.bfloat16).to(device)
    model.eval()
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    def render(task):
        if task["mode"] == "raw":
            return tok(task["input_text"], return_tensors="pt")
        msgs = [{"role": "user", "content": task["input_text"]}]
        ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
        return {"input_ids": ids}

    # pre-flight: render everything, 1-token generate one task per arm (engineering check)
    for t in tasks:
        render(t)
    seen_arms = set()
    for t in tasks:
        if t["arm"] in seen_arms:
            continue
        seen_arms.add(t["arm"])
        enc = {k: v.to(device) for k, v in render(t).items()}
        with torch.no_grad():
            model.generate(**enc, max_new_tokens=1, do_sample=False, pad_token_id=pad_id)
    print(f"[d1] pre-flight OK on arms: {sorted(seen_arms)}", flush=True)

    # ---------- main loop ----------
    t_start = time.time()
    n_run = 0
    with open(gen_path, "a") as gf:
        for i, t in enumerate(tasks):
            if (t["arm"], t["name"]) in done:
                continue
            enc = {k: v.to(device) for k, v in render(t).items()}
            n_in = enc["input_ids"].shape[1]
            t0 = time.time()
            with torch.no_grad():
                out = model.generate(**enc, max_new_tokens=MAX_NEW[t["arm"]],
                                     do_sample=False, pad_token_id=pad_id)
            gen = tok.decode(out[0][n_in:], skip_special_tokens=True)
            rec = dict(suite=t["suite"], arm=t["arm"], name=t["name"], gen=gen,
                       n_new=int(out.shape[1] - n_in), seconds=round(time.time() - t0, 2))
            gf.write(json.dumps(rec) + "\n")
            gf.flush()
            n_run += 1
            if n_run % 10 == 0:
                el = time.time() - t_start
                print(f"[d1] {n_run} new ({i+1}/{len(tasks)} listed) elapsed {el/60:.1f}m", flush=True)

    # ---------- scoring (sealed rules) ----------
    recs = [json.loads(l) for l in open(gen_path)]
    by_arm = {}
    for r in recs:
        by_arm.setdefault(r["arm"], {})[r["name"]] = r
    task_ix = {(t["arm"], t["name"]): t for t in tasks}

    def rate(arm, ep):
        rows = by_arm.get(arm, {})
        hits_p = hits_s = n = 0
        for name, r in rows.items():
            t = task_ix.get((arm, name))
            if t is None or ep not in t["endpoints"]:
                continue
            n += 1
            strings = t["endpoints"][ep]
            if any(match_primary(r["gen"], s) for s in strings):
                hits_p += 1
            if any(match_substring(r["gen"], s) for s in strings):
                hits_s += 1
        return dict(n=n, primary_hits=hits_p, primary_rate=(hits_p / n if n else None),
                    substring_hits=hits_s, substring_rate=(hits_s / n if n else None))

    results = {
        "MH-imm_target": rate("MH-imm", "target"),
        "MH-cot_target": rate("MH-cot", "target"),
        "MH-cot_bridge": rate("MH-cot", "bridge"),
        "AS-imm_label": rate("AS-imm", "label"),
        "AS-cot_label": rate("AS-cot", "label"),
        "TY-imm_label": rate("TY-imm", "label"),
    }

    def band(p):
        if p is None:
            return "no-data"
        if p >= THRESH_WITHIN:
            return "within_competence"
        if p <= THRESH_OUTSIDE:
            return "outside_competence"
        return "indeterminate"

    mh_cot = results["MH-cot_target"]["primary_rate"]
    mh_imm = results["MH-imm_target"]["primary_rate"]
    verdicts = {
        "multihop_competence_band": band(mh_cot),
        "association_competence_band": band(results["AS-cot_label"]["primary_rate"]),
        "typo_descriptive_rate": results["TY-imm_label"]["primary_rate"],
        "externalization_signature": (mh_cot is not None and mh_imm is not None
                                      and mh_cot >= THRESH_WITHIN and mh_imm <= THRESH_OUTSIDE),
    }

    manifest = dict(
        schema_version="rom-jspace-d1-competence-audit-v1",
        run_uuid=run_uuid, seed=SEED, protocol="results/prereg/JSPACE_R1_DIAGNOSTIC_PROTOCOL_2026-08-16.md",
        seal_commit="752dee7", model=MODEL_ID, model_revision=MODEL_REV, device=device,
        dtype="bfloat16", decoding="greedy", max_new_tokens=MAX_NEW,
        wrappers=dict(MH_cot=WRAP_MH_COT, AS_cot=WRAP_AS_COT, AS_imm_suffix=SUFFIX_AS_IMM,
                      TY_imm_suffix=SUFFIX_TY_IMM),
        eligibility_manifest_sha256=ELIG_SHA256, eval_file_sha256_at_fetch=fetch_hashes,
        populations=counts,
        env=dict(python=platform.python_version(), torch=torch.__version__,
                 platform=platform.platform(),
                 transformers=__import__("transformers").__version__),
        git_commit=subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                  text=True).stdout.strip(),
        started_utc=None, finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        wall_seconds=round(time.time() - t_start, 1),
    )
    json.dump(manifest, open(os.path.join(outdir, "manifest.json"), "w"), indent=1, sort_keys=True)
    json.dump(dict(results=results, verdicts=verdicts, thresholds=dict(
        within=THRESH_WITHIN, outside=THRESH_OUTSIDE)),
        open(os.path.join(outdir, "report.json"), "w"), indent=1, sort_keys=True)

    lines = ["# D1 — R1-1.5B behavioural competence audit", "",
             f"Run `{run_uuid}`; sealed protocol §3; seal commit 752dee7. Diagnostic-only, non-gating.", "",
             "| Endpoint | n | primary rate | substring rate |", "|---|---:|---:|---:|"]
    for k, v in results.items():
        pr = "—" if v["primary_rate"] is None else f"{v['primary_rate']:.3f}"
        sr = "—" if v["substring_rate"] is None else f"{v['substring_rate']:.3f}"
        lines.append(f"| {k} | {v['n']} | {pr} | {sr} |")
    lines += ["", f"**Verdicts:** {json.dumps(verdicts)}"]
    open(os.path.join(outdir, "REPORT.md"), "w").write("\n".join(lines) + "\n")

    os.remove(os.path.join(outdir, "RESUME"))
    print(f"[d1] DONE {run_uuid} wall={manifest['wall_seconds']}s", flush=True)
    print(json.dumps(dict(results=results, verdicts=verdicts), indent=1), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)
