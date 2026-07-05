#!/usr/bin/env python3
"""Post-training spillover — Step 1: generate the contrastive safety dataset.

Builds an LLM-generated contrastive dataset of harmful / non-harmful prompts
(+ safe target responses) for the safety post-training intervention. Uses the
Bedrock proxy (CLAUDE_PROXY_URL / CLAUDE_PROXY_KEY); pass --mock to produce a
deterministic offline dataset with the same schema (no credentials needed).

Examples
--------
    # offline smoke (no proxy):
    python pt01_generate_contrastive.py --mock --n-pairs 8 --out data/safety_contrastive_mock.json

    # real generation (needs CLAUDE_PROXY_* in env, e.g. on the Spark):
    python pt01_generate_contrastive.py --n-pairs 250 --out data/safety_contrastive.json
"""

from __future__ import annotations

import argparse
import logging

from src.safety_posttrain import contrastive as C

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("pt01")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-pairs", type=int, default=250,
                    help="number of matched harmful/benign pairs (-> 2x records)")
    ap.add_argument("--out", default="data/safety_contrastive.json")
    ap.add_argument("--mock", action="store_true",
                    help="deterministic offline dataset (no proxy)")
    ap.add_argument("--no-responses", action="store_true",
                    help="prompts only (skip target-response generation)")
    ap.add_argument("--model", default=C.ANNOTATION_MODEL,
                    help="proxy model id for generation")
    args = ap.parse_args()

    with_responses = not args.no_responses
    if args.mock:
        log.info("MOCK mode: building deterministic offline dataset")
        records = C.mock_dataset(args.n_pairs, with_responses=with_responses)
    else:
        log.info("generating %d pairs via proxy model %s", args.n_pairs, args.model)
        records = C.build_dataset(args.n_pairs, model=args.model,
                                  with_responses=with_responses)

    C.save_dataset(records, args.out)
    summ = C.summarise(records)
    log.info("wrote %d records -> %s", summ["n"], args.out)
    log.info("  by label:    %s", summ["by_label"])
    log.info("  by category: %s", summ["by_category"])


if __name__ == "__main__":
    main()
