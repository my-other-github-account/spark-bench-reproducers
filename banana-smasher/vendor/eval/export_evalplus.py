#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pathlib
from typing import Dict, List

from p968_common import ARMS, DATASET_CONTRACTS, load_dataset, row_path, sha256, valid_row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows-root", type=pathlib.Path, required=True)
    parser.add_argument("--model-arm", required=True)
    parser.add_argument("--dataset-name", choices=sorted(DATASET_CONTRACTS), required=True)
    parser.add_argument("--dataset-path", type=pathlib.Path, required=True)
    parser.add_argument("--decode-arm", choices=sorted(ARMS), required=True)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    parser.add_argument("--n-samples", type=int, default=0)
    args = parser.parse_args()

    dataset, receipt = load_dataset(args.dataset_name, args.dataset_path)
    n_samples = args.n_samples or int(ARMS[args.decode_arm]["n"])
    output: List[Dict[str, str]] = []
    index_rows = []
    for source in dataset:
        for sample_index in range(n_samples):
            seed = int(ARMS[args.decode_arm]["seed_start"]) + sample_index
            expected = {"schema": "p968-generation-row-v1", "task_id": source["task_id"], "dataset": args.dataset_name, "decode_arm": args.decode_arm, "sample_index": sample_index, "seed": seed, "model_arm": args.model_arm}
            path = row_path(args.rows_root, args.model_arm, args.dataset_name, args.decode_arm, sample_index, source["task_id"])
            if not valid_row(path, expected):
                raise RuntimeError("missing or invalid row %s" % path)
            row = json.loads(path.read_text())
            identifier = "%s:%s:%s:s%03d:%s" % (args.model_arm, args.dataset_name, args.decode_arm, sample_index, source["task_id"])
            output.append({"task_id": source["task_id"], "solution": row["output"]["text"], "_identifier": identifier})
            index_rows.append({"task_id": source["task_id"], "sample_index": sample_index, "seed": seed, "source_path": str(path), "source_sha256": sha256(path), "identifier": identifier})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in output))
    index_path = args.out.with_suffix(args.out.suffix + ".index.json")
    index_path.write_text(json.dumps({"schema": "p968-evalplus-export-index-v1", "status": "PASS", "dataset": receipt, "model_arm": args.model_arm, "decode_arm": args.decode_arm, "n_samples": n_samples, "rows": index_rows, "jsonl_sha256": sha256(args.out)}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "PASS", "rows": len(output), "jsonl": str(args.out), "sha256": sha256(args.out), "index": str(index_path), "index_sha256": sha256(index_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
