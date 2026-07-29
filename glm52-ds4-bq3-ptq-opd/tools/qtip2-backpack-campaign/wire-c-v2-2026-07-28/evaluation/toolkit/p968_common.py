#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import os
import pathlib
import struct
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple

PREFIX = "Please provide a self-contained Python script that solves the following problem in a markdown code block:"
DATASET_CONTRACTS = {
    "humaneval": {
        "sha256": "42526ec0e7d5f3ee0b06d6ced98f8c8bae3d76519151bfb3d36f79010645bd7f",
        "count": 164,
        "prefix": "HumanEval/",
    },
    "mbpp": {
        "sha256": "b54e762755248ca411b523c917fa9f93c07b5ff2966bf60b3917b853926a3dad",
        "count": 378,
        "prefix": "Mbpp/",
    },
}
ARMS = {
    "sampled": {"n": 5, "temperature": 0.2, "top_p": 0.95, "seed_start": 10000},
    "greedy": {"n": 3, "temperature": 0.0, "top_p": 1.0, "seed_start": 20000},
}
MAX_TOKENS = 4096


def sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def sha_i64(values: Iterable[int]) -> str:
    h = hashlib.sha256()
    for value in values:
        h.update(struct.pack("<q", int(value)))
    return h.hexdigest()


def atomic_json(path: pathlib.Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.%d.%d" % (os.getpid(), time.time_ns()))
    with tmp.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(str(tmp), str(path))


def load_dataset(name: str, path: pathlib.Path) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if name not in DATASET_CONTRACTS:
        raise ValueError("unknown dataset %s" % name)
    contract = DATASET_CONTRACTS[name]
    observed_sha = sha256(path)
    if observed_sha != contract["sha256"]:
        raise RuntimeError("%s dataset hash drift: %s" % (name, observed_sha))
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    ids = [row["task_id"] for row in rows]
    if len(rows) != contract["count"] or len(set(ids)) != len(ids):
        raise RuntimeError("%s coverage/key uniqueness drift" % name)
    if not all(task_id.startswith(contract["prefix"]) for task_id in ids):
        raise RuntimeError("%s task prefix drift" % name)
    if not all(isinstance(row.get("prompt"), str) and row["prompt"] for row in rows):
        raise RuntimeError("%s prompt missing" % name)
    receipt = {
        "name": name,
        "path": str(path),
        "sha256": observed_sha,
        "count": len(rows),
        "task_ids_sha256": sha256_json(ids),
        "first_task_id": ids[0],
        "last_task_id": ids[-1],
    }
    return rows, receipt


def conversation(row: Dict[str, Any]) -> List[Dict[str, str]]:
    prompt = row["prompt"].strip() + "\n"
    user = PREFIX + "\n```python\n" + prompt.strip() + "\n```"
    return [{"role": "user", "content": user}]


def row_path(root: pathlib.Path, model_arm: str, dataset: str, decode_arm: str, sample_index: int, task_id: str) -> pathlib.Path:
    safe = task_id.replace("/", "_")
    return root / model_arm / dataset / decode_arm / ("s%03d" % sample_index) / (safe + ".json")


def valid_row(path: pathlib.Path, expected: Dict[str, Any]) -> bool:
    try:
        row = json.loads(path.read_text(encoding="utf-8"))
        for key, value in expected.items():
            if row.get(key) != value:
                return False
        text = row["output"]["text"]
        if not isinstance(text, str):
            return False
        return row["output"]["text_sha256"] == hashlib.sha256(text.encode()).hexdigest()
    except Exception:
        return False


def summarize_vllm_logprobs(token_ids: List[int], logprobs: Optional[List[Any]]) -> Dict[str, Any]:
    if logprobs is None:
        return {"status": "UNAVAILABLE", "positions": 0}
    chosen_lp: List[Optional[float]] = []
    top_gap: List[Optional[float]] = []
    chosen_rank: List[Optional[int]] = []
    for token_id, pos in zip(token_ids, logprobs):
        entries = []
        for key, value in (pos or {}).items():
            if isinstance(value, dict):
                lp = float(value["logprob"])
                rank = value.get("rank")
            else:
                lp = float(getattr(value, "logprob"))
                rank = getattr(value, "rank", None)
            entries.append((int(key), lp, None if rank is None else int(rank)))
        entries.sort(key=lambda item: item[1], reverse=True)
        by_id = {item[0]: item for item in entries}
        selected = by_id.get(int(token_id))
        chosen_lp.append(None if selected is None else round(selected[1], 7))
        chosen_rank.append(None if selected is None else selected[2])
        top_gap.append(None if len(entries) < 2 else round(entries[0][1] - entries[1][1], 7))
    return {
        "status": "AVAILABLE",
        "positions": len(chosen_lp),
        "chosen_logprobs": chosen_lp,
        "top1_top2_gaps": top_gap,
        "chosen_ranks": chosen_rank,
    }


def summarize_openai_logprobs(choice: Dict[str, Any]) -> Dict[str, Any]:
    content = ((choice.get("logprobs") or {}).get("content") or [])
    if not content:
        return {"status": "UNAVAILABLE", "positions": 0}
    selected: List[Optional[float]] = []
    gaps: List[Optional[float]] = []
    ranks: List[Optional[int]] = []
    token_hashes: List[str] = []
    for pos in content:
        token_piece = str(pos.get("token", ""))
        lp = pos.get("logprob")
        tops = sorted((pos.get("top_logprobs") or []), key=lambda value: float(value["logprob"]), reverse=True)
        selected.append(None if lp is None else round(float(lp), 7))
        gaps.append(None if len(tops) < 2 else round(float(tops[0]["logprob"]) - float(tops[1]["logprob"]), 7))
        rank = next((idx + 1 for idx, value in enumerate(tops) if str(value.get("token", "")) == token_piece), None)
        ranks.append(rank)
        token_hashes.append(hashlib.sha256(token_piece.encode()).hexdigest()[:16])
    return {
        "status": "AVAILABLE",
        "positions": len(content),
        "chosen_logprobs": selected,
        "top1_top2_gaps": gaps,
        "chosen_ranks": ranks,
        "token_sha256_prefixes": token_hashes,
    }
