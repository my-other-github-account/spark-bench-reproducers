#!/usr/bin/env python3
"""Demonstrate why summing tokenizer.encode(SSE_chunk) is not a valid
completion-token count.

BPE/tokenizer merges can cross arbitrary SSE chunk boundaries. The only
valid generated-token count for a fixed-TG OpenAI stream is either exact
per-token IDs emitted by the server, or the final server `usage.completion_tokens`.

Run on Spark-3 with the Qwen tokenizer, for example:

  /home/banana_bae/venvs/vllm/bin/python scripts/prove_sse_chunk_tokenization_overcounts.py \
      --tokenizer /home/banana_bae/models/Qwen3.6-27B-NVFP4-unsloth
"""
import argparse
import json

from transformers import AutoTokenizer

EXAMPLES = [
    ["Hel", "lo", " world"],
    [" multi", "-", "token", " boundary"],
    ["The quick brown", " fox jumps", " over the lazy dog"],
    ["import", " torch", "\n", "def", " foo", "():", "\n    return", " 42"],
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    rows = []
    for chunks in EXAMPLES:
        joined = "".join(chunks)
        joint_ids = tok.encode(joined, add_special_tokens=False)
        chunk_ids = [tok.encode(c, add_special_tokens=False) for c in chunks]
        sum_chunk_token_count = sum(len(ids) for ids in chunk_ids)
        row = {
            "chunks": chunks,
            "joined_text": joined,
            "joint_token_count": len(joint_ids),
            "sum_chunk_token_count": sum_chunk_token_count,
            "overcount": sum_chunk_token_count - len(joint_ids),
            "joint_ids": joint_ids,
            "chunk_ids": chunk_ids,
        }
        rows.append(row)

    print(json.dumps({"tokenizer": args.tokenizer, "rows": rows}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
