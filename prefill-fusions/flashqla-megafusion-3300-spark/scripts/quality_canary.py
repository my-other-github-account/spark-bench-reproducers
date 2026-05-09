#!/usr/bin/env python3
import argparse
import json
import re
import sys
import time
import urllib.request


def post(base_url, model, messages, max_tokens=160):
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=180) as r:
        obj = json.loads(r.read())
    return obj["choices"][0]["message"]["content"], time.time() - t0, obj.get("usage")


def is_prime(n):
    return n > 1 and all(n % d for d in range(2, int(n ** 0.5) + 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", default="qwen35-27b-axionml-nvfp4")
    args = ap.parse_args()

    system = {"role": "system", "content": "Answer directly. Do not show reasoning or a thinking process."}
    filler = ("Alice maintains the north sensor array. Bruno calibrates the east antenna. "
              "Cara inspects the battery logs. Diego updates the firmware checklist. ")
    long_prompt = (filler * 60) + "\nFINAL FACT: The deployment window is Tuesday at 14:30 UTC.\n" \
        "Question: According to FINAL FACT, when is the deployment window? Answer in one short sentence."

    tests = [
        ("factual", "In one sentence: what causes tides on Earth?", 220),
        ("arithmetic", "Compute 17*23 + 19. Give only the integer.", 120),
        ("json", "Return valid compact JSON with keys color and count, where color is green and count is 3. No markdown.", 160),
        ("code", "Write a Python function named is_palindrome(s) that ignores case and non-alphanumeric characters. Return only code.", 320),
        ("translation", "Translate to French: The small robot opened the red door.", 160),
        ("instruction", "List exactly three prime numbers greater than 20, separated by commas, no other text.", 120),
        ("long_fact", long_prompt, 600),
    ]

    results = []
    failures = []
    for name, prompt, max_tokens in tests:
        out, seconds, usage = post(args.base_url, args.model, [system, {"role": "user", "content": prompt}], max_tokens)
        results.append({"id": name, "output": out, "seconds": seconds, "usage": usage})
        lo = out.lower()
        ok = True
        if name == "factual":
            ok = "moon" in lo and "sun" in lo and "gravit" in lo
        elif name == "arithmetic":
            ok = out.strip() == "410"
        elif name == "json":
            try:
                ok = json.loads(out) == {"color": "green", "count": 3}
            except Exception:
                ok = False
        elif name == "code":
            ok = "def is_palindrome" in out and "isalnum" in out and "[::-1]" in out
        elif name == "translation":
            ok = "Le petit robot" in out and "porte rouge" in out
        elif name == "instruction":
            nums = [int(x) for x in re.findall(r"\d+", out)]
            ok = len(nums) == 3 and all(n > 20 and is_prime(n) for n in nums)
        elif name == "long_fact":
            ok = "Tuesday at 14:30 UTC" in out
        print(("PASS" if ok else "FAIL"), name, f"{seconds:.2f}s", out[:240].replace("\n", " "))
        if not ok:
            failures.append(name)

    print(json.dumps(results, indent=2, ensure_ascii=False))
    if failures:
        print("quality canary failures: " + ", ".join(failures), file=sys.stderr)
        raise SystemExit(1)
    print("quality canary: PASS")


if __name__ == "__main__":
    main()
