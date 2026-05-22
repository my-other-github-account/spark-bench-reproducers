#!/usr/bin/env python3
"""Tiny OpenAI-compatible mock server for proving streaming chunk token accounting bugs.

It emits deterministic SSE chat/completions chunks without choices[0].token_ids and
then reports the authoritative final usage.completion_tokens. This is intended for
llama-benchy reproduction of the chunk-local tokenizer overcount issue.
"""

from __future__ import annotations

import argparse
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


class Handler(BaseHTTPRequestHandler):
    server_version = "mock-openai-chunk-server/0.1"

    def log_message(self, fmt: str, *args: Any) -> None:
        self.server.log.write((fmt % args) + "\n")
        self.server.log.flush()

    def _json(self, status: int, obj: dict[str, Any]) -> None:
        data = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        if self.path.rstrip("/") == "/v1/models":
            self._json(200, {"object": "list", "data": [{"id": self.server.model, "object": "model"}]})
        else:
            self._json(404, {"error": {"message": f"unknown path {self.path}"}})

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0") or "0")
        body = self.rfile.read(length).decode("utf-8", errors="replace")
        try:
            req = json.loads(body) if body else {}
        except Exception:
            req = {"_raw": body}
        self.server.request_count += 1
        rid = f"mockcmpl-{self.server.request_count}"
        self.server.log.write(json.dumps({"request_id": rid, "path": self.path, "request": req}) + "\n")
        self.server.log.flush()
        if self.path.rstrip("/") not in ("/v1/chat/completions", "/v1/completions"):
            self._json(404, {"error": {"message": f"unknown path {self.path}"}})
            return
        if not req.get("stream", False):
            text = "".join(self.server.chunks)
            self._json(200, {
                "id": rid,
                "object": "chat.completion",
                "created": int(time.time()),
                "model": self.server.model,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "length"}],
                "usage": {"prompt_tokens": self.server.prompt_tokens, "completion_tokens": self.server.completion_tokens, "total_tokens": self.server.prompt_tokens + self.server.completion_tokens},
            })
            return

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True

        def send(obj: dict[str, Any]) -> None:
            self.wfile.write(b"data: ")
            self.wfile.write(json.dumps(obj).encode("utf-8"))
            self.wfile.write(b"\n\n")
            self.wfile.flush()
            if self.server.delay:
                time.sleep(self.server.delay)

        # OpenAI-style role chunk.
        send({
            "id": rid,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": self.server.model,
            "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
        })
        # Content chunks intentionally have no choices[0].token_ids.
        for chunk in self.server.chunks:
            send({
                "id": rid,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": self.server.model,
                "choices": [{"index": 0, "delta": {"content": chunk}, "finish_reason": None}],
            })
        # Final chunk carries authoritative usage.
        send({
            "id": rid,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": self.server.model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "length"}],
            "usage": {"prompt_tokens": self.server.prompt_tokens, "completion_tokens": self.server.completion_tokens, "total_tokens": self.server.prompt_tokens + self.server.completion_tokens},
        })
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=18091)
    ap.add_argument("--model", default="mock-qwen36")
    ap.add_argument("--chunks", default='["Hel", "lo", " world"]', help="JSON list of SSE delta.content chunks")
    ap.add_argument("--completion-tokens", type=int, default=2, help="Authoritative final usage.completion_tokens")
    ap.add_argument("--prompt-tokens", type=int, default=2048)
    ap.add_argument("--delay", type=float, default=0.01)
    ap.add_argument("--log", default="/tmp/mock_openai_chunk_server.log")
    args = ap.parse_args()

    chunks = json.loads(args.chunks)
    with open(args.log, "a", buffering=1) as log:
        httpd = ThreadingHTTPServer((args.host, args.port), Handler)
        httpd.model = args.model
        httpd.chunks = chunks
        httpd.completion_tokens = args.completion_tokens
        httpd.prompt_tokens = args.prompt_tokens
        httpd.delay = args.delay
        httpd.log = log
        httpd.request_count = 0
        log.write(json.dumps({"event": "start", "host": args.host, "port": args.port, "model": args.model, "chunks": chunks, "completion_tokens": args.completion_tokens}) + "\n")
        log.flush()
        httpd.serve_forever()


if __name__ == "__main__":
    main()
