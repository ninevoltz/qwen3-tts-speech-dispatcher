# coding=utf-8
from __future__ import annotations

import argparse
import json
import os
import socket
import sys
from typing import Optional

from qwen_tts.cli.speechd_text_sanitize import sanitize_speechd_text


def _load_text(text_arg: Optional[str]) -> str:
    raw = text_arg if text_arg is not None else sys.stdin.read()
    text = sanitize_speechd_text(raw)
    if not text.strip():
        raise ValueError("No input text provided")
    return text


def _recv_line(conn: socket.socket, limit: int = 1_000_000) -> bytes:
    chunks = []
    total = 0
    while True:
        b = conn.recv(4096)
        if not b:
            break
        chunks.append(b)
        total += len(b)
        if total > limit:
            raise RuntimeError("Response too large")
        if b"\n" in b:
            break
    data = b"".join(chunks)
    if b"\n" in data:
        data = data.split(b"\n", 1)[0]
    return data


def main() -> int:
    p = argparse.ArgumentParser(prog="qwen-tts-speechd-client")
    p.add_argument("--socket", default=os.environ.get("QWEN_SPEECHD_SOCKET", "/tmp/qwen3-tts-speechd.sock"))
    p.add_argument("--output", required=True)
    p.add_argument("--text", default=None)
    p.add_argument("--voice", default=None)
    p.add_argument("--speaker", default=None)
    p.add_argument("--language", default="auto")
    p.add_argument("--instruct", default=os.environ.get("QWEN_SPEECHD_INSTRUCT"))
    p.add_argument(
        "--mode",
        choices=["custom_voice", "voice_design", "voice_clone_prompt"],
        default=os.environ.get("QWEN_SPEECHD_MODE"),
    )
    p.add_argument("--timeout", type=float, default=None)
    args = p.parse_args()

    try:
        text = _load_text(args.text)
    except Exception as exc:
        print(f"qwen-tts-speechd-client: {exc}", file=sys.stderr)
        return 2

    req = {
        "text": text,
        "output": args.output,
        "voice": args.voice,
        "speaker": args.speaker,
        "language": args.language,
        "instruct": args.instruct,
        "mode": args.mode,
    }

    try:
        conn = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        if args.timeout is not None:
            conn.settimeout(args.timeout)
        conn.connect(args.socket)
        conn.sendall((json.dumps(req, ensure_ascii=True) + "\n").encode("utf-8"))
        resp_raw = _recv_line(conn)
        conn.close()
        resp = json.loads(resp_raw.decode("utf-8")) if resp_raw else {}
        if not resp.get("ok"):
            print(f"qwen-tts-speechd-client: {resp.get('error', 'unknown error')}", file=sys.stderr)
            return 1
        return 0
    except Exception as exc:
        print(f"qwen-tts-speechd-client: daemon call failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
