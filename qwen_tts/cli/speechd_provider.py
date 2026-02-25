# coding=utf-8
"""Speech Dispatcher helper CLI for Qwen3-TTS.

This module is designed to be invoked by Speech Dispatcher `sd_generic`.
It reads text from stdin (or --text), synthesizes speech with Qwen3-TTS
CustomVoice, and writes a WAV file.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Dict, Optional

import soundfile as sf
import torch

from qwen_tts import Qwen3TTSModel

# Basic aliases from common speech-dispatcher symbolic voices.
VOICE_ALIASES: Dict[str, str] = {
    "male1": "aiden",
    "male2": "ryan",
    "male3": "eric",
    "female1": "serena",
    "female2": "vivian",
    "female3": "sohee",
}


def _parse_dtype(value: str) -> torch.dtype:
    v = (value or "").strip().lower()
    if v in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if v in {"float16", "fp16", "half"}:
        return torch.float16
    if v in {"float32", "fp32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {value}")


def _select_device_and_dtype(device_arg: str, dtype_arg: str) -> tuple[str, torch.dtype]:
    device = (device_arg or "").strip().lower()
    if device and device != "auto":
        return device_arg, _parse_dtype(dtype_arg)

    try:
        cuda_ok = torch.cuda.is_available()
    except Exception:
        cuda_ok = False

    if cuda_ok:
        try:
            bf16_ok = torch.cuda.is_bf16_supported()
        except Exception:
            bf16_ok = False
        return "cuda", (torch.bfloat16 if bf16_ok else torch.float16)

    return "cpu", torch.float32


def _load_text(text_arg: Optional[str]) -> str:
    if text_arg is not None:
        raw = text_arg
    else:
        raw = sys.stdin.read()

    # Speech Dispatcher can send multiline chunks; collapse whitespace.
    text = re.sub(r"\s+", " ", raw).strip()
    if not text:
        raise ValueError("No input text provided")
    return text


def _resolve_speaker(tts: Qwen3TTSModel, requested: Optional[str], fallback: str) -> str:
    supported = tts.get_supported_speakers()
    if not supported:
        return requested or fallback

    supported_map = {s.lower(): s for s in supported}

    candidate = (requested or "").strip()
    if candidate:
        c_lower = candidate.lower()
        if c_lower in supported_map:
            return supported_map[c_lower]
        alias = VOICE_ALIASES.get(c_lower)
        if alias and alias in supported_map:
            return supported_map[alias]

    fb_lower = fallback.lower()
    if fb_lower in supported_map:
        return supported_map[fb_lower]

    # Deterministic fallback if configured value is invalid.
    return sorted(supported)[0]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="qwen-tts-speechd",
        description="Synthesize speech with Qwen3-TTS for Speech Dispatcher.",
    )
    p.add_argument("--text", default=None, help="Text to speak (default: read from stdin)")
    p.add_argument("--output", required=True, help="Output WAV file path")
    p.add_argument(
        "--checkpoint",
        default=os.environ.get("QWEN_SPEECHD_CHECKPOINT", "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"),
        help="Model checkpoint path or HF repo id",
    )
    p.add_argument("--device", default=os.environ.get("QWEN_SPEECHD_DEVICE", "auto"), help="torch device map or auto")
    p.add_argument(
        "--dtype",
        default=os.environ.get("QWEN_SPEECHD_DTYPE", "bfloat16"),
        choices=["bfloat16", "bf16", "float16", "fp16", "float32", "fp32"],
        help="Model dtype",
    )
    p.add_argument(
        "--speaker",
        default=os.environ.get("QWEN_SPEECHD_SPEAKER"),
        help="Qwen speaker name (for CustomVoice). If omitted, uses --voice or a fallback.",
    )
    p.add_argument(
        "--voice",
        default=None,
        help="Speech Dispatcher $VOICE value; used as speaker if --speaker is unset.",
    )
    p.add_argument(
        "--language",
        default=os.environ.get("QWEN_SPEECHD_LANGUAGE", "auto"),
        help="Language for generation (e.g. auto, english, chinese)",
    )
    p.add_argument(
        "--instruct",
        default=os.environ.get("QWEN_SPEECHD_INSTRUCT", ""),
        help="Optional style instruction",
    )
    p.add_argument("--max-new-tokens", type=int, default=2048, help="Generation max_new_tokens")
    p.add_argument("--temperature", type=float, default=None, help="Sampling temperature")
    p.add_argument("--top-k", type=int, default=None, help="Sampling top_k")
    p.add_argument("--top-p", type=float, default=None, help="Sampling top_p")
    p.add_argument("--flash-attn", action="store_true", help="Enable flash_attention_2")
    # Accepted for compatibility with speech-dispatcher placeholders.
    p.add_argument("--rate", default=None, help=argparse.SUPPRESS)
    p.add_argument("--pitch", default=None, help=argparse.SUPPRESS)
    return p


def main() -> int:
    args = build_parser().parse_args()

    try:
        text = _load_text(args.text)
    except Exception as exc:
        print(f"qwen-tts-speechd: {exc}", file=sys.stderr)
        return 2

    selected_device, dtype = _select_device_and_dtype(args.device, args.dtype)

    kwargs = {
        "device_map": selected_device,
        "dtype": dtype,
    }
    if args.flash_attn:
        kwargs["attn_implementation"] = "flash_attention_2"

    try:
        tts = Qwen3TTSModel.from_pretrained(args.checkpoint, **kwargs)

        requested_speaker = args.speaker or args.voice
        speaker = _resolve_speaker(tts, requested_speaker, fallback="Aiden")

        gen_kwargs = {
            "max_new_tokens": args.max_new_tokens,
            "non_streaming_mode": True,
        }
        if args.temperature is not None:
            gen_kwargs["temperature"] = args.temperature
        if args.top_k is not None:
            gen_kwargs["top_k"] = args.top_k
        if args.top_p is not None:
            gen_kwargs["top_p"] = args.top_p

        wavs, sr = tts.generate_custom_voice(
            text=text,
            language=args.language,
            speaker=speaker,
            instruct=(args.instruct or ""),
            **gen_kwargs,
        )
        if not wavs:
            raise RuntimeError("Model returned empty audio")

        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(output_path), wavs[0], int(sr), subtype="PCM_16")
    except Exception as exc:
        print(f"qwen-tts-speechd: synthesis failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
