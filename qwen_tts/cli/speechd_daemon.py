# coding=utf-8
from __future__ import annotations

import argparse
import json
import os
import re
import socket
import sys
from pathlib import Path
from typing import List, Optional

import soundfile as sf
import torch

from qwen_tts import Qwen3TTSModel, VoiceClonePromptItem
from qwen_tts.cli.speechd_provider import _resolve_speaker, _select_device_and_dtype


def _load_text(s: Optional[str]) -> str:
    text = re.sub(r"\s+", " ", (s or "")).strip()
    if not text:
        raise ValueError("No input text provided")
    return text


_DIGIT_WORDS = {
    "0": "zero",
    "1": "one",
    "2": "two",
    "3": "three",
    "4": "four",
    "5": "five",
    "6": "six",
    "7": "seven",
    "8": "eight",
    "9": "nine",
}
_SEP_WORDS = {
    "-": "dash",
    "/": "slash",
    "_": "underscore",
}


def _normalize_english_text(text: str) -> str:
    # Expand all-caps initialisms and any alnum model/part tokens to reduce
    # multilingual pronunciation drift (e.g. RX-7800XT -> R X dash seven ...).
    token_re = re.compile(r"[A-Za-z0-9][A-Za-z0-9_/-]*")

    def _expand_token(tok: str) -> str:
        if tok.isdigit():
            return " ".join(_DIGIT_WORDS.get(ch, ch) for ch in tok)

        if any(ch.isdigit() for ch in tok):
            out = []
            for ch in tok:
                if ch.isdigit():
                    out.append(_DIGIT_WORDS.get(ch, ch))
                elif ch.isalpha():
                    out.append(ch.upper())
                elif ch in _SEP_WORDS:
                    out.append(_SEP_WORDS[ch])
            return " ".join(out) if out else tok

        if re.fullmatch(r"[A-Z]{2,}", tok):
            return " ".join(tok)

        return tok

    return token_re.sub(lambda m: _expand_token(m.group(0)), text)


def _normalize_clone_text(text: str) -> str:
    # Clone prompting is sensitive to punctuation-heavy strings in some cases.
    # Keep sentence punctuation, but make token separators explicit and stable.
    out = text
    out = out.replace("’", "'").replace("`", "'")
    out = out.replace("“", '"').replace("”", '"')
    out = out.replace("/", " slash ")
    out = re.sub(r"(?<=\w)'(?=\w)", "", out)
    out = re.sub(r"\s+", " ", out).strip()
    return out


def _fallback_clone_text(text: str) -> str:
    # Conservative fallback used only if clone generation fails.
    out = _normalize_clone_text(text)
    out = out.replace(".", ",")
    out = re.sub(r"[;:!?]", ",", out)
    out = re.sub(r",\s*,+", ", ", out)
    out = re.sub(r"\s+", " ", out).strip(" ,")
    return out


def _fallback_voice_design_text(text: str) -> str:
    # Keep content intact but de-emphasize hard sentence boundaries that can
    # occasionally destabilize generation for some prompts.
    out = text.replace(".", ",")
    out = re.sub(r"[;:!?]", ",", out)
    out = re.sub(r",\s*,+", ", ", out)
    out = re.sub(r"\s+", " ", out).strip(" ,")
    return out


def _normalize_voice_design_text(text: str) -> str:
    # VoiceDesign can become unstable on hard sentence boundaries in long lines.
    # Convert strong punctuation to soft pauses before generation.
    out = text.replace("...", ", ")
    out = out.replace(".", ", ")
    out = re.sub(r"[;:!?]", ", ", out)
    out = re.sub(r",\s*,+", ", ", out)
    out = re.sub(r"\s+", " ", out).strip(" ,")
    return out


def _load_voice_clone_prompt_items(path: str) -> List[VoiceClonePromptItem]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("Voice clone prompt file must contain a dict payload")
    items_raw = payload.get("items")
    if not isinstance(items_raw, list) or not items_raw:
        raise ValueError("Voice clone prompt file has no items")

    items: List[VoiceClonePromptItem] = []
    for d in items_raw:
        if not isinstance(d, dict):
            raise ValueError("Invalid prompt item format")
        ref_code = d.get("ref_code", None)
        if ref_code is not None and not torch.is_tensor(ref_code):
            ref_code = torch.tensor(ref_code)
        ref_spk = d.get("ref_spk_embedding", None)
        if ref_spk is None:
            raise ValueError("Missing ref_spk_embedding in prompt item")
        if not torch.is_tensor(ref_spk):
            ref_spk = torch.tensor(ref_spk)

        items.append(
            VoiceClonePromptItem(
                ref_code=ref_code,
                ref_spk_embedding=ref_spk,
                x_vector_only_mode=bool(d.get("x_vector_only_mode", False)),
                icl_mode=bool(d.get("icl_mode", not bool(d.get("x_vector_only_mode", False)))),
                ref_text=d.get("ref_text", None),
            )
        )
    return items


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="qwen-tts-speechd-daemon")
    p.add_argument("--socket", default=os.environ.get("QWEN_SPEECHD_SOCKET", "/tmp/qwen3-tts-speechd.sock"))
    p.add_argument(
        "--checkpoint",
        default=os.environ.get(
            "QWEN_SPEECHD_CHECKPOINT",
            "/home/john/.cache/huggingface/hub/models--Qwen--Qwen3-TTS-12Hz-0.6B-CustomVoice/snapshots/22fe0656d05e0d0d2ca5cd129449e3487b043c59",
        ),
    )
    p.add_argument("--device", default=os.environ.get("QWEN_SPEECHD_DEVICE", "cpu"))
    p.add_argument("--dtype", default=os.environ.get("QWEN_SPEECHD_DTYPE", "float32"))
    p.add_argument(
        "--mode",
        choices=["custom_voice", "voice_design", "voice_clone_prompt"],
        default=os.environ.get("QWEN_SPEECHD_MODE", "custom_voice"),
        help="Synthesis mode: predefined speaker (custom_voice) or natural-language voice design (voice_design).",
    )
    p.add_argument(
        "--voice-design-checkpoint",
        default=os.environ.get("QWEN_SPEECHD_VOICE_DESIGN_CHECKPOINT", "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"),
        help="Checkpoint used when mode=voice_design.",
    )
    p.add_argument(
        "--voice-clone-checkpoint",
        default=os.environ.get("QWEN_SPEECHD_VOICE_CLONE_CHECKPOINT", "Qwen/Qwen3-TTS-12Hz-1.7B-Base"),
        help="Checkpoint used when mode=voice_clone_prompt.",
    )
    p.add_argument(
        "--voice-clone-prompt-file",
        default=os.environ.get("QWEN_SPEECHD_VOICE_CLONE_PROMPT_FILE", ""),
        help="Path to saved voice clone prompt .pt file (required for mode=voice_clone_prompt).",
    )
    p.add_argument("--language", default=os.environ.get("QWEN_SPEECHD_LANGUAGE", "auto"))
    p.add_argument("--speaker", default=os.environ.get("QWEN_SPEECHD_SPEAKER", "Aiden"))
    p.add_argument("--instruct", default=os.environ.get("QWEN_SPEECHD_INSTRUCT", ""))
    p.add_argument(
        "--respect-sd-voice",
        action="store_true",
        default=str(os.environ.get("QWEN_SPEECHD_RESPECT_SD_VOICE", "")).lower() in {"1", "true", "yes", "on"},
        help="If set, prefer Speech Dispatcher VOICE over configured/default speaker.",
    )
    p.add_argument(
        "--respect-sd-language",
        action="store_true",
        default=str(os.environ.get("QWEN_SPEECHD_RESPECT_SD_LANGUAGE", "")).lower() in {"1", "true", "yes", "on"},
        help="If set, prefer Speech Dispatcher LANGUAGE over configured/default language.",
    )
    p.add_argument(
        "--normalize-english-text",
        action="store_true",
        default=str(os.environ.get("QWEN_SPEECHD_NORMALIZE_ENGLISH_TEXT", "1")).lower() in {"1", "true", "yes", "on"},
        help="Normalize numbers/acronyms/model identifiers when language is English.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=int(os.environ.get("QWEN_SPEECHD_SEED", "1234")),
        help="Deterministic seed for generation to reduce run-to-run variation.",
    )
    p.add_argument(
        "--do-sample",
        action="store_true",
        default=str(os.environ.get("QWEN_SPEECHD_DO_SAMPLE", "0")).lower() in {"1", "true", "yes", "on"},
        help="Enable sampling in generation (less deterministic).",
    )
    p.add_argument(
        "--subtalker-dosample",
        action="store_true",
        default=str(os.environ.get("QWEN_SPEECHD_SUBTALKER_DOSAMPLE", "0")).lower() in {"1", "true", "yes", "on"},
        help="Enable sub-talker sampling (less deterministic).",
    )
    p.add_argument(
        "--max-new-tokens-clone",
        type=int,
        default=int(os.environ.get("QWEN_SPEECHD_MAX_NEW_TOKENS_CLONE", "768")),
        help="Generation max_new_tokens specifically for voice_clone_prompt mode.",
    )
    p.add_argument(
        "--max-new-tokens-voice-design",
        type=int,
        default=int(os.environ.get("QWEN_SPEECHD_MAX_NEW_TOKENS_VOICE_DESIGN", "768")),
        help="Generation max_new_tokens specifically for voice_design mode.",
    )
    p.add_argument(
        "--normalize-clone-text",
        action="store_true",
        default=str(os.environ.get("QWEN_SPEECHD_NORMALIZE_CLONE_TEXT", "1")).lower() in {"1", "true", "yes", "on"},
        help="Normalize punctuation for voice_clone_prompt mode before generation.",
    )
    p.add_argument(
        "--retry-clone-with-safe-punctuation",
        action="store_true",
        default=str(os.environ.get("QWEN_SPEECHD_RETRY_CLONE_WITH_SAFE_PUNCT", "1")).lower()
        in {"1", "true", "yes", "on"},
        help="Retry clone generation with conservative punctuation if the first attempt fails.",
    )
    p.add_argument(
        "--retry-voice-design-with-safe-punctuation",
        action="store_true",
        default=str(os.environ.get("QWEN_SPEECHD_RETRY_VOICE_DESIGN_WITH_SAFE_PUNCT", "1")).lower()
        in {"1", "true", "yes", "on"},
        help="Retry voice_design generation with conservative punctuation if the first attempt fails.",
    )
    p.add_argument(
        "--normalize-voice-design-text",
        action="store_true",
        default=str(os.environ.get("QWEN_SPEECHD_NORMALIZE_VOICE_DESIGN_TEXT", "1")).lower()
        in {"1", "true", "yes", "on"},
        help="Normalize punctuation before voice_design generation.",
    )
    p.add_argument("--max-new-tokens", type=int, default=int(os.environ.get("QWEN_SPEECHD_MAX_NEW_TOKENS", "384")))
    return p


def _send(conn: socket.socket, payload: dict) -> None:
    try:
        conn.sendall((json.dumps(payload, ensure_ascii=True) + "\n").encode("utf-8"))
    except (BrokenPipeError, ConnectionResetError):
        # Probe clients may disconnect immediately; do not crash daemon.
        pass


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
            raise RuntimeError("Request too large")
        if b"\n" in b:
            break
    data = b"".join(chunks)
    if b"\n" in data:
        data = data.split(b"\n", 1)[0]
    return data


def _handle_conn(
    conn: socket.socket,
    tts: Qwen3TTSModel,
    mode: str,
    voice_clone_prompt_items: Optional[List[VoiceClonePromptItem]],
    synth_lock,
    default_language: str,
    default_speaker: str,
    default_instruct: str,
    respect_sd_voice: bool,
    respect_sd_language: bool,
    normalize_english_text: bool,
    seed: int,
    do_sample: bool,
    subtalker_dosample: bool,
    max_new_tokens_clone: int,
    max_new_tokens_voice_design: int,
    normalize_clone_text: bool,
    retry_clone_with_safe_punctuation: bool,
    retry_voice_design_with_safe_punctuation: bool,
    normalize_voice_design_text: bool,
    max_new_tokens: int,
) -> None:
    try:
        raw = _recv_line(conn)
        if not raw:
            return
        req = json.loads(raw.decode("utf-8")) if raw else {}
        text = _load_text(req.get("text"))
        output = req.get("output")
        if not output:
            raise ValueError("Missing output")
        if respect_sd_language:
            language = req.get("language") or default_language
        else:
            language = default_language
        request_mode = req.get("mode") or mode
        if request_mode != mode:
            raise ValueError(
                f"Daemon is running in mode={mode!r}, but request asked for mode={request_mode!r}. Restart daemon with desired mode."
            )
        if normalize_english_text and str(language).strip().lower() == "english":
            text = _normalize_english_text(text)
        if respect_sd_voice:
            requested_speaker = req.get("voice") or req.get("speaker") or default_speaker
        else:
            requested_speaker = req.get("speaker") or default_speaker
        instruct = req.get("instruct")
        if instruct is None:
            instruct = default_instruct

        # The model object is not guaranteed thread-safe; serialize generation.
        with synth_lock:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            if mode == "voice_design":
                design_text = _normalize_voice_design_text(text) if normalize_voice_design_text else text
                try:
                    wavs, sr = tts.generate_voice_design(
                        text=design_text,
                        language=language,
                        instruct=instruct or "",
                        do_sample=do_sample,
                        subtalker_dosample=subtalker_dosample,
                        max_new_tokens=max_new_tokens_voice_design,
                        non_streaming_mode=True,
                    )
                except Exception:
                    if not retry_voice_design_with_safe_punctuation:
                        raise
                    safe_text = _fallback_voice_design_text(design_text)
                    if not safe_text or safe_text == design_text:
                        raise
                    wavs, sr = tts.generate_voice_design(
                        text=safe_text,
                        language=language,
                        instruct=instruct or "",
                        do_sample=do_sample,
                        subtalker_dosample=subtalker_dosample,
                        max_new_tokens=max_new_tokens_voice_design,
                        non_streaming_mode=True,
                    )
            elif mode == "voice_clone_prompt":
                if not voice_clone_prompt_items:
                    raise ValueError("voice_clone_prompt mode requires a loaded prompt file")
                clone_text = _normalize_clone_text(text) if normalize_clone_text else text
                try:
                    wavs, sr = tts.generate_voice_clone(
                        text=clone_text,
                        language=language,
                        voice_clone_prompt=voice_clone_prompt_items,
                        do_sample=do_sample,
                        subtalker_dosample=subtalker_dosample,
                        max_new_tokens=max_new_tokens_clone,
                        non_streaming_mode=True,
                    )
                except Exception:
                    if not retry_clone_with_safe_punctuation:
                        raise
                    safe_clone_text = _fallback_clone_text(clone_text)
                    if not safe_clone_text or safe_clone_text == clone_text:
                        raise
                    wavs, sr = tts.generate_voice_clone(
                        text=safe_clone_text,
                        language=language,
                        voice_clone_prompt=voice_clone_prompt_items,
                        do_sample=do_sample,
                        subtalker_dosample=subtalker_dosample,
                        max_new_tokens=max_new_tokens_clone,
                        non_streaming_mode=True,
                    )
            else:
                speaker = _resolve_speaker(tts, requested_speaker, fallback=default_speaker)
                wavs, sr = tts.generate_custom_voice(
                    text=text,
                    language=language,
                    speaker=speaker,
                    instruct=instruct or "",
                    do_sample=do_sample,
                    subtalker_dosample=subtalker_dosample,
                    max_new_tokens=max_new_tokens,
                    non_streaming_mode=True,
                )
        if not wavs:
            raise RuntimeError("Model returned empty audio")
        out = Path(output)
        out.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(out), wavs[0], int(sr), subtype="PCM_16")
        _send(conn, {"ok": True})
    except Exception as exc:
        _send(conn, {"ok": False, "error": f"{type(exc).__name__}: {exc}"})
    finally:
        conn.close()


def main() -> int:
    args = _build_parser().parse_args()

    sock_path = Path(args.socket)
    sock_path.parent.mkdir(parents=True, exist_ok=True)
    if sock_path.exists():
        sock_path.unlink()

    device, dtype = _select_device_and_dtype(args.device, args.dtype)
    if args.mode == "voice_design":
        checkpoint = args.voice_design_checkpoint
    elif args.mode == "voice_clone_prompt":
        checkpoint = args.voice_clone_checkpoint
    else:
        checkpoint = args.checkpoint

    voice_clone_prompt_items: Optional[List[VoiceClonePromptItem]] = None
    if args.mode == "voice_clone_prompt":
        prompt_path = (args.voice_clone_prompt_file or "").strip()
        if not prompt_path:
            raise ValueError("QWEN_SPEECHD_VOICE_CLONE_PROMPT_FILE is required for mode=voice_clone_prompt")
        voice_clone_prompt_items = _load_voice_clone_prompt_items(prompt_path)

    tts = Qwen3TTSModel.from_pretrained(checkpoint, device_map=device, dtype=dtype)
    class _NoopLock:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            return False

    synth_lock = _NoopLock()

    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(str(sock_path))
    os.chmod(sock_path, 0o666)
    server.listen(16)

    while True:
        conn, _ = server.accept()
        _handle_conn(
            conn,
            tts,
            args.mode,
            voice_clone_prompt_items,
            synth_lock,
            args.language,
            args.speaker,
            args.instruct,
            args.respect_sd_voice,
            args.respect_sd_language,
            args.normalize_english_text,
            args.seed,
            args.do_sample,
            args.subtalker_dosample,
            args.max_new_tokens_clone,
            args.max_new_tokens_voice_design,
            args.normalize_clone_text,
            args.retry_clone_with_safe_punctuation,
            args.retry_voice_design_with_safe_punctuation,
            args.normalize_voice_design_text,
            args.max_new_tokens,
        )


if __name__ == "__main__":
    raise SystemExit(main())
