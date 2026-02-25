---
title: Qwen3-TTS Demo
emoji: 🎙️
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 5.33.0
app_file: app.py
pinned: false
license: apache-2.0
suggested_hardware: zero-a10g
---

## Local Speech Dispatcher provider (sd_generic)

This repo now includes a local `speech-dispatcher` integration for Qwen3-TTS:

- Provider CLI: `qwen_tts/cli/speechd_provider.py`
- Wrapper script: `extras/speech-dispatcher/qwen3-tts-speechd.sh`
- Module template: `extras/speech-dispatcher/qwen3-tts-generic.conf`

### 1) Install Python deps

```bash
cd /home/john/git-projects/Qwen3-TTS
python -m pip install -r requirements.txt
```

### 2) Copy module config

```bash
mkdir -p ~/.config/speech-dispatcher/modules
cp extras/speech-dispatcher/qwen3-tts-generic.conf ~/.config/speech-dispatcher/modules/qwen3-tts.conf
```

If your repo path is different, edit this line in `~/.config/speech-dispatcher/modules/qwen3-tts.conf`:

```conf
GenericExecuteSynth "cat | /home/john/git-projects/Qwen3-TTS/extras/speech-dispatcher/qwen3-tts-speechd.sh"
```

### 3) Enable module in speechd config

Add this to `~/.config/speech-dispatcher/speechd.conf`:

```conf
AddModule "qwen3tts" "sd_generic" "qwen3-tts.conf"
DefaultModule "qwen3tts"
```

### 4) Optional model/runtime env vars

Set these before starting `speech-dispatcher`:

```bash
export QWEN_SPEECHD_CHECKPOINT="Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
export QWEN_SPEECHD_DEVICE="cuda:0"         # use cpu if no GPU
export QWEN_SPEECHD_DTYPE="bfloat16"        # or float16/float32
export QWEN_SPEECHD_SPEAKER="Aiden"         # optional
export QWEN_SPEECHD_LANGUAGE="auto"         # optional
export HF_TOKEN="<your_huggingface_token>"  # optional for gated/private models
```

### 5) Restart and test

```bash
systemctl --user daemon-reload
systemctl --user restart qwen3-tts-daemon.service
systemctl --user restart speech-dispatcher.socket
spd-say -o qwen3tts "Hello from Qwen three text to speech."
```
