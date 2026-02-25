# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
qwen_tts: Qwen-TTS package.
"""

import os
import sys
import ctypes

# Ensure pip-installed cuDNN libs are discoverable before importing torch.
def _maybe_add_cudnn_lib_path():
    py_ver = f"python{sys.version_info.major}.{sys.version_info.minor}"
    candidates = [
        os.path.join(sys.prefix, "lib", py_ver, "site-packages", "nvidia", "cudnn", "lib"),
        os.path.join(sys.prefix, "lib", "site-packages", "nvidia", "cudnn", "lib"),
        os.path.join(sys.base_prefix, "lib", py_ver, "site-packages", "nvidia", "cudnn", "lib"),
        os.path.join(sys.base_prefix, "lib", "site-packages", "nvidia", "cudnn", "lib"),
    ]
    for path in candidates:
        if os.path.isdir(path):
            existing = os.environ.get("LD_LIBRARY_PATH", "")
            if path not in existing.split(":"):
                os.environ["LD_LIBRARY_PATH"] = f"{path}:{existing}" if existing else path
            break

_maybe_add_cudnn_lib_path()

def _maybe_preload_cudnn_libs():
    py_ver = f"python{sys.version_info.major}.{sys.version_info.minor}"
    candidates = [
        os.path.join(sys.prefix, "lib", py_ver, "site-packages", "nvidia", "cudnn", "lib"),
        os.path.join(sys.prefix, "lib", "site-packages", "nvidia", "cudnn", "lib"),
        os.path.join(sys.base_prefix, "lib", py_ver, "site-packages", "nvidia", "cudnn", "lib"),
        os.path.join(sys.base_prefix, "lib", "site-packages", "nvidia", "cudnn", "lib"),
    ]
    lib_dir = next((p for p in candidates if os.path.isdir(p)), None)
    if not lib_dir:
        return
    for lib_name in (
        "libcudnn.so.9",
        "libcudnn_ops.so.9",
        "libcudnn_cnn.so.9",
        "libcudnn_graph.so.9",
        "libcudnn_adv.so.9",
        "libcudnn_heuristic.so.9",
        "libcudnn_engines_precompiled.so.9",
        "libcudnn_engines_runtime_compiled.so.9",
    ):
        lib_path = os.path.join(lib_dir, lib_name)
        if os.path.exists(lib_path):
            try:
                ctypes.CDLL(lib_path, mode=ctypes.RTLD_GLOBAL)
            except OSError:
                pass

_maybe_preload_cudnn_libs()

from .inference.qwen3_tts_model import Qwen3TTSModel, VoiceClonePromptItem
from .inference.qwen3_tts_tokenizer import Qwen3TTSTokenizer

__all__ = ["__version__"]
__version__ = "0.0.1"
