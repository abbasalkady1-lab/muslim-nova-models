"""Export an openai-format Whisper checkpoint with sherpa-onnx's own exporter.

sherpa-onnx's scripts/whisper/export-onnx.py only takes model names; this
runs it unchanged on our checkpoint by swapping its loader. The exporter
names its outputs after the model name ("base-..."), which is the right
architecture for whisper-base-ar-quran.

Usage: python export_onnx.py SHERPA_EXPORT_SCRIPT CHECKPOINT.pt
"""
import importlib.util
import sys

import whisper

script, checkpoint = sys.argv[1], sys.argv[2]
spec = importlib.util.spec_from_file_location("sherpa_export", script)
exporter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exporter)

exporter.load_model = lambda name: whisper.load_model(checkpoint, device="cpu")
sys.argv = [script, "--model", "base"]

from whisper.model import disable_sdpa  # noqa: E402  (as the exporter does)

with disable_sdpa():
    exporter.main()
