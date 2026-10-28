# Notice

These files are derived works distributed with the Muslim Nova app.

- **whisper-base-ar-quran** by Tarteel AI
  (https://huggingface.co/tarteel-ai/whisper-base-ar-quran), licensed under
  the Apache License 2.0 (see `LICENSE`). It is a fine-tune of OpenAI's
  Whisper base (MIT). Changes: weights renamed from the Hugging Face to the
  openai-whisper layout, exported to ONNX with sherpa-onnx's exporter and
  quantized to int8. The model's behaviour is unchanged; the release
  workflow checks the conversion numerically.
- **Silero VAD** (https://github.com/snakers4/silero-vad), MIT licence,
  as packaged by sherpa-onnx.

No recordings, user data or app code are stored in this repository.
