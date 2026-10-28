"""Recognise known recitations with the exported int8 model, as the app will.

Each case is a published ayah recording and its imla'i text; the release
fails unless the model gets at least `MIN_WORDS` of the words right. The
transcripts go into the release notes.

Usage: python smoke_test.py MODEL_DIR WAV_DIR > results.md
"""
import re
import sys
import wave
from pathlib import Path

import numpy as np
import sherpa_onnx

MIN_WORDS = 0.8

# (file, expected imla'i text) - Hafs, from the KFGQPC imla'i column.
CASES = [
    ("001001.wav", "بسم الله الرحمن الرحيم"),
    ("001002.wav", "الحمد لله رب العالمين"),
    ("002002.wav", "ذلك الكتاب لا ريب فيه هدى للمتقين"),
    ("112001.wav", "قل هو الله أحد"),
]


def normalize(text: str) -> list[str]:
    """The app's comparison form (lib/core/text/arabic_normalizer.dart)."""
    text = re.sub("[ؐ-ًؚ-ٰٟۖ-ۭـ]", "", text)
    text = re.sub("[آأإٱٲٳ]", "ا", text)
    for a, b in (("ى", "ي"), ("ی", "ي"), ("ک", "ك"),
                 ("ة", "ه"), ("ؤ", "و"), ("ئ", "ي"),
                 ("ء", "")):
        text = text.replace(a, b)
    text = re.sub("[^ء-ي\\s]", " ", text)
    return text.split()


def main(model_dir: str, wav_dir: str) -> int:
    d = Path(model_dir)
    recognizer = sherpa_onnx.OfflineRecognizer.from_whisper(
        encoder=str(d / "encoder.int8.onnx"),
        decoder=str(d / "decoder.int8.onnx"),
        tokens=str(d / "tokens.txt"),
        language="ar",
        task="transcribe",
        num_threads=2,
    )
    failed = False
    print("| Recording | Expected | Recognised | Words right |")
    print("|---|---|---|---|")
    for name, expected in CASES:
        with wave.open(str(Path(wav_dir) / name)) as w:
            assert w.getframerate() == 16000 and w.getnchannels() == 1
            samples = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
        stream = recognizer.create_stream()
        stream.accept_waveform(16000, samples.astype(np.float32) / 32768)
        recognizer.decode_stream(stream)
        text = stream.result.text.strip()
        want, got = normalize(expected), normalize(text)
        right = sum(1 for w in want if w in got) / len(want)
        failed |= right < MIN_WORDS
        print(f"| {name} | {expected} | {text} | {right:.0%} |")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
