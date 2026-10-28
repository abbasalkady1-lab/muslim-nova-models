# Muslim Nova models

Speech models the Muslim Nova app downloads on demand to check a user's
Quran recitation on the phone. Audio never leaves the device; these files
only make that possible offline.

Each release is built by `.github/workflows/release-model.yml` from a pinned
upstream revision, checked numerically after conversion and on real
recitations, and lists the SHA-256 of every file. The app pins those hashes
in its code and refuses any file that does not match.

| Release | Contents |
|---|---|
| `whisper-base-ar-quran-v1` | Tarteel's Quran Whisper base, int8, for sherpa-onnx; Silero VAD |

Licences: see `NOTICE.md` and `LICENSE` (Apache-2.0).
