"""Convert a Hugging Face Whisper checkpoint to the openai-whisper format.

sherpa-onnx exports Whisper through openai-whisper, while tarteel-ai's
Quran model is published in Hugging Face format. The two hold the same
weights under different names; this renames them, builds the openai
`dims`, and checks the result numerically against the Hugging Face model
before writing it.

Usage: python convert_hf_to_openai.py HF_DIR OUT.pt
"""
import sys

import torch
from transformers import WhisperForConditionalGeneration

# Hugging Face name fragment -> openai name fragment. Order matters: the
# cross-attention and top-level layer norms must be renamed before the
# shorter fragments they contain.
RENAMES = [
    ("encoder_attn.k_proj", "cross_attn.key"),
    ("encoder_attn.q_proj", "cross_attn.query"),
    ("encoder_attn.v_proj", "cross_attn.value"),
    ("encoder_attn.out_proj", "cross_attn.out"),
    ("encoder_attn_layer_norm", "cross_attn_ln"),
    ("self_attn.k_proj", "attn.key"),
    ("self_attn.q_proj", "attn.query"),
    ("self_attn.v_proj", "attn.value"),
    ("self_attn.out_proj", "attn.out"),
    ("self_attn_layer_norm", "attn_ln"),
    ("final_layer_norm", "mlp_ln"),
    (".fc1.", ".mlp.0."),
    (".fc2.", ".mlp.2."),
    ("encoder.layers.", "encoder.blocks."),
    ("decoder.layers.", "decoder.blocks."),
    ("encoder.layer_norm.", "encoder.ln_post."),
    ("decoder.layer_norm.", "decoder.ln."),
    ("encoder.embed_positions.weight", "encoder.positional_embedding"),
    ("decoder.embed_positions.weight", "decoder.positional_embedding"),
    ("decoder.embed_tokens.", "decoder.token_embedding."),
]


def openai_name(hf_name: str) -> str:
    name = hf_name.removeprefix("model.")
    for old, new in RENAMES:
        name = name.replace(old, new)
    return name


def main(hf_dir: str, out: str) -> None:
    hf = WhisperForConditionalGeneration.from_pretrained(hf_dir).eval()
    c = hf.config
    dims = {
        "n_mels": c.num_mel_bins,
        "n_vocab": c.vocab_size,
        "n_audio_ctx": c.max_source_positions,
        "n_audio_state": c.d_model,
        "n_audio_head": c.encoder_attention_heads,
        "n_audio_layer": c.encoder_layers,
        "n_text_ctx": c.max_target_positions,
        "n_text_state": c.d_model,
        "n_text_head": c.decoder_attention_heads,
        "n_text_layer": c.decoder_layers,
    }
    state = {
        openai_name(k): v
        for k, v in hf.state_dict().items()
        # The output projection is tied to the token embedding in openai.
        if k != "proj_out.weight"
    }
    torch.save({"dims": dims, "model_state_dict": state}, out)

    # Same input, same logits: the conversion is only a renaming.
    import whisper

    oa = whisper.load_model(out, device="cpu").eval()
    torch.manual_seed(0)
    mel = torch.randn(1, c.num_mel_bins, 3000)
    tokens = torch.tensor([[50258, 50272, 50359, 50363]])  # sot, ar, transcribe, notimestamps
    with torch.no_grad():
        expected = hf(input_features=mel, decoder_input_ids=tokens).logits
        actual = oa(mel, tokens)
    diff = (expected - actual).abs().max().item()
    print(f"max logit difference after conversion: {diff:.2e}")
    if diff > 1e-3:
        sys.exit("conversion changed the model's output")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
