#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import onnx


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "public" / "models" / "moss" / "audio-tokenizer-nano-onnx"
META_PATH = MODEL_DIR / "codec_browser_onnx_meta.json"


def main() -> None:
    meta = json.loads(META_PATH.read_text())
    config = meta["codec_config"]
    channels = int(config["channels"])
    code_length = int(__import__("os").environ.get("MOSS_TOKEN_STEPS", "3"))
    frame_samples = int(config["downsample_rate"]) * (code_length - 1)
    decoder_audio_samples = int(config["downsample_rate"]) * code_length

    decoder_dims: dict[str, int] = {
        "code_length": code_length,
        "audio_length": decoder_audio_samples,
        "Castaudio_dim_0": 1,
        "Castaudio_dim_1": channels,
    }

    for cache in meta["streaming_decode"]["attention_caches"]:
        index = int(cache["index"])
        decoder_dims[f"Sliceattn_cached_keys_out_{index}_dim_2"] = int(cache["cache_shape"][2])
        decoder_dims[f"Sliceattn_cached_values_out_{index}_dim_2"] = int(cache["cache_shape"][2])
        decoder_dims[f"Castattn_cached_positions_out_{index}_dim_1"] = int(cache["positions_shape"][1])

    optimize_model(
        MODEL_DIR / "moss_audio_tokenizer_encode.onnx",
        MODEL_DIR / "moss_audio_tokenizer_encode.webgpu.onnx",
        {
            "batch": 1,
            "waveform_length": frame_samples,
            "code_length": code_length,
        },
    )
    optimize_model(
        MODEL_DIR / "moss_audio_tokenizer_decode_step.onnx",
        MODEL_DIR / "moss_audio_tokenizer_decode_step.webgpu.onnx",
        decoder_dims,
    )


def optimize_model(source: Path, target: Path, dim_overrides: dict[str, int]) -> None:
    model = onnx.load_model(source, load_external_data=False)
    replaced = 0

    for value_info in list(model.graph.input) + list(model.graph.output) + list(model.graph.value_info):
        tensor_type = value_info.type.tensor_type
        if not tensor_type.HasField("shape"):
            continue

        for dim in tensor_type.shape.dim:
            if dim.dim_param and dim.dim_param in dim_overrides:
                dim.dim_value = dim_overrides[dim.dim_param]
                dim.ClearField("dim_param")
                replaced += 1

    onnx.save_model(model, target)
    print(f"{target.relative_to(ROOT)} {target.stat().st_size} bytes, static dims patched: {replaced}")
    source.unlink()
    print(f"removed transient source {source.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
