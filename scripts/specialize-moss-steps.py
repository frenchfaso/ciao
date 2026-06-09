#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import onnx


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "public" / "models" / "moss" / "audio-tokenizer-nano-onnx"
META_PATH = MODEL_DIR / "codec_browser_onnx_meta.json"
TOKEN_STEPS = 7


def main() -> None:
    meta = json.loads(META_PATH.read_text())
    downsample_rate = int(meta["codec_config"]["downsample_rate"])
    encoder = specialize_encoder(TOKEN_STEPS, downsample_rate)
    decoder = specialize_decoder(TOKEN_STEPS, downsample_rate, precision="fp16")
    embed_external_data(encoder)
    embed_external_data(decoder)
    write_runtime_meta(meta, encoder, decoder)
    remove_transient_exports()


def specialize_encoder(token_steps: int, downsample_rate: int) -> Path:
    input_samples = downsample_rate * (token_steps - 1)
    target = MODEL_DIR / f"moss_audio_tokenizer_encode.steps{token_steps}.webgpu.onnx"
    specialize_model(
        MODEL_DIR / "moss_audio_tokenizer_encode.webgpu.onnx",
        target,
        replacements={
            7_680: input_samples,
            3: token_steps,
        },
    )
    return target


def specialize_decoder(token_steps: int, downsample_rate: int, precision: str) -> Path:
    raw_decoder_samples = downsample_rate * token_steps
    target = MODEL_DIR / f"moss_audio_tokenizer_decode_step.{precision}.steps{token_steps}.webgpu.onnx"
    specialize_model(
        MODEL_DIR / f"moss_audio_tokenizer_decode_step.{precision}.webgpu.onnx",
        target,
        replacements={
            11_520: raw_decoder_samples,
            3: token_steps,
        },
    )
    return target


def specialize_model(source: Path, target: Path, replacements: dict[int, int]) -> None:
    model = onnx.load(source, load_external_data=False)
    patched = 0

    for value_info in list(model.graph.input) + list(model.graph.output) + list(model.graph.value_info):
        tensor_type = value_info.type.tensor_type
        if not tensor_type.HasField("shape"):
            continue

        for dim in tensor_type.shape.dim:
            replacement = replacements.get(dim.dim_value)
            if replacement is None:
                continue

            dim.dim_value = replacement
            patched += 1

    target.unlink(missing_ok=True)
    onnx.save_model(model, target)
    target.chmod(0o644)
    print(f"{target.relative_to(ROOT)} {target.stat().st_size} bytes, dims patched: {patched}")


def embed_external_data(path: Path) -> None:
    model = onnx.load_model(path, load_external_data=True)
    onnx.save_model(model, path, save_as_external_data=False)
    path.chmod(0o644)
    print(f"{path.relative_to(ROOT)} embedded {path.stat().st_size} bytes")


def write_runtime_meta(meta: dict, encoder: Path, decoder: Path) -> None:
    runtime_meta = {
        "format_version": meta["format_version"],
        "checkpoint_path": meta["checkpoint_path"],
        "runtime_files": {
            "encode": encoder.name,
            "decode_step": decoder.name,
        },
        "codec_config": meta["codec_config"],
        "onnx": meta["onnx"],
        "streaming_decode": meta["streaming_decode"],
    }
    for cache in runtime_meta["streaming_decode"]["attention_caches"]:
        cache["cache_dtype"] = "float16"
    META_PATH.write_text(json.dumps(runtime_meta, indent=2) + "\n")
    META_PATH.chmod(0o644)
    print(f"{META_PATH.relative_to(ROOT)} {META_PATH.stat().st_size} bytes")


def remove_transient_exports() -> None:
    transient_files = [
        "moss_audio_tokenizer_encode.onnx",
        "moss_audio_tokenizer_decode_step.onnx",
        "moss_audio_tokenizer_encode.webgpu.onnx",
        "moss_audio_tokenizer_decode_step.webgpu.onnx",
        "moss_audio_tokenizer_decode_step.fp16.webgpu.onnx",
        "moss_audio_tokenizer_encode.data",
        "moss_audio_tokenizer_decode_shared.data",
        "moss_audio_tokenizer_decode_shared.fp16.data",
    ]
    stale_variants = [
        "moss_audio_tokenizer_encode.steps4.webgpu.onnx",
        "moss_audio_tokenizer_encode.steps5.webgpu.onnx",
        "moss_audio_tokenizer_encode.steps8.webgpu.onnx",
        "moss_audio_tokenizer_decode_step.steps4.webgpu.onnx",
        "moss_audio_tokenizer_decode_step.steps5.webgpu.onnx",
        "moss_audio_tokenizer_decode_step.steps7.webgpu.onnx",
        "moss_audio_tokenizer_decode_step.steps8.webgpu.onnx",
        "moss_audio_tokenizer_decode_step.fp16.steps4.webgpu.onnx",
        "moss_audio_tokenizer_decode_step.fp16.steps5.webgpu.onnx",
        "moss_audio_tokenizer_decode_step.fp16.steps8.webgpu.onnx",
    ]

    for file in transient_files + stale_variants:
        path = MODEL_DIR / file
        path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
