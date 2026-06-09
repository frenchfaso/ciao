#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import re
import warnings
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper
from onnxconverter_common import float16


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "public" / "models" / "moss" / "audio-tokenizer-nano-onnx"

ENCODER_FP32 = MODEL_DIR / "moss_audio_tokenizer_encode.webgpu.onnx"
DECODER_FP32 = MODEL_DIR / "moss_audio_tokenizer_decode_step.webgpu.onnx"

DECODER_FP16 = MODEL_DIR / "moss_audio_tokenizer_decode_step.fp16.webgpu.onnx"
DECODER_FP16_DATA = MODEL_DIR / "moss_audio_tokenizer_decode_shared.fp16.data"

META_PATH = MODEL_DIR / "codec_browser_onnx_meta.json"
FLOAT_TYPES = {TensorProto.FLOAT, TensorProto.FLOAT16}

# ONNX Runtime WebGPU either requires these ops in fp32 or handles them better
# as fp32 in this MOSS export. Quantizer and RoPE nodes are blocked by name below.
FP32_OP_BLOCK = list(
    dict.fromkeys(
        float16.DEFAULT_OP_BLOCK_LIST
        + [
            "Range",
            "Div",
            "Pow",
            "Sqrt",
            "Exp",
            "Sin",
            "Cos",
            "ReduceMean",
            "Softmax",
            "TopK",
            "ArgMax",
            "ScatterND",
            "Where",
            "Less",
            "Greater",
            "GreaterOrEqual",
            "LessOrEqual",
            "Equal",
            "And",
            "Or",
            "Not",
            "NonZero",
            "Shape",
            "ConstantOfShape",
        ]
    )
)

FP32_OPS = {"Div", "Sqrt", "Pow", "Exp", "Sin", "Cos", "ReduceMean", "Softmax", "Range"}
HOMOGENEOUS_INPUTS = {
    "Add": [0, 1],
    "Sub": [0, 1],
    "Mul": [0, 1],
    "Div": [0, 1],
    "Pow": [0, 1],
    "MatMul": [0, 1],
    "Conv": [0, 1, 2],
    "LayerNormalization": [0, 1, 2],
    "Sqrt": [0],
    "Exp": [0],
    "Sin": [0],
    "Cos": [0],
    "Softmax": [0],
    "ReduceMean": [0],
    "Where": [1, 2],
    "Range": [0, 1, 2],
    "Concat": None,
}

BALANCED_BLOCK_PATTERNS = [
    re.compile(r"^/encoder\.1/input_proj/MatMul$"),
    re.compile(r"^/encoder\.1/transformer/layers\.0/norm1/LayerNormalization$"),
    re.compile(r"^/encoder\.1/transformer/layers\.0/self_attn/in_proj/MatMul$"),
    re.compile(r"^/encoder\.1/transformer/layers\.0/self_attn/Div$"),
    re.compile(r"^/out_proj(?:_\d+)?/Conv$"),
    re.compile(r"^/output_proj/Conv$"),
    re.compile(r"^/input_proj/MatMul$"),
    re.compile(r"^/norm1/LayerNormalization$"),
    re.compile(r"^/in_proj/MatMul$"),
    re.compile(r"^/Div$"),
]


def main() -> None:
    warnings.filterwarnings("ignore")
    # onnxconverter-common 1.16 can crash removing casts on these large graphs.
    float16.remove_unnecessary_cast_node = lambda graph: None

    convert_model(DECODER_FP32, DECODER_FP16, DECODER_FP16_DATA, fp16_decoder_cache=True)
    validate_candidate()


def convert_model(source: Path, target: Path, target_data: Path, fp16_decoder_cache: bool = False) -> None:
    base = onnx.load_model(source, load_external_data=False)
    node_block_list = precision_node_block_list(base)

    model = onnx.load_model(source, load_external_data=True)
    model = float16.convert_float_to_float16(
        model,
        keep_io_types=True,
        disable_shape_infer=True,
        op_block_list=FP32_OP_BLOCK,
        node_block_list=node_block_list,
        check_fp16_ready=False,
    )

    if fp16_decoder_cache:
        set_decoder_cache_io_type(model, TensorProto.FLOAT16)

    cast_count = harmonize_float_inputs(model)
    output_cast_count = preserve_graph_output_types(model)
    removed_noop_casts = remove_noop_casts(model)
    removed_cast_pairs = remove_reversible_cast_pairs(model)
    value_info_count = len(model.graph.value_info)
    del model.graph.value_info[:]

    target.unlink(missing_ok=True)
    target_data.unlink(missing_ok=True)
    onnx.save_model(
        model,
        target,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=target_data.name,
        size_threshold=1024,
    )
    target.chmod(0o644)
    target_data.chmod(0o644)

    check_session(target)
    print(
        f"{target.relative_to(ROOT)} {target.stat().st_size} bytes, "
        f"{target_data.relative_to(ROOT)} {target_data.stat().st_size} bytes, "
        f"casts {cast_count}, output casts {output_cast_count}, "
        f"removed noop casts {removed_noop_casts}, removed cast pairs {removed_cast_pairs}, "
        f"removed value_info {value_info_count}"
    )


def set_decoder_cache_io_type(model: onnx.ModelProto, elem_type: int) -> None:
    for value in list(model.graph.input) + list(model.graph.output):
        if (
            value.type.HasField("tensor_type")
            and "attn_cached_" in value.name
            and ("keys" in value.name or "values" in value.name)
        ):
            value.type.tensor_type.elem_type = elem_type


def precision_node_block_list(model: onnx.ModelProto) -> list[str]:
    blocked: set[str] = set()
    for node in model.graph.node:
        name = node.name.lower()
        if "/quantizer/" in name or "quantizer" in name or "/rope/" in name:
            blocked.add(node.name)
        elif any(pattern.match(node.name) for pattern in BALANCED_BLOCK_PATTERNS):
            blocked.add(node.name)
    return sorted(blocked)


def harmonize_float_inputs(model: onnx.ModelProto) -> int:
    total_casts = 0
    for round_index in range(10):
        value_types = collect_value_types(model)
        next_nodes = []
        round_casts = 0

        for node in model.graph.node:
            input_indexes = HOMOGENEOUS_INPUTS.get(node.op_type)
            if input_indexes is not None or node.op_type in HOMOGENEOUS_INPUTS:
                if input_indexes is None:
                    input_indexes = range(len(node.input))

                input_types = [
                    value_types.get(node.input[index])
                    for index in input_indexes
                    if index < len(node.input) and value_types.get(node.input[index]) in FLOAT_TYPES
                ]
                if input_types:
                    target_type = target_float_type(node, set(input_types))
                    for index in input_indexes:
                        if index >= len(node.input):
                            continue

                        input_name = node.input[index]
                        input_type = value_types.get(input_name)
                        if not input_name or input_type not in FLOAT_TYPES or input_type == target_type:
                            continue

                        round_casts += 1
                        cast_output = (
                            f"{input_name}_ciao_fp16_cast_"
                            f"{'fp16' if target_type == TensorProto.FLOAT16 else 'fp32'}_"
                            f"{round_index}_{round_casts}"
                        )
                        next_nodes.append(
                            helper.make_node(
                                "Cast",
                                [input_name],
                                [cast_output],
                                name=f"ciao/fp16_cast_{round_index}_{round_casts}",
                                to=target_type,
                            )
                        )
                        node.input[index] = cast_output

            next_nodes.append(node)

        del model.graph.node[:]
        model.graph.node.extend(next_nodes)
        total_casts += round_casts
        if round_casts == 0:
            break

    return total_casts


def target_float_type(node: onnx.NodeProto, input_types: set[int]) -> int:
    name = node.name.lower()
    if node.op_type in FP32_OPS or "/rope/" in name or "quantizer" in name:
        return TensorProto.FLOAT
    if node.op_type in {"MatMul", "Conv", "LayerNormalization"} and TensorProto.FLOAT16 in input_types:
        return TensorProto.FLOAT16
    if TensorProto.FLOAT in input_types and TensorProto.FLOAT16 in input_types:
        return TensorProto.FLOAT
    if TensorProto.FLOAT16 in input_types:
        return TensorProto.FLOAT16
    return TensorProto.FLOAT


def preserve_graph_output_types(model: onnx.ModelProto) -> int:
    value_types = collect_value_types(model)
    producer_by_output = {output: node for node in model.graph.node for output in node.output}
    appended_nodes = []
    cast_count = 0

    for graph_output in model.graph.output:
        if not graph_output.type.HasField("tensor_type"):
            continue

        expected_type = graph_output.type.tensor_type.elem_type
        actual_type = value_types.get(graph_output.name)
        if (
            expected_type not in FLOAT_TYPES
            or actual_type not in FLOAT_TYPES
            or expected_type == actual_type
            or graph_output.name not in producer_by_output
        ):
            continue

        producer = producer_by_output[graph_output.name]
        intermediate = f"{graph_output.name}_pre_output_cast_{cast_count}"
        for index, output_name in enumerate(producer.output):
            if output_name == graph_output.name:
                producer.output[index] = intermediate

        for node in model.graph.node:
            if node is producer:
                continue
            for index, input_name in enumerate(node.input):
                if input_name == graph_output.name:
                    node.input[index] = intermediate

        cast_count += 1
        appended_nodes.append(
            helper.make_node(
                "Cast",
                [intermediate],
                [graph_output.name],
                name=f"ciao/graph_output_cast_{cast_count}",
                to=expected_type,
            )
        )

    model.graph.node.extend(appended_nodes)
    return cast_count


def remove_noop_casts(model: onnx.ModelProto) -> int:
    removed_total = 0
    graph_outputs = {output.name for output in model.graph.output}

    for _ in range(20):
        value_types = collect_value_types(model)
        replacements: dict[str, str] = {}
        next_nodes = []
        removed = 0

        for node in model.graph.node:
            if (
                node.op_type == "Cast"
                and len(node.input) == 1
                and len(node.output) == 1
                and node.output[0] not in graph_outputs
                and value_types.get(node.input[0]) == value_types.get(node.output[0])
                and value_types.get(node.input[0]) in FLOAT_TYPES
            ):
                replacements[node.output[0]] = node.input[0]
                removed += 1
                continue

            for index, input_name in enumerate(node.input):
                while input_name in replacements:
                    input_name = replacements[input_name]
                node.input[index] = input_name
            next_nodes.append(node)

        del model.graph.node[:]
        model.graph.node.extend(next_nodes)
        removed_total += removed
        if removed == 0:
            break

    return removed_total


def remove_reversible_cast_pairs(model: onnx.ModelProto) -> int:
    removed_total = 0
    graph_outputs = {output.name for output in model.graph.output}

    for _ in range(20):
        value_types = collect_value_types(model)
        producer_by_output = {output: node for node in model.graph.node for output in node.output}
        consumers_by_input: dict[str, list[onnx.NodeProto]] = {}
        for node in model.graph.node:
            for input_name in node.input:
                consumers_by_input.setdefault(input_name, []).append(node)

        nodes_to_remove: set[str] = set()
        replacements: dict[str, str] = {}

        for node in model.graph.node:
            if node.op_type != "Cast" or len(node.input) != 1 or len(node.output) != 1:
                continue

            previous = producer_by_output.get(node.input[0])
            if (
                not previous
                or previous.op_type != "Cast"
                or len(previous.input) != 1
                or len(previous.output) != 1
                or previous.output[0] in graph_outputs
                or node.output[0] in graph_outputs
                or len(consumers_by_input.get(previous.output[0], [])) != 1
            ):
                continue

            if (
                value_types.get(previous.input[0]) in FLOAT_TYPES
                and value_types.get(node.output[0]) == value_types.get(previous.input[0])
            ):
                nodes_to_remove.add(previous.name)
                nodes_to_remove.add(node.name)
                replacements[node.output[0]] = previous.input[0]

        if not nodes_to_remove:
            break

        next_nodes = []
        for node in model.graph.node:
            for index, input_name in enumerate(node.input):
                while input_name in replacements:
                    input_name = replacements[input_name]
                node.input[index] = input_name
            if node.name not in nodes_to_remove:
                next_nodes.append(node)

        del model.graph.node[:]
        model.graph.node.extend(next_nodes)
        removed_total += len(nodes_to_remove)

    return removed_total


def collect_value_types(model: onnx.ModelProto) -> dict[str, int]:
    try:
        inferred = onnx.shape_inference.infer_shapes(model, strict_mode=False, data_prop=False)
    except Exception:
        inferred = model

    value_types: dict[str, int] = {}
    for value_info in list(inferred.graph.input) + list(inferred.graph.output) + list(inferred.graph.value_info):
        if value_info.type.HasField("tensor_type"):
            value_types[value_info.name] = value_info.type.tensor_type.elem_type

    for initializer in inferred.graph.initializer:
        value_types[initializer.name] = initializer.data_type

    for node in inferred.graph.node:
        if node.op_type == "Constant":
            for attribute in node.attribute:
                if attribute.name == "value" and attribute.HasField("t"):
                    for output in node.output:
                        value_types[output] = attribute.t.data_type
        elif node.op_type == "Cast":
            cast_type = next((attribute.i for attribute in node.attribute if attribute.name == "to"), None)
            if cast_type is not None:
                for output in node.output:
                    value_types[output] = cast_type

    return value_types


def check_session(model_path: Path) -> None:
    onnx.checker.check_model(str(model_path))
    session_options = ort.SessionOptions()
    session_options.log_severity_level = 3
    ort.InferenceSession(str(model_path), sess_options=session_options, providers=["CPUExecutionProvider"])


def validate_candidate() -> None:
    meta = json.loads(META_PATH.read_text())
    session_options = ort.SessionOptions()
    session_options.log_severity_level = 3

    enc32 = ort.InferenceSession(str(ENCODER_FP32), sess_options=session_options, providers=["CPUExecutionProvider"])
    dec32 = ort.InferenceSession(str(DECODER_FP32), sess_options=session_options, providers=["CPUExecutionProvider"])
    dec16 = ort.InferenceSession(str(DECODER_FP16), sess_options=session_options, providers=["CPUExecutionProvider"])

    sample_rate = int(meta["codec_config"]["sample_rate"])
    channels = int(meta["codec_config"]["channels"])
    samples = int(meta["codec_config"]["downsample_rate"]) * 2
    waveform = np.zeros((1, channels, samples), dtype=np.float32)

    for sample in range(samples):
        t = sample / sample_rate
        value = (
            math.sin(2 * math.pi * 220 * t) * 0.12
            + math.sin(2 * math.pi * 440 * t) * 0.04
            + math.sin(2 * math.pi * 730 * t) * 0.015
        )
        waveform[0, :, sample] = value

    encoder_feeds = {
        "waveform": waveform,
        "input_lengths": np.array([samples], dtype=np.int32),
    }
    codes32, lengths32 = enc32.run(None, encoder_feeds)
    decoder_feeds32 = decoder_feeds(meta, codes32, lengths32)
    decoder_feeds16 = decoder_feeds(meta, codes32, lengths32, cache_dtype=np.float16)
    audio32 = dec32.run(None, decoder_feeds32)[0].astype(np.float32)
    audio16 = dec16.run(None, decoder_feeds16)[0].astype(np.float32)

    error = audio32 - audio16
    rmse = float(np.sqrt(np.mean(error * error)))
    snr = float(10 * np.log10((np.mean(audio32 * audio32) + 1e-12) / (np.mean(error * error) + 1e-12)))

    if snr < 40:
        raise RuntimeError(f"fp16 decoder SNR too low: {snr:.1f} dB")

    print(f"fp16 validation: decoder rmse {rmse:.6f}, snr {snr:.1f} dB")


def decoder_feeds(
    meta: dict,
    codes: np.ndarray,
    lengths: np.ndarray,
    cache_dtype: np.dtype = np.float32,
) -> dict[str, np.ndarray]:
    feeds: dict[str, np.ndarray] = {
        "audio_codes": codes.astype(np.int32),
        "audio_code_lengths": lengths.astype(np.int32),
    }

    for offset in meta["streaming_decode"]["transformer_offsets"]:
        feeds[offset["input_name"]] = np.zeros(offset["shape"], dtype=np.int32)

    for cache in meta["streaming_decode"]["attention_caches"]:
        feeds[cache["offset_input_name"]] = np.zeros(cache["offset_shape"], dtype=np.int32)
        feeds[cache["cached_keys_input_name"]] = np.zeros(cache["cache_shape"], dtype=cache_dtype)
        feeds[cache["cached_values_input_name"]] = np.zeros(cache["cache_shape"], dtype=cache_dtype)
        feeds[cache["cached_positions_input_name"]] = np.zeros(cache["positions_shape"], dtype=np.int32)

    return feeds


if __name__ == "__main__":
    main()
