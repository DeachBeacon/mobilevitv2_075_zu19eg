"""Export an independent DPU-front FP32 PB and a PS-tail FP32 PB.

Use the original frozen gnfold PB, not Vitis deploy_model.pb.
The fixed cut is stack3_block2_pre_2_conv/Conv2D:0, shape [1,32,32,96].
"""
from __future__ import print_function

import argparse
import json
from pathlib import Path

from common import (CUT_TENSOR, DEFAULT_ARTIFACT_DIR, DEFAULT_SOURCE, FEATURE_SHAPE,
                    IMAGE_SHAPE, LOGITS_SHAPE, OUTPUT, SOURCE_INPUT, TAIL_INPUT,
                    graph_info, import_graph, read_graph, sha256, tf1, validate_interface)
import numpy as np
import tensorflow as tf
from export_numpy import prepare_numpy_export


def replace_edge(edge, old_name, new_name):
    control = "^" if edge.startswith("^") else ""
    parts = edge.lstrip("^").split(":", 1)
    if parts[0] != old_name:
        return edge
    return control + new_name + (":" + parts[1] if len(parts) == 2 else "")


def split_graph(original):
    full_graph = import_graph(original)
    validate_interface(full_graph, SOURCE_INPUT, OUTPUT, IMAGE_SHAPE, LOGITS_SHAPE)
    cut = full_graph.get_tensor_by_name(CUT_TENSOR)
    if cut.dtype != tf.float32 or cut.shape.as_list() != FEATURE_SHAPE:
        raise ValueError("Unexpected cut tensor: {} {}".format(cut.dtype, cut.shape))
    cut_name = CUT_TENSOR.split(":")[0]
    tail_name = TAIL_INPUT.split(":")[0]
    if any(n.name == tail_name for n in original.node):
        raise ValueError("Source already has a node named {}".format(tail_name))

    # All ancestors of the cut are retained, including frozen weights and BN data.
    front = tf1.graph_util.extract_sub_graph(original, [cut_name])

    # Replace the cut output by a new feature Placeholder, then prune the prefix.
    modified = tf1.GraphDef()
    modified.CopyFrom(original)
    for node in modified.node:
        if node.name == cut_name:
            node.name = tail_name
            node.op = "Placeholder"
            node.ClearField("input")
            node.ClearField("attr")
            node.device = ""
            node.attr["dtype"].type = tf.float32.as_datatype_enum
            node.attr["shape"].shape.CopyFrom(cut.shape.as_proto())
        else:
            for i, edge in enumerate(node.input):
                node.input[i] = replace_edge(edge, cut_name, tail_name)
    tail = tf1.graph_util.extract_sub_graph(modified, [OUTPUT.split(":")[0]])

    validate_interface(import_graph(front), SOURCE_INPUT, CUT_TENSOR, IMAGE_SHAPE, FEATURE_SHAPE)
    validate_interface(import_graph(tail), TAIL_INPUT, OUTPUT, FEATURE_SHAPE, LOGITS_SHAPE)
    forbidden = {"Transpose", "Rsqrt", "SquaredDifference", "Softmax", "SplitV", "Sum"}
    found = sorted({n.op for n in front.node} & forbidden)
    if found:
        raise ValueError("Unexpected operators remain in the CNN prefix: {}".format(found))
    return front, tail


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-pb", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    args = parser.parse_args()
    source_path = args.source_pb.resolve()
    original, source_data = read_graph(source_path)
    front, tail = split_graph(original)
    numpy_graph, numpy_weights = prepare_numpy_export(tail, TAIL_INPUT, OUTPUT)
    out_dir = args.output_dir.resolve()
    organized = out_dir == DEFAULT_ARTIFACT_DIR.resolve()
    outputs = [("front_baseline/fp32/dpu_front_fp32.pb" if organized else "dpu_front_fp32.pb", front),
               ("tail_shared/ps_tail_fp32.pb" if organized else "ps_tail_fp32.pb", tail)]
    if any((out_dir / name).resolve() == source_path for name, _ in outputs):
        raise ValueError("Output must not overwrite the source PB")
    manifest = {
        "format_version": 1, "tensorflow_version": tf.__version__,
        "source": {"file": str(source_path), "sha256": sha256(source_data)},
        "cut_tensor": CUT_TENSOR, "feature_shape": FEATURE_SHAPE,
        "dpu_front": graph_info(front, outputs[0][0], SOURCE_INPUT, CUT_TENSOR, IMAGE_SHAPE, FEATURE_SHAPE),
        "ps_tail": graph_info(tail, outputs[1][0], TAIL_INPUT, OUTPUT, FEATURE_SHAPE, LOGITS_SHAPE),
        "note": "FP32 graph export only. The front must be quantized and compiled before DPU execution.",
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, graph_def in outputs:
        (out_dir / name).parent.mkdir(parents=True, exist_ok=True)
        (out_dir / name).write_bytes(graph_def.SerializeToString())
        print("Exported:", out_dir / name)
    numpy_json = out_dir / ("tail_shared/ps_tail_numpy.json" if organized else "ps_tail_numpy.json")
    numpy_npz = out_dir / ("tail_shared/ps_tail_numpy_weights.npz" if organized else "ps_tail_numpy_weights.npz")
    numpy_json.write_text(json.dumps(numpy_graph, indent=2), encoding="utf-8")
    np.savez_compressed(str(numpy_npz), **numpy_weights)
    manifest["ps_tail_numpy"] = {
        "graph_file": numpy_json.relative_to(out_dir).as_posix(), "graph_sha256": sha256(numpy_json.read_bytes()),
        "weights_file": numpy_npz.relative_to(out_dir).as_posix(), "weights_sha256": sha256(numpy_npz.read_bytes()),
        "runtime_dependencies": "Python 3.7 + NumPy 1.17-compatible APIs; no TensorFlow",
    }
    print("Exported NumPy tail:", numpy_json, numpy_npz)
    manifest_path = out_dir / ("front_baseline/records/split_manifest.json" if organized else "manifest.json")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Front:", SOURCE_INPUT, "->", CUT_TENSOR, FEATURE_SHAPE)
    print("Tail: ", TAIL_INPUT, "->", OUTPUT, LOGITS_SHAPE)
    print("Each PB has exactly one Placeholder input. Run verify_split.py next.")


if __name__ == "__main__":
    main()
