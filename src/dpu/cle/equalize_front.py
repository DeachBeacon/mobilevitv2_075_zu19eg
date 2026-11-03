"""Export a separate FP32 cross-layer-equalized front, with residual-safe scales.

Only parameters change after inference BN folding. Input/output scales are fixed
to one. Native Vitis re-quantization and accuracy evaluation remain mandatory.
"""
from __future__ import print_function

import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_paths import PROJECT, ROOT, SOURCE, FRONT, TAIL, NUMPY_GRAPH, CALIB, CLE, SAMPLES, SPLIT_MANIFEST, configure_imports
configure_imports()

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import numpy as np
import tensorflow as tf
from common import (ROOT, SOURCE_INPUT, CUT_TENSOR, OUTPUT, IMAGE_SHAPE, FEATURE_SHAPE,
                    LOGITS_SHAPE, DEFAULT_CALIB_DIR, ImageReader, cpu_config, import_graph,
                    read_graph, sha256, tf1, validate_interface)
from ps_tail import PSTail


def edge_node(edge):
    return edge.lstrip("^").split(":")[0]


def add_constant(graph_def, name, array):
    if any(n.name == name for n in graph_def.node):
        raise ValueError("Parameter name already exists: {}".format(name))
    node = graph_def.node.add()
    node.name, node.op = name, "Const"
    node.attr["dtype"].type = tf.float32.as_datatype_enum
    node.attr["value"].tensor.CopyFrom(tf.make_tensor_proto(np.asarray(array, dtype=np.float32)))
    return name


def fold_front(front_def, output_tensor=CUT_TENSOR):
    """Fold Conv/DWConv -> inference BN, cloning all changed parameters."""
    graph = import_graph(front_def)
    folded = tf1.GraphDef()
    folded.CopyFrom(front_def)
    nodes = {n.name: n for n in folded.node}
    layers = []
    with tf1.Session(graph=graph, config=cpu_config()) as session:
        for original in front_def.node:
            if original.op not in ("Conv2D", "DepthwiseConv2dNative"):
                continue
            op = graph.get_operation_by_name(original.name)
            if op.get_attr("data_format") != b"NHWC":
                raise ValueError("Only NHWC is supported")
            weight = session.run(op.inputs[1])
            consumers = list(op.outputs[0].consumers())
            batchnorms = [c for c in consumers if c.type == "FusedBatchNormV3"]
            if batchnorms and (len(batchnorms) != 1 or len(consumers) != 1):
                raise ValueError("BN folding would change another consumer: {}".format(op.name))
            channels = weight.shape[2] * weight.shape[3] if original.op == "DepthwiseConv2dNative" else weight.shape[3]
            bias = np.zeros(channels, dtype=np.float32)
            bn_name = None
            if batchnorms:
                bn = batchnorms[0]
                if bn.get_attr("is_training") or bn.inputs[0].name != op.outputs[0].name:
                    raise ValueError("Only direct inference BN is supported")
                if any(output.consumers() for output in bn.outputs[1:]):
                    raise ValueError("BN auxiliary outputs are in use: {}".format(bn.name))
                gamma, beta, mean, variance = session.run(bn.inputs[1:5])
                alpha = gamma / np.sqrt(variance + np.float32(bn.get_attr("epsilon")))
                factor = alpha.reshape(1, 1, weight.shape[2], weight.shape[3]) if original.op == "DepthwiseConv2dNative" else alpha.reshape(1, 1, 1, -1)
                weight = weight * factor
                bias = beta - mean * alpha
                bn_name = bn.name
            elif original.name != output_tensor.split(":")[0]:
                raise ValueError("Unexpected non-final convolution without BN: {}".format(original.name))
            index = len(layers)
            weight_name = add_constant(folded, "cle_params/layer_{:02d}/weights".format(index), weight)
            nodes[original.name].input[1] = weight_name
            bias_name = None
            if bn_name:
                bias_name = add_constant(folded, "cle_params/layer_{:02d}/bias".format(index), bias)
                replacement = nodes[bn_name]
                replacement.op = "BiasAdd"
                replacement.ClearField("input")
                replacement.input.extend([original.name, bias_name])
                replacement.ClearField("attr")
                replacement.attr["T"].type = tf.float32.as_datatype_enum
                replacement.attr["data_format"].s = b"NHWC"
            layers.append({"name": original.name, "op": original.op, "input": edge_node(original.input[0]),
                           "weight_node": weight_name, "bias_node": bias_name,
                           "weight": weight.astype(np.float64), "bias": bias.astype(np.float64)})
    folded = tf1.graph_util.extract_sub_graph(folded, [output_tensor.split(":")[0]])
    for node in folded.node:
        node.device = ""
    return folded, layers


class ChannelGroups(object):
    def __init__(self):
        self.parents = {}
        self.channels = {}

    def add(self, name, channels):
        if name not in self.parents:
            self.parents[name] = name
            self.channels[name] = channels

    def find(self, name):
        if self.parents[name] != name:
            self.parents[name] = self.find(self.parents[name])
        return self.parents[name]

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if self.channels[a] != self.channels[b]:
            raise ValueError("Channel mismatch at scale constraint")
        if a != b:
            first, second = sorted([a, b])
            self.parents[second] = first


def feature_groups(graph_def, layers, input_tensor=SOURCE_INPUT, output_tensor=CUT_TENSOR):
    graph = import_graph(graph_def)
    groups = ChannelGroups()
    visited = set()
    additions = []

    def visit(name):
        if name in visited:
            return
        op = graph.get_operation_by_name(name)
        tensor = op.outputs[0]
        dims = tensor.shape.as_list()
        if tensor.dtype != tf.float32 or len(dims) != 4 or dims[-1] is None:
            raise ValueError("Expected fixed-channel NHWC feature: {}".format(name))
        groups.add(name, dims[-1])
        visited.add(name)
        if op.type == "Placeholder":
            if tensor.name != input_tensor:
                raise ValueError("Unexpected runtime input: {}".format(name))
        elif op.type in ("Conv2D", "DepthwiseConv2dNative"):
            visit(op.inputs[0].op.name)
        elif op.type in ("BiasAdd", "Identity", "Relu", "Pad"):
            parent = op.inputs[0].op.name
            visit(parent)
            groups.union(name, parent)
        elif op.type in ("Add", "AddV2"):
            branches = [t.op.name for t in op.inputs]
            for parent in branches:
                visit(parent)
                groups.union(name, parent)
            additions.append({"node": name, "branches": branches})
        else:
            raise ValueError("Unsupported feature operator: {} [{}]".format(name, op.type))

    visit(output_tensor.split(":")[0])
    members = defaultdict(list)
    for name in sorted(visited):
        members[groups.find(name)].append(name)
    pinned = {groups.find(input_tensor.split(":")[0]), groups.find(output_tensor.split(":")[0])}
    for layer in layers:
        layer["input_group"] = groups.find(layer["input"])
        layer["output_group"] = groups.find(layer["name"])
        if layer["input_group"] == layer["output_group"]:
            raise ValueError("Self-coupled scale group requires a different optimizer: {}".format(layer["name"]))
    return groups, dict(members), pinned, additions


def scaled_weight(layer, scales):
    incoming, outgoing = scales[layer["input_group"]], scales[layer["output_group"]]
    if layer["op"] == "DepthwiseConv2dNative":
        factor = outgoing.reshape(layer["weight"].shape[2:]) / incoming[:, None]
        return layer["weight"] * factor[None, None, :, :]
    return layer["weight"] * outgoing[None, None, None, :] / incoming[None, None, :, None]


def input_ranges(layer, weight):
    return np.max(np.abs(weight), axis=(0, 1, 3))


def output_ranges(layer, weight):
    if layer["op"] == "DepthwiseConv2dNative":
        return np.max(np.abs(weight), axis=(0, 1)).reshape(-1)
    return np.max(np.abs(weight), axis=(0, 1, 2))


def equalize(layers, groups, members, pinned, iterations=64, damping=0.5):
    scales = {name: np.ones(groups.channels[name], dtype=np.float64) for name in members}
    history = []
    for iteration in range(iterations):
        largest = 0.0
        for group in members:
            if group in pinned:
                continue
            producers = [l for l in layers if l["output_group"] == group]
            consumers = [l for l in layers if l["input_group"] == group]
            if not producers or not consumers:
                raise ValueError("Unanchored channel scale: {}".format(group))
            before = np.maximum.reduce([output_ranges(l, scaled_weight(l, scales)) for l in producers])
            after = np.maximum.reduce([input_ranges(l, scaled_weight(l, scales)) for l in consumers])
            valid = (before > 1e-12) & (after > 1e-12)
            log_adjustment = np.zeros_like(before)
            log_adjustment[valid] = 0.5 * damping * (np.log(after[valid]) - np.log(before[valid]))
            updated = np.clip(scales[group] * np.exp(np.clip(log_adjustment, -np.log(4), np.log(4))), 1e-4, 1e4)
            largest = max(largest, float(np.max(np.abs(np.log(updated / scales[group])))))
            scales[group] = updated
        history.append({"iteration": iteration + 1, "largest_log_scale_update": largest})
        if largest < 1e-4:
            break
    return scales, history


def apply_scales(graph_def, layers, scales):
    candidate = tf1.GraphDef()
    candidate.CopyFrom(graph_def)
    nodes = {n.name: n for n in candidate.node}
    stats = []
    for layer in layers:
        weight = np.asarray(scaled_weight(layer, scales), dtype=np.float32)
        bias = np.asarray(layer["bias"] * scales[layer["output_group"]], dtype=np.float32)
        if not np.isfinite(weight).all() or not np.isfinite(bias).all():
            raise ValueError("Non-finite equalized parameters")
        nodes[layer["weight_node"]].attr["value"].tensor.CopyFrom(tf.make_tensor_proto(weight))
        if layer["bias_node"]:
            nodes[layer["bias_node"]].attr["value"].tensor.CopyFrom(tf.make_tensor_proto(bias))
        elif np.any(bias):
            raise ValueError("Final output unexpectedly requires a new bias")
        stats.append({"name": layer["name"], "op": layer["op"],
                      "before_weight_abs_max": float(np.max(np.abs(layer["weight"]))),
                      "after_weight_abs_max": float(np.max(np.abs(weight))),
                      "before_bias_abs_max": float(np.max(np.abs(layer["bias"]))),
                      "after_bias_abs_max": float(np.max(np.abs(bias)))})
    return candidate, stats


def verify(original, folded, candidate, tail_path, image_dir, samples):
    full_graph, folded_graph, candidate_graph = [import_graph(g) for g in (original, folded, candidate)]
    images = sorted(p for p in image_dir.rglob("*") if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    if not images:
        raise ValueError("No verification images found")
    count = min(samples, len(images)) if samples else len(images)
    indices = np.linspace(0, len(images) - 1, count, dtype=int)
    results = []
    random_checks = []
    reader = ImageReader()
    first_sample = None
    tolerance = {"rtol": 1e-4, "atol": 1e-4}

    def feature_check(actual, reference):
        delta = actual.astype(np.float64) - reference.astype(np.float64)
        maximum = float(np.max(np.abs(delta)))
        denominator = float(np.linalg.norm(reference.astype(np.float64)))
        relative = float(np.linalg.norm(delta) / denominator) if denominator else (0.0 if not np.any(delta) else float("inf"))
        # BN folding itself changes FP32 accumulation/cancellation. Bound both
        # absolute error and tensor-relative L2 error, rather than dividing by
        # individual near-zero entries. Logits and classes are checked separately.
        return maximum, relative, bool(np.isfinite(actual).all() and maximum <= 1e-3 and relative <= 1e-5)
    try:
        with tf1.Session(graph=full_graph, config=cpu_config()) as full_session, \
                tf1.Session(graph=folded_graph, config=cpu_config()) as folded_session, \
                tf1.Session(graph=candidate_graph, config=cpu_config()) as candidate_session, \
                PSTail(tail_path) as tail:
            for number, index in enumerate(indices):
                path = images[index]
                image = reader.read(path)
                reference, reference_logits = full_session.run([CUT_TENSOR, OUTPUT], {SOURCE_INPUT: image})
                actual = candidate_session.run(CUT_TENSOR, {SOURCE_INPUT: image})
                logits = tail.predict(actual)
                same_class = bool(logits.argmax() == reference_logits.argmax())
                maximum, relative, feature_passed = feature_check(actual, reference)
                result = {"image": path.relative_to(image_dir).as_posix(),
                          "feature_max_abs_error": maximum, "feature_relative_l2_error": relative,
                          "logit_max_abs_error": float(np.max(np.abs(logits - reference_logits))),
                          "reference_class": int(reference_logits.argmax()), "candidate_class": int(logits.argmax()),
                          "passed": bool(same_class and feature_passed
                                         and np.allclose(logits, reference_logits, **tolerance))}
                results.append(result)
                if number == 0:
                    folded_feature = folded_session.run(CUT_TENSOR, {SOURCE_INPUT: image})
                    first_sample = (image, reference, actual, reference_logits, logits)
                    result["folded_feature_max_abs_error"] = float(np.max(np.abs(folded_feature - reference)))
                if number == 0 or (number + 1) % 25 == 0 or number + 1 == count:
                    print("[{}/{}] {} feature_error={:.3g} logit_error={:.3g} passed={}".format(
                        number + 1, count, result["image"], result["feature_max_abs_error"], result["logit_max_abs_error"], result["passed"]), flush=True)
            rng = np.random.RandomState(20261029)
            for _ in range(3):
                image = rng.uniform(-2.2, 2.7, IMAGE_SHAPE).astype(np.float32)
                reference = full_session.run(CUT_TENSOR, {SOURCE_INPUT: image})
                actual = candidate_session.run(CUT_TENSOR, {SOURCE_INPUT: image})
                maximum, relative, feature_passed = feature_check(actual, reference)
                random_checks.append({"feature_max_abs_error": maximum,
                                      "feature_relative_l2_error": relative, "passed": feature_passed})
    finally:
        reader.close()
    return {"passed": all(r["passed"] for r in results + random_checks), "sample_count": count,
            "logit_tolerance": tolerance, "feature_tolerance": {"max_abs_error": 1e-3, "relative_l2_error": 1e-5},
            "max_feature_abs_error": max(r["feature_max_abs_error"] for r in results),
            "max_feature_relative_l2_error": max(r["feature_relative_l2_error"] for r in results),
            "max_logit_abs_error": max(r["logit_max_abs_error"] for r in results),
            "class_agreement": sum(r["reference_class"] == r["candidate_class"] for r in results) / float(count),
            "results": results, "random_checks": random_checks,
            "scope": "Host FP32 equivalence only; INT8/native FixNeuron/DPU accuracy not validated"}, first_sample


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-pb", type=Path, default=SOURCE)
    parser.add_argument("--tail-pb", type=Path, default=TAIL)
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_CALIB_DIR)
    parser.add_argument("--output-dir", type=Path, default=CLE)
    parser.add_argument("--samples", type=int, default=0, help="0 verifies all images")
    parser.add_argument("--iterations", type=int, default=64)
    parser.add_argument("--damping", type=float, default=0.5)
    args = parser.parse_args()
    if args.samples < 0 or args.iterations < 1 or not 0 < args.damping <= 1:
        parser.error("Invalid samples/iterations/damping")
    if args.output_dir.exists():
        parser.error("Output directory already exists; choose a new --output-dir")
    original, original_data = read_graph(args.source_pb)
    validate_interface(import_graph(original), SOURCE_INPUT, OUTPUT, IMAGE_SHAPE, LOGITS_SHAPE)
    front = tf1.graph_util.extract_sub_graph(original, [CUT_TENSOR.split(":")[0]])
    folded, layers = fold_front(front)
    groups, members, pinned, additions = feature_groups(folded, layers)
    scales, history = equalize(layers, groups, members, pinned, args.iterations, args.damping)
    candidate, parameter_stats = apply_scales(folded, layers, scales)
    validate_interface(import_graph(candidate), SOURCE_INPUT, CUT_TENSOR, IMAGE_SHAPE, FEATURE_SHAPE)
    verification, sample = verify(original, folded, candidate, args.tail_pb, args.image_dir, args.samples)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "fp32").mkdir()
    (args.output_dir / "records").mkdir()
    (args.output_dir / "samples").mkdir()
    (args.output_dir / "records/fp32_verification.json").write_text(json.dumps(verification, indent=2), encoding="utf-8")
    if not verification["passed"]:
        raise SystemExit("FP32 equivalence failed; candidate PB was not exported. See fp32_verification.json")
    filename = "dpu_front_cle_fp32.pb"
    data = candidate.SerializeToString()
    (args.output_dir / "fp32" / filename).write_bytes(data)
    group_info = [{"group": group, "members": names, "pinned": group in pinned,
                   "scales": scales[group].tolist()} for group, names in members.items()]
    manifest = {"format_version": 1, "source_file": str(args.source_pb.resolve()), "source_sha256": sha256(original_data),
                "tail_sha256": sha256(args.tail_pb.read_bytes()), "candidate_file": filename,
                "candidate_sha256": sha256(data), "input_tensor": SOURCE_INPUT, "output_tensor": CUT_TENSOR,
                "input_shape": IMAGE_SHAPE, "output_shape": FEATURE_SHAPE, "layout": "NHWC",
                "ops": dict(Counter(n.op for n in candidate.node)), "conv_count": len(layers),
                "folded_bn_count": sum(l["bias_node"] is not None for l in layers),
                "parameter_stats": parameter_stats, "channel_groups": group_info,
                "residual_additions": additions, "equalization_history": history,
                "fp32_verification_passed": True, "int8_accuracy_validated": False,
                "method": "Residual-constrained coordinate channel range equalization; positive parameter scales only"}
    (args.output_dir / "records/manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    for name, array in zip(("sample_input_fp32", "reference_features", "cle_features", "reference_logits", "cle_logits"), sample):
        np.save(str(args.output_dir / "samples" / (name + ".npy")), array, allow_pickle=False)
    print("Exported:", args.output_dir / "fp32" / filename)
    print(json.dumps({k: verification[k] for k in ("sample_count", "max_feature_abs_error", "max_logit_abs_error", "class_agreement", "passed")}, indent=2))
    print("Re-quantize and evaluate with native Vitis FixNeuron before compiling this candidate.")


if __name__ == "__main__":
    main()
