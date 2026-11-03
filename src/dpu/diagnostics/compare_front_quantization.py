"""Compare FP32, quantizer-folded FP32, and quantized-front + FP32-tail logits.

--backend native must run in Vitis AI 1.2 with tensorflow.contrib.decent_q.
--backend emulated is a host diagnostic: fixed-position rounding/clipping only;
it is NOT a bit-exact replacement for the Vitis quantizer or DPU arithmetic.
Original models are read only. Only --output and its sample directory are written.
"""
from __future__ import print_function

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
sys.dont_write_bytecode = True
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_paths import PROJECT, ROOT, SOURCE, FRONT, TAIL, NUMPY_GRAPH, CALIB, CLE, SAMPLES, SPLIT_MANIFEST, configure_imports
configure_imports()


import numpy as np
import tensorflow as tf
from common import CUT_TENSOR, OUTPUT, SOURCE_INPUT, ImageReader, cpu_config, import_graph, read_graph
from ps_tail import PSTail


def transform_fix_neurons(original, quantize, skip_names=(), position_overrides=None):
    """Use standard TF ops for inspection of the frozen, fixed-position graph."""
    result = tf.GraphDef()
    result.CopyFrom(original)
    del result.node[:]
    fixes = []
    for original_node in original.node:
        if original_node.op != "FixNeuron":
            result.node.add().CopyFrom(original_node)
            continue
        node = original_node
        position = (position_overrides or {}).get(node.name, int(node.attr["quantize_pos"].i))
        bits = int(node.attr["bit_width"].i)
        fixes.append({"name": node.name, "position": position, "bits": bits,
                      "mode": int(node.attr["mode"].i)})
        if not quantize or node.name in skip_names:
            replacement = result.node.add()
            replacement.name, replacement.op = node.name, "Identity"
            replacement.input.extend(node.input)
            replacement.attr["T"].type = tf.float32.as_datatype_enum
            continue
        # Fresh names cannot collide with the original graph. Keep the final tensor name.
        prefix = "host_fixed_quant_{}".format(len(fixes))
        graph = tf.Graph()
        with graph.as_default():
            with tf.name_scope(prefix):
                x = tf.placeholder(tf.float32, name="input")
                scale = tf.constant(float(2.0 ** position), dtype=tf.float32, name="scale")
                rounded = tf.floor(x * scale + np.float32(0.5), name="round_half_up")
                clipped = tf.clip_by_value(rounded, -float(2 ** (bits - 1)), float(2 ** (bits - 1) - 1),
                                           name="clip")
            tf.divide(clipped, scale, name=node.name)
        for generated in graph.as_graph_def().node:
            if generated.name == prefix + "/input":
                continue
            replacement = result.node.add()
            replacement.CopyFrom(generated)
            for index, edge in enumerate(replacement.input):
                if edge in (prefix + "/input", prefix + "/input:0"):
                    replacement.input[index] = node.input[0]
    return result, fixes


def metrics(actual, reference):
    delta = actual.astype(np.float64) - reference.astype(np.float64)
    rmse = float(np.sqrt(np.mean(delta ** 2)))
    norm = float(np.sqrt(np.mean(reference.astype(np.float64) ** 2)))
    a, b = actual.reshape(-1).astype(np.float64), reference.reshape(-1).astype(np.float64)
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    return {"max_abs_error": float(np.max(np.abs(delta))), "rmse": rmse,
            "normalized_rmse": rmse / norm if norm else None,
            "cosine_similarity": float(np.dot(a, b) / denominator) if denominator else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("native", "emulated"), default="native")
    parser.add_argument("--source-pb", type=Path, default=SOURCE)
    parser.add_argument("--quantized-pb", type=Path, default=ROOT / "models/front_baseline/int8/quantize_eval_model.pb")
    parser.add_argument("--tail-pb", type=Path, default=TAIL)
    parser.add_argument("--image-dir", type=Path, default=CALIB)
    parser.add_argument("--samples", type=int, default=10, help="0 means all images")
    parser.add_argument("--selection", choices=("uniform", "first"), default="uniform")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.samples < 0:
        parser.error("--samples must be nonnegative")
    if args.output.exists():
        parser.error("--output already exists; choose a new report filename")
    if args.backend == "native":
        try:
            from tensorflow.contrib import decent_q
        except ImportError:
            raise SystemExit("Native evaluation requires the Vitis AI 1.2 vitis-ai-tensorflow environment")
    original, source_data = read_graph(args.source_pb)
    quantized, quantized_data = read_graph(args.quantized_pb)
    folded, fixes = transform_fix_neurons(quantized, quantize=False)
    evaluated = transform_fix_neurons(quantized, quantize=True)[0] if args.backend == "emulated" else quantized
    full_graph, folded_graph, evaluated_graph = [import_graph(g) for g in (original, folded, evaluated)]
    output_node = CUT_TENSOR.rsplit(":", 1)[0] + "/aquant:0"
    images = sorted(p for p in args.image_dir.rglob("*") if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    if not images:
        raise ValueError("No images found in {}".format(args.image_dir))
    count = min(args.samples, len(images)) if args.samples else len(images)
    indices = np.linspace(0, len(images) - 1, count, dtype=int) if args.selection == "uniform" else range(count)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sample_directory = args.output.parent / (args.output.stem + "_sample")
    sample_directory.mkdir(exist_ok=False)
    reader = ImageReader()
    results = []
    try:
        with tf.Session(graph=full_graph, config=cpu_config()) as full_session, \
                tf.Session(graph=folded_graph, config=cpu_config()) as folded_session, \
                tf.Session(graph=evaluated_graph, config=cpu_config()) as quant_session, \
                PSTail(args.tail_pb) as tail:
            for number, index in enumerate(indices):
                path = images[index]
                image = reader.read(path)
                features, logits = full_session.run([CUT_TENSOR, OUTPUT], {SOURCE_INPUT: image})
                folded_features = folded_session.run(output_node, {SOURCE_INPUT: image})
                quant_features = quant_session.run(output_node, {SOURCE_INPUT: image})
                folded_logits = tail.predict(folded_features)
                quant_logits = tail.predict(quant_features)
                name = path.relative_to(args.image_dir).as_posix()
                classes = {"c{}".format(i): i for i in range(10)}
                label = classes.get(path.relative_to(args.image_dir).parts[0])
                result = {"image": name, "label": label, "float_class": int(logits.argmax()),
                          "folded_float_class": int(folded_logits.argmax()), "quantized_front_class": int(quant_logits.argmax()),
                          "folded_features": metrics(folded_features, features),
                          "quantized_features": metrics(quant_features, features),
                          "folded_logits_max_error": float(np.max(np.abs(folded_logits - logits))),
                          "quantized_logits_max_error": float(np.max(np.abs(quant_logits - logits)))}
                results.append(result)
                print("[{}/{}] {} label={} FP32={} folded={} {}={}".format(
                    number + 1, count, name, label, result["float_class"], result["folded_float_class"],
                    args.backend, result["quantized_front_class"]), flush=True)
                if number == 0:
                    for filename, array in [("sample_input_fp32.npy", image), ("float_features.npy", features),
                                            ("quantized_features.npy", quant_features), ("float_logits.npy", logits),
                                            ("quantized_logits.npy", quant_logits)]:
                        np.save(str(sample_directory / filename), array, allow_pickle=False)
    finally:
        reader.close()
    labeled = [r for r in results if r["label"] is not None]
    report = {"backend": args.backend, "sample_count": len(results), "source_sha256": hashlib.sha256(source_data).hexdigest(),
              "quantized_graph_sha256": hashlib.sha256(quantized_data).hexdigest(), "fixed_positions": fixes,
              "tail_sha256": hashlib.sha256(args.tail_pb.read_bytes()).hexdigest(),
              "image_dir": str(args.image_dir.resolve()), "tensorflow_version": tf.__version__,
              "label_mapping": "cN -> model output index N", "results": results,
              "scope": "Native FixNeuron evaluation" if args.backend == "native" else
              "Host fixed-point rounding/clipping approximation; NOT bit-exact Vitis or DPU validation",
              "class_agreement_with_float": sum(r["quantized_front_class"] == r["float_class"] for r in results) / float(len(results)),
              "predicted_class_histogram": dict(Counter(r["quantized_front_class"] for r in results)),
              "accuracy": {key: sum(r[key] == r["label"] for r in labeled) / float(len(labeled)) if labeled else None
                           for key in ("float_class", "folded_float_class", "quantized_front_class")}}
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("backend", "sample_count", "scope", "accuracy", "predicted_class_histogram")}, indent=2))
    print("Saved:", args.output)


if __name__ == "__main__":
    main()
