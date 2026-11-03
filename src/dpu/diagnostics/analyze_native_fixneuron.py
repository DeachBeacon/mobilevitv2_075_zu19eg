"""Read native FixNeuron evidence and run HOST-ONLY ablations for localization.

Bypasses and position overrides below are experiments, not deployable models.
In particular, pre-adjustment positions violate the recorded DPU shift limits.
"""
from __future__ import print_function

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
from tensorflow.python.framework import tensor_util
from common import CUT_TENSOR, SOURCE_INPUT, ImageReader, cpu_config, import_graph, read_graph
from compare_front_quantization import metrics, transform_fix_neurons
from ps_tail import PSTail


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    native_path = ROOT / "diagnostics/native_quant_accuracy_10.json"
    native = json.loads(native_path.read_text(encoding="utf-8"))
    assert native["backend"] == "native"
    source_path = SOURCE
    quant_path = ROOT / "models/front_baseline/int8/quantize_eval_model.pb"
    assert sha256(source_path) == native["source_sha256"]
    assert sha256(quant_path) == native["quantized_graph_sha256"]
    quantized, _ = read_graph(quant_path)
    output_name = CUT_TENSOR.rsplit(":", 1)[0] + "/aquant:0"
    fixes = [n for n in quantized.node if n.op == "FixNeuron"]
    names = {n.name for n in fixes}
    activations = {n.name for n in fixes if n.attr["mode"].i == 1}
    weights = names - activations
    depthwise = {n.name for n in fixes if n.attr["mode"].i == 2}
    warning_names = {prefix + "/depthwise_kernel/wquant" for prefix in
                     ("stack1_block1_deep_2_dw_conv", "stack2_block2_deep_2_dw_conv")}
    original_positions = {"stack1_block1_deep_2_dw_conv/depthwise_kernel/wquant": -2,
                          "stack2_block2_deep_2_dw_conv/depthwise_kernel/wquant": -4}
    configurations = [
        ("all_quantized_baseline", set(), None),
        ("weights_biases_only", activations, None),
        ("activations_only", weights, None),
        ("bypass_two_warning_weights", warning_names, None),
        ("bypass_all_depthwise_weights", depthwise, None),
        ("restore_pre_adjustment_positions_INVALID_FOR_DPU", set(), original_positions),
        ("only_two_warning_weights_quantized", names - warning_names, None),
        ("bypass_final_feature_quantizer", {output_name[:-2]}, None),
        ("only_final_feature_quantizer", names - {output_name[:-2]}, None),
    ]
    sample_dir = native_path.parent / (native_path.stem + "_sample")
    sample_input = np.load(str(sample_dir / "sample_input_fp32.npy"), allow_pickle=False)
    sample_reference = np.load(str(sample_dir / "float_features.npy"), allow_pickle=False)
    sample_native = np.load(str(sample_dir / "quantized_features.npy"), allow_pickle=False)
    report = {"native_report": str(native_path), "native_accuracy": native["accuracy"],
              "source_hash_matches": True, "quantized_graph_hash_matches": True,
              "native_mean_feature_normalized_rmse": float(np.mean([
                  r["quantized_features"]["normalized_rmse"] for r in native["results"]])),
              "native_mean_feature_cosine": float(np.mean([
                  r["quantized_features"]["cosine_similarity"] for r in native["results"]])),
              "ablation_scope": "Host approximations/bypasses only; not native or DPU repair validation", "ablations": {}}
    reader = ImageReader()
    try:
        images = [reader.read(CALIB / r["image"]) for r in native["results"]]
    finally:
        reader.close()
    folded_graph = import_graph(transform_fix_neurons(quantized, False)[0])
    with tf.Session(graph=folded_graph, config=cpu_config()) as session:
        reference_stages = session.run([n + ":0" for n in sorted(activations)], {SOURCE_INPUT: sample_input})
    with PSTail(TAIL) as tail:
        for name, skip, overrides in configurations:
            graph = import_graph(transform_fix_neurons(quantized, True, skip, overrides)[0])
            results = []
            with tf.Session(graph=graph, config=cpu_config()) as session:
                if name == "all_quantized_baseline":
                    sample = session.run(output_name, {SOURCE_INPUT: sample_input})
                    report["baseline_sample_matches_native_exactly"] = bool(np.array_equal(sample, sample_native))
                    report["baseline_sample_max_error_vs_native"] = float(np.max(np.abs(sample - sample_native)))
                    actual_stages = session.run([n + ":0" for n in sorted(activations)], {SOURCE_INPUT: sample_input})
                    report["first_sample_activation_errors"] = [
                        {"name": node_name, "shape": list(actual.shape), **metrics(actual, reference)}
                        for node_name, actual, reference in zip(sorted(activations), actual_stages, reference_stages)]
                for image, record in zip(images, native["results"]):
                    feature = session.run(output_name, {SOURCE_INPUT: image})
                    predicted = int(tail.predict(feature).argmax())
                    results.append({"image": record["image"], "label": record["label"], "predicted": predicted})
            report["ablations"][name] = {"correct": sum(r["label"] == r["predicted"] for r in results),
                                         "sample_count": len(results),
                                         "predicted_histogram": dict(Counter(r["predicted"] for r in results)),
                                         "results": results}
            print(name, report["ablations"][name]["correct"], "/", len(results), flush=True)
    nodes = {n.name: n for n in quantized.node}
    report["weight_range_stats"] = []
    for node in fixes:
        parent = nodes[node.input[0].split(":")[0]]
        if parent.op != "Const" or node.name not in weights:
            continue
        value = tensor_util.MakeNdarray(parent.attr["value"].tensor)
        p, bits = node.attr["quantize_pos"].i, node.attr["bit_width"].i
        step = float(2.0 ** -p)
        lower, upper = -(2 ** (bits - 1)) * step, (2 ** (bits - 1) - 1) * step
        rounded = np.floor(value / step + np.float32(0.5))
        clipped = np.clip(rounded, -(2 ** (bits - 1)), 2 ** (bits - 1) - 1) * step
        report["weight_range_stats"].append({
            "name": node.name, "position": int(p), "step": step, "elements": int(value.size),
            "min": float(value.min()), "max": float(value.max()), "range": [lower, upper],
            "outside_range_count": int(np.count_nonzero((value < lower) | (value > upper))),
            "outside_range_energy_fraction": float(np.sum(np.where((value < lower) | (value > upper), value, 0).astype(np.float64) ** 2)
                                                   / np.sum(value.astype(np.float64) ** 2)) if np.any(value) else 0.0,
            "rounded_zero_fraction": float(np.mean(clipped == 0)), **metrics(clipped, value)})
    report["warning_layers"] = {r["name"]: r for r in report["weight_range_stats"] if r["name"] in warning_names}
    path = ROOT / "diagnostics/native_fixneuron_analysis_current.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("Saved:", path)


if __name__ == "__main__":
    main()
