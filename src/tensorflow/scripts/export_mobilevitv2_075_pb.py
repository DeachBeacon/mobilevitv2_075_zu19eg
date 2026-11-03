"""Export and verify a training-branch-free MobileViTv2 GraphDef in TF 1.15."""
import argparse
import json
import os
from _paths import PROJECT, TF_ROOT, TF_RECORDS, TF_WEIGHTS, INFERENCE, DEPLOY, REFERENCE_PB
import sys

import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
from keras_cv_attention_models import backend, mobilevit
from inspect_mobilevitv2_pb import inspect_graph, sha256
from verify_mobilevitv2_tf115 import inspect_loaded_h5


HERE = os.path.dirname(os.path.abspath(__file__))
ARTIFACT_DIR = str(TF_RECORDS)
WEIGHTS = str(TF_WEIGHTS)
REFERENCE = os.path.join(ARTIFACT_DIR, "pytorch_reference.npz")
BASELINE_PB = str(REFERENCE_PB)
PB_PATH = str(INFERENCE)
REPORT_PATH = os.path.join(ARTIFACT_DIR, "pure_inference_export_report.json")


def read_graph(path):
    graph_def = backend.tf.GraphDef()
    with backend.tf.gfile.GFile(path, "rb") as source:
        graph_def.ParseFromString(source.read())
    return graph_def


def run_graph(graph_def, input_name, output_name, batches):
    tf = backend.tf
    graph = tf.Graph()
    with graph.as_default():
        tf.import_graph_def(graph_def, name="")
    input_tensor = graph.get_tensor_by_name(input_name)
    output_tensor = graph.get_tensor_by_name(output_name)
    if input_tensor.shape.as_list() != [None, 256, 256, 3] or input_tensor.dtype != tf.float32:
        raise AssertionError("Unexpected PB input: {} {}".format(input_tensor.shape, input_tensor.dtype))
    if output_tensor.shape.as_list() != [None, 10] or output_tensor.dtype != tf.float32:
        raise AssertionError("Unexpected PB output: {} {}".format(output_tensor.shape, output_tensor.dtype))
    config = tf.ConfigProto(device_count={"GPU": 0}, allow_soft_placement=True)
    outputs = []
    # No initialization or learning-phase feed is needed by the imported PB.
    with tf.Session(graph=graph, config=config) as session:
        for batch in batches:
            logits = session.run(output_tensor, feed_dict={input_tensor: batch})
            if logits.dtype != np.float32 or logits.shape != (len(batch), 10) or not np.isfinite(logits).all():
                raise AssertionError("PB returned invalid logits")
            outputs.append(logits)
    return outputs


def compare(actual, expected, rtol, atol, label):
    difference = np.abs(actual - expected)
    report = {
        "max_abs_error": float(np.max(difference)),
        "mean_abs_error": float(np.mean(difference)),
        "rtol": rtol, "atol": atol,
        "allclose": bool(np.allclose(actual, expected, rtol=rtol, atol=atol)),
        "class_ids_equal": bool(np.array_equal(np.argmax(actual, axis=1), np.argmax(expected, axis=1))),
    }
    if not report["allclose"] or not report["class_ids_equal"]:
        raise AssertionError("{}: {}".format(label, report))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=PB_PATH)
    parser.add_argument("--report", default=REPORT_PATH)
    args = parser.parse_args()
    tf = backend.tf
    if tuple(str(tf.__version__).split(".")[:2]) != ("1", "15"):
        raise RuntimeError("Use TensorFlow 1.15; found {}".format(tf.__version__))
    pb_path = os.path.abspath(args.output)
    report_path = os.path.abspath(args.report)
    if os.path.normcase(os.path.realpath(pb_path)) == os.path.normcase(os.path.realpath(BASELINE_PB)):
        raise ValueError("Choose a new output path; the original PB is retained for comparison")

    baseline_hash = sha256(BASELINE_PB)
    baseline_graph = read_graph(BASELINE_PB)
    weights_hash = sha256(WEIGHTS)
    print("Building MobileViTv2-0.75 with inference fixed BEFORE construction...", flush=True)
    tf.keras.backend.clear_session()
    # predict() selects the false branch at runtime; freezing alone retains both.
    # A Python 0 before building BN/Dropout prevents training branches being built.
    tf.keras.backend.set_learning_phase(0)
    model = mobilevit.MobileViT_V2_075_TF115(WEIGHTS)
    weight_report = inspect_loaded_h5(model)
    if len(model.weights) != 244:
        raise AssertionError("Expected all 244 model weight tensors")
    input_name = model.inputs[0].name
    output_name = model.outputs[0].name
    bn_layers = [layer for layer in model.layers if isinstance(layer, backend.layers.BatchNormalization)]
    bn_state = [layer.get_weights()[2:4] for layer in bn_layers]
    if len(bn_layers) != 25 or any(abs(float(layer.epsilon) - 1e-5) > 1e-12 for layer in bn_layers):
        raise AssertionError("Expected 25 BatchNorm layers with epsilon=1e-5")

    with np.load(REFERENCE) as reference:
        inputs = reference["inputs_nhwc"].astype(np.float32)
        pytorch_logits = reference["logits"].astype(np.float32)
    if inputs.shape != (3, 256, 256, 3) or pytorch_logits.shape != (3, 10):
        raise AssertionError("Unexpected reference data shapes")
    batches = [inputs] + [inputs[index:index + 1] for index in range(3)]
    source_outputs = [model.predict(batch, batch_size=len(batch), verbose=0) for batch in batches]
    for layer, before in zip(bn_layers, bn_state):
        if any(not np.array_equal(old, new) for old, new in zip(before, layer.get_weights()[2:4])):
            raise AssertionError("Inference changed BatchNorm state: {}".format(layer.name))

    session = tf.keras.backend.get_session()
    frozen_graph = tf.graph_util.convert_variables_to_constants(
        session, session.graph.as_graph_def(), [output_name.split(":")[0]]
    )
    for node in frozen_graph.node:
        node.device = ""
    structure = inspect_graph(frozen_graph)
    if not structure["pure_inference"]:
        raise AssertionError("Frozen graph retains training structure: {}".format(structure))
    print("Structure passed: {} nodes, 25 inference BN, no training/control branches.".format(len(frozen_graph.node)), flush=True)

    os.makedirs(os.path.dirname(pb_path), exist_ok=True)
    temporary_path = pb_path + ".tmp"
    try:
        with tf.gfile.GFile(temporary_path, "wb") as output:
            output.write(frozen_graph.SerializeToString())
        loaded_graph = read_graph(temporary_path)
        frozen_outputs = run_graph(loaded_graph, input_name, output_name, batches + [inputs])
        baseline_outputs = run_graph(baseline_graph, input_name, output_name, batches)
        comparisons = []
        for index, batch in enumerate(batches):
            comparisons.append({
                "mode": "batch_3" if index == 0 else ["zero", "standard_normal", "normalized_rgb"][index - 1],
                "batch_size": len(batch),
                "vs_keras_model": compare(frozen_outputs[index], source_outputs[index], 1e-5, 1e-5, "PB/Keras"),
                "vs_original_pb": compare(frozen_outputs[index], baseline_outputs[index], 1e-5, 1e-5, "PB/original PB"),
                "vs_pytorch_reference": compare(
                    frozen_outputs[index], pytorch_logits if index == 0 else pytorch_logits[index - 1:index],
                    1e-4, 1e-4, "PB/PyTorch reference"
                ),
                "class_ids": np.argmax(frozen_outputs[index], axis=1).tolist(),
            })
        singles = np.concatenate(frozen_outputs[1:4], axis=0)
        consistency = compare(frozen_outputs[0], singles, 1e-4, 2e-4, "Batch/single inference")
        if not np.array_equal(frozen_outputs[0], frozen_outputs[4]):
            raise AssertionError("Repeated inference is not deterministic")
        if sha256(WEIGHTS) != weights_hash or sha256(BASELINE_PB) != baseline_hash:
            raise AssertionError("Source weights or original PB changed during export")
        os.replace(temporary_path, pb_path)
    finally:
        if os.path.isfile(temporary_path):
            os.remove(temporary_path)

    report = {
        "tensorflow_version": tf.__version__, "python_version": sys.version.split()[0],
        "model_source": os.path.abspath(mobilevit.__file__),
        "weights_path": WEIGHTS, "weights_sha256": weights_hash,
        "strict_h5_load": weight_report,
        "original_pb_path": BASELINE_PB, "original_pb_sha256": baseline_hash,
        "original_pb_structure": inspect_graph(baseline_graph),
        "pb_path": pb_path, "pb_sha256": sha256(pb_path), "pb_bytes": os.path.getsize(pb_path),
        "learning_phase_fixed_before_model_construction": 0,
        "input_tensor": input_name, "input_shape": model.input_shape,
        "input_dtype": "float32", "input_layout": "NHWC",
        "input_preprocessing": "RGB / 255, then (x - [0.485,0.456,0.406]) / [0.229,0.224,0.225]; outside PB",
        "output_tensor": output_name, "output_shape": model.output_shape,
        "output_dtype": "float32", "output_semantics": "10 raw logits; no softmax",
        "structure": structure, "comparisons": comparisons,
        "batch_vs_single": consistency,
        "batch_norm_moving_stats_unchanged": True,
        "repeat_inference_exactly_equal": True,
        "fresh_tf115_graph_session_without_initializers_or_learning_phase_feed": True,
        "reference_npz_sha256": sha256(REFERENCE),
        "full_validation_dataset_tested": False,
    }
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as output:
        json.dump(report, output, indent=2)
    print("Verified PB saved: {}".format(pb_path))
    print("Nodes: {}; bytes: {}; SHA256: {}".format(len(frozen_graph.node), report["pb_bytes"], report["pb_sha256"]))
    print("Max PB/Keras error: {}".format(max(item["vs_keras_model"]["max_abs_error"] for item in comparisons)))
    print("Max PB/original PB error: {}".format(max(item["vs_original_pb"]["max_abs_error"] for item in comparisons)))
    print("Report saved: {}".format(report_path))


if __name__ == "__main__":
    main()
