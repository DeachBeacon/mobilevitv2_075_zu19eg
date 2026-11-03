"""Inspect a MobileViTv2 GraphDef for retained training branches (TF 1.15)."""
import argparse
import collections
import hashlib
import json
import os
from _paths import PROJECT, TF_ROOT, TF_RECORDS, TF_WEIGHTS, INFERENCE, DEPLOY, REFERENCE_PB

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import tensorflow as tf


CONTROL_FLOW_OPS = {
    "Switch", "RefSwitch", "Merge", "RefMerge", "If", "StatelessIf",
    "While", "StatelessWhile", "Enter", "RefEnter", "Exit", "RefExit",
    "NextIteration", "RefNextIteration", "LoopCond",
}


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def inspect_graph(graph_def):
    # Also inspect function bodies so function calls cannot hide training ops.
    scoped_nodes = [(node.name, node) for node in graph_def.node]
    for function in graph_def.library.function:
        scoped_nodes.extend((function.signature.name + "/" + node.name, node) for node in function.node_def)
    counts = collections.Counter(node.op for _, node in scoped_nodes)
    batch_norms = [(name, node) for name, node in scoped_nodes if node.op.startswith("FusedBatchNorm")]
    training_batch_norms = [name for name, node in batch_norms if node.attr["is_training"].b]
    learning_phase_nodes = [name for name, _ in scoped_nodes if "learning_phase" in name.lower()]
    variable_ops = {
        op: count for op, count in counts.items()
        if "Variable" in op or "Assign" in op or op in {"VarHandleOp", "VarIsInitializedOp"}
        or op.startswith(("ResourceApply", "Apply", "ResourceScatter"))
    }
    random_ops = {op: count for op, count in counts.items() if "Random" in op or "Multinomial" in op}
    control_flow = {op: count for op, count in counts.items() if op in CONTROL_FLOW_OPS}
    # tf.nn.moments uses StopGradient in its forward variance calculation;
    # it is an identity at inference and is not a gradient/training branch.
    gradient_nodes = [name for name, node in scoped_nodes
                      if "gradients/" in name.lower()
                      or ("Grad" in node.op and node.op != "StopGradient")
                      or "optimizer/" in name.lower()]
    placeholders = [{"name": node.name, "op": node.op}
                    for node in graph_def.node if node.op.startswith("Placeholder")]
    # A deliberately strict structural contract for this fixed model.
    pure_inference = (
        placeholders == [{"name": "input_1", "op": "Placeholder"}]
        and len(batch_norms) == 25
        and not any((training_batch_norms, learning_phase_nodes, variable_ops,
                     random_ops, control_flow, gradient_nodes))
        and not graph_def.library.gradient
    )
    return {
        "node_count": len(graph_def.node),
        "function_count": len(graph_def.library.function),
        "op_counts": dict(sorted(counts.items())),
        "placeholders": placeholders,
        "control_flow_ops": control_flow,
        "batch_norm_count": len(batch_norms),
        "training_batch_norm_count": len(training_batch_norms),
        "training_batch_norm_nodes": training_batch_norms,
        "inference_batch_norm_count": len(batch_norms) - len(training_batch_norms),
        "learning_phase_nodes": learning_phase_nodes,
        "variable_or_update_ops": variable_ops,
        "random_ops": random_ops,
        "gradient_or_optimizer_nodes": gradient_nodes,
        "registered_gradient_count": len(graph_def.library.gradient),
        "pure_inference": bool(pure_inference),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pb", help="Frozen GraphDef to inspect")
    parser.add_argument("--report", help="Optional JSON audit output")
    parser.add_argument("--require-pure-inference", action="store_true")
    args = parser.parse_args()
    graph_def = tf.GraphDef()
    with tf.gfile.GFile(args.pb, "rb") as source:
        graph_def.ParseFromString(source.read())
    report = inspect_graph(graph_def)
    report.update({"pb_path": os.path.abspath(args.pb), "pb_sha256": sha256(args.pb),
                   "pb_bytes": os.path.getsize(args.pb), "tensorflow_version": tf.__version__})
    if args.report:
        with open(args.report, "w", encoding="utf-8") as output:
            json.dump(report, output, indent=2)
    print(json.dumps(report, indent=2))
    if args.require_pure_inference and not report["pure_inference"]:
        raise AssertionError("GraphDef fails this MobileViTv2 model's pure inference contract")


if __name__ == "__main__":
    main()
