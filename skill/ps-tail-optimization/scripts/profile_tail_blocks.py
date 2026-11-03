"""Board-side profiler: per-block wall time of the NumPy tail, plus a BLAS check.

Run this on the board before changing the split.  It answers two questions:

  1. How is the per-image time distributed over the tail blocks?
  2. Does this NumPy have a real BLAS?  (np.matmul GFLOP/s is the quick test)

Usage:
    python3 profile_tail_blocks.py --graph ps_tail_numpy.json \
        --features-npy sample_features_fp32.npy --repeat 3
"""
from __future__ import print_function

import argparse
import re
import sys
import time
from collections import Counter, defaultdict

import numpy as np

from ps_tail_numpy import NumpyPSTail, evaluate_node, tensor_key

BLOCK = re.compile(r"(stack\d+_block\d+)")
CONV = {"Conv2D", "DepthwiseConv2dNative"}
ATTENTION = {"Softmax", "SplitV", "Sum", "MatMul", "Mean"}


def block_of(name):
    top = name.split("/")[0]
    if top.startswith("avg_pool") or top.startswith("predictions"):
        return "HEAD"
    match = BLOCK.search(top)
    return match.group(1) if match else "io/reshape"


def macs(op, inputs, out):
    if op == "Conv2D":
        weights = inputs[1]
        n, oh, ow, cout = out.shape
        return int(n * oh * ow * cout * weights.shape[0] * weights.shape[1] * weights.shape[2])
    if op == "DepthwiseConv2dNative":
        weights = inputs[1]
        n, oh, ow, channels = out.shape
        return int(n * oh * ow * channels * weights.shape[0] * weights.shape[1])
    if op == "MatMul":
        left, right = inputs[0], inputs[1]
        return int(np.prod(left.shape[:-1]) * left.shape[-1] * right.shape[-1])
    return 0


def run_once(tail, features, record):
    nodes = tail.nodes
    uses = Counter(tensor_key(edge) for node in nodes for edge in node["inputs"])
    values = {}
    blocks = defaultdict(lambda: {"seconds": 0.0, "conv_seconds": 0.0, "macs": 0,
                                  "ops": Counter()})
    ops = defaultdict(lambda: {"seconds": 0.0, "count": 0})
    for node in nodes:
        op, name = node["op"], node["name"]
        if op == "Const":
            outputs = [tail.weights[node["weight_key"]]]
            elapsed = 0.0
            inputs = []
        elif op == "Placeholder":
            outputs = [features]
            elapsed = 0.0
            inputs = []
        else:
            inputs = [values[tensor_key(edge)] for edge in node["inputs"]]
            started = time.time()
            outputs = evaluate_node(op, inputs, node["attrs"])
            elapsed = time.time() - started
        if record and op not in ("Const", "Placeholder"):
            key = block_of(name)
            blocks[key]["seconds"] += elapsed
            if op in CONV:
                blocks[key]["conv_seconds"] += elapsed
            blocks[key]["macs"] += macs(op, inputs, outputs[0])
            blocks[key]["ops"][op] += 1
            ops[op]["seconds"] += elapsed
            ops[op]["count"] += 1
        for port, value in enumerate(outputs):
            key = name + ":" + str(port)
            if uses[key] > 0 or key == tail.graph["output_tensor"]:
                values[key] = value
        for edge in node["inputs"]:
            key = tensor_key(edge)
            uses[key] -= 1
            if uses[key] == 0 and key != tail.graph["output_tensor"]:
                del values[key]
    return blocks, ops


def bench_matmul(size, repeat=5):
    left = np.random.rand(size, size).astype(np.float32)
    right = np.random.rand(size, size).astype(np.float32)
    left.dot(right)
    started = time.time()
    for _ in range(repeat):
        left.dot(right)
    elapsed = time.time() - started
    gflops = 2.0 * size ** 3 * repeat / elapsed / 1e9
    print("matmul {0}x{0}: {1:.2f} GFLOP/s ({2:.4f} s per call)".format(
        size, gflops, elapsed / repeat))
    if gflops < 1.0:
        print("  -> below ~1 GFLOP/s: NumPy is likely running without a usable BLAS.")
    return gflops


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", default="ps_tail_numpy.json")
    parser.add_argument("--weights", default=None)
    parser.add_argument("--features-npy", required=True)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--bench-size", type=int, default=512)
    args = parser.parse_args()

    print("python", sys.version.split()[0], "numpy", np.__version__)
    try:
        np.__config__.show()
    except Exception as error:
        print("np.__config__.show() failed:", error)
    bench_matmul(args.bench_size)

    features = np.load(args.features_npy, allow_pickle=False)
    tail = NumpyPSTail(args.graph, args.weights)
    run_once(tail, features, False)  # warm-up

    totals = defaultdict(lambda: {"seconds": 0.0, "conv_seconds": 0.0, "macs": 0,
                                  "ops": Counter()})
    total_ops = defaultdict(lambda: {"seconds": 0.0, "count": 0})
    for _ in range(args.repeat):
        blocks, ops = run_once(tail, features, True)
        for key, value in blocks.items():
            totals[key]["seconds"] += value["seconds"]
            totals[key]["conv_seconds"] += value["conv_seconds"]
            totals[key]["macs"] += value["macs"]
            totals[key]["ops"] += value["ops"]
        for key, value in ops.items():
            total_ops[key]["seconds"] += value["seconds"]
            total_ops[key]["count"] += value["count"]

    n = args.repeat
    total_seconds = sum(v["seconds"] for v in totals.values()) / n
    print("\ntail total: {0:.1f} ms/frame, {1:.3f} GMAC".format(
        total_seconds * 1000, sum(v["macs"] for v in totals.values()) / n / 1e9))
    print("\n{0:<20}{1:>10}{2:>10}{3:>10}{4:>8}{5:>8}".format(
        "block", "ms", "conv_ms", "MACs(M)", "share", "attn"))
    for key, value in sorted(totals.items(), key=lambda kv: -kv[1]["seconds"]):
        has_attention = any(op in ATTENTION for op in value["ops"])
        print("{0:<20}{1:>10.2f}{2:>10.2f}{3:>10.1f}{4:>7.1%}{5:>8}".format(
            key, value["seconds"] / n * 1000, value["conv_seconds"] / n * 1000,
            value["macs"] / n / 1e6, value["seconds"] / n / total_seconds,
            "yes" if has_attention else "no"))
    print("\n{0:<24}{1:>8}{2:>10}".format("op", "count", "ms"))
    for key, value in sorted(total_ops.items(), key=lambda kv: -kv[1]["seconds"]):
        print("{0:<24}{1:>8}{2:>10.2f}".format(key, value["count"], value["seconds"] / n * 1000))


if __name__ == "__main__":
    main()
