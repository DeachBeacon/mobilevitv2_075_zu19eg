"""A/B benchmark: ps_tail_numpy.py (baseline) vs ps_tail_fast.py on the same input.

Checks numerical agreement and measures per-frame time of both interpreters, plus
a 512x512 matmul GFLOP/s so the BLAS situation is visible in the same output.

Usage:
    python3 bench_tail.py --graph ps_tail_numpy.json \
        --features-npy sample_features_fp32.npy --repeat 5
"""
from __future__ import print_function
import sys
from pathlib import Path
_PROJECT=Path(__file__).resolve().parents[1]
for _path in (_PROJECT/'src/dpu',_PROJECT/'src/dpu/split',_PROJECT/'board/deploy_best_fused/runtime'):
    sys.path.insert(0,str(_path))


import argparse
import json
import time

import numpy as np

from ps_tail_numpy import NumpyPSTail
from ps_tail_fast import FastTail
from ps_tail_fused import FusedTail


def bench_matmul(size=512, repeat=5):
    left = np.random.rand(size, size).astype(np.float32)
    right = np.random.rand(size, size).astype(np.float32)
    left.dot(right)
    started = time.time()
    for _ in range(repeat):
        left.dot(right)
    elapsed = time.time() - started
    return 2.0 * size ** 3 * repeat / elapsed / 1e9


def bench(model, features, repeat):
    model.predict(features)  # warm-up
    started = time.time()
    for _ in range(repeat):
        model.predict(features)
    elapsed = time.time() - started
    return elapsed / repeat


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", default="ps_tail_numpy.json")
    parser.add_argument("--weights", default=None)
    parser.add_argument("--features-npy", required=True)
    parser.add_argument("--repeat", type=int, default=5)
    args = parser.parse_args()

    features = np.load(args.features_npy, allow_pickle=False)
    started = time.time()
    baseline = NumpyPSTail(args.graph, args.weights)
    load_baseline = time.time() - started
    started = time.time()
    fast = FastTail(args.graph, args.weights)
    load_fast = time.time() - started
    started = time.time()
    fused = FusedTail(args.graph, args.weights)
    load_fused = time.time() - started

    reference = baseline.predict(features)
    candidate = fast.predict(features)
    diff = float(np.max(np.abs(reference - candidate)))
    same_class = int(np.argmax(reference)) == int(np.argmax(candidate))
    fused_out = fused.predict(features)
    fused_diff = float(np.max(np.abs(reference - fused_out)))
    fused_same = int(np.argmax(reference)) == int(np.argmax(fused_out))

    baseline_ms = bench(baseline, features, args.repeat) * 1000
    fast_ms = bench(fast, features, args.repeat) * 1000
    fused_ms = bench(fused, features, args.repeat) * 1000

    result = {
        "numpy_version": np.__version__,
        "matmul_gflops": round(bench_matmul(), 2),
        "repeat": args.repeat,
        "load_s": {"baseline": round(load_baseline, 3), "fast": round(load_fast, 3),
                   "fused": round(load_fused, 3)},
        "ms_per_frame": {"baseline": round(baseline_ms, 2), "fast": round(fast_ms, 2),
                         "fused": round(fused_ms, 2)},
        "speedup": round(baseline_ms / fast_ms, 2),
        "fused_speedup": round(baseline_ms / fused_ms, 2),
        "nodes": {"baseline": len(baseline.nodes), "fast": len(fast.nodes),
                  "fused": len(fused.nodes)},
        "folded": fast.folded,
        "fused_ops": fused.fused_ops,
        "max_abs_logit_diff": diff,
        "same_class": same_class,
        "fused_max_abs_logit_diff": fused_diff,
        "fused_same_class": fused_same,
        "self_check_passed": bool(diff < 1e-4 and same_class and fused_diff < 1e-4 and fused_same),
    }
    print(json.dumps(result, indent=2))
    if not result["self_check_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
