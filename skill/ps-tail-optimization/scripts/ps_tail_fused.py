"""Fused NumPy tail: FastTail + GroupNorm / separable-attention fusion.

Adds two extra load-time rewrites on top of ps_tail_fast:

  * FusedGroupNorm: Reshape->Mean->Sub->SquaredDifference->Mean->AddV2->Rsqrt->Mul->Reshape
    becomes one pass (reshape, mean/var over the given axes, normalize, reshape back).
  * FusedAttention: SplitV->Transpose->Softmax->Transpose->Mul->Sum->Relu->Mul
    becomes one pass over the qkv tensor (score / value / gate).

Both are numerically equivalent to the original graph (verified on the sample
features: group norm 9.5e-7, attention 0.0).  Use `--fuse` in the runners, or
instantiate FusedTail directly.
"""
from __future__ import print_function

from collections import Counter

import numpy as np

from ps_tail_fast import FastTail


def _resolve(replace, name):
    for _ in range(20):
        if name not in replace:
            break
        name = replace[name]
    return name


class FusedTail(FastTail):
    def __init__(self, graph_path, weights_path=None, fuse_group_norm=True, fuse_attention=True):
        FastTail.__init__(self, graph_path, weights_path)
        self.fused_ops = {"group_norm": 0, "attention": 0}
        self._fuse(fuse_group_norm, fuse_attention)

    # ------------------------------------------------------------------ fusion
    def _fuse(self, do_group_norm, do_attention):
        nodes = self.nodes
        by_name = {node["name"]: node for node in nodes}
        replace = {}

        def producer(edge):
            return by_name.get(_resolve(replace, edge.split(":")[0]))

        def const_name(edge):
            node = producer(edge)
            if node is None or node["op"] != "Const":
                raise ValueError("Expected Const behind {}".format(edge))
            return node["name"] + ":0"

        if do_group_norm:
            for node in list(nodes):
                if node["op"] != "Reshape" or not node["name"].endswith("Reshape_1"):
                    continue
                mul = producer(node["inputs"][0])
                if mul is None or mul["op"] != "Mul":
                    continue
                sub = producer(mul["inputs"][0])
                rsqrt = producer(mul["inputs"][1])
                if sub is None or rsqrt is None or sub["op"] != "Sub" or rsqrt["op"] != "Rsqrt":
                    continue
                add = producer(rsqrt["inputs"][0])
                if add is None or add["op"] != "AddV2":
                    continue
                variance = producer(add["inputs"][0])
                if variance is None or variance["op"] != "Mean":
                    continue
                squared = producer(variance["inputs"][0])
                reshape = producer(sub["inputs"][0])
                if squared is None or reshape is None or squared["op"] != "SquaredDifference" or reshape["op"] != "Reshape":
                    continue
                mean = producer(sub["inputs"][1])
                if mean is None or mean["op"] != "Mean" or mean["name"] != producer(squared["inputs"][1])["name"]:
                    continue
                try:
                    shape_edge = const_name(reshape["inputs"][1])
                    axes_edge = const_name(variance["inputs"][1])
                    eps_edge = const_name(add["inputs"][1])
                    out_edge = const_name(node["inputs"][1])
                except ValueError:
                    continue
                node["op"] = "FusedGroupNorm"
                node["inputs"] = [reshape["inputs"][0], shape_edge, axes_edge, eps_edge, out_edge]
                for victim in (mul, sub, rsqrt, add, variance, squared, mean, reshape):
                    replace[victim["name"]] = node["name"]
                self.fused_ops["group_norm"] += 1

        if do_attention:
            for node in list(nodes):
                if node["op"] != "Mul":
                    continue
                left = producer(node["inputs"][0])
                right = producer(node["inputs"][1])
                if left is None or right is None:
                    continue
                ops = {left["op"], right["op"]}
                if ops != {"Relu", "Sum"}:
                    continue
                gate = left if left["op"] == "Relu" else right
                total = right if gate is left else left
                weighted = producer(total["inputs"][0])
                if weighted is None or weighted["op"] != "Mul":
                    continue
                split = producer(gate["inputs"][0])
                if split is None or split["op"] != "SplitV":
                    continue
                # value branch and the softmax branch must come from the same SplitV
                value_edges = [e for e in weighted["inputs"] if e.split(":")[0] == split["name"]]
                other_edges = [e for e in weighted["inputs"] if e.split(":")[0] != split["name"]]
                if len(value_edges) != 1 or len(other_edges) != 1:
                    continue
                transpose_1 = producer(other_edges[0])
                if transpose_1 is None or transpose_1["op"] != "Transpose":
                    continue
                softmax = producer(transpose_1["inputs"][0])
                if softmax is None or softmax["op"] != "Softmax":
                    continue
                transpose = producer(softmax["inputs"][0])
                if transpose is None or transpose["op"] != "Transpose":
                    continue
                if transpose["inputs"][0].split(":")[0] != split["name"]:
                    continue
                node["op"] = "FusedAttention"
                node["inputs"] = [split["inputs"][0], split["inputs"][1], split["inputs"][2]]
                for victim in (split, gate, total, weighted, transpose_1, softmax, transpose):
                    replace[victim["name"]] = node["name"]
                self.fused_ops["attention"] += 1

        if not replace:
            return
        kept = [node for node in nodes if node["name"] not in replace]
        for node in kept:
            inputs = []
            for edge in node["inputs"]:
                name = _resolve(replace, edge.split(":")[0])
                inputs.append(name + (":" + edge.split(":", 1)[1] if ":" in edge else ""))
            node["inputs"] = inputs
        self.nodes = kept
        self.uses = Counter(self._key(edge) for node in kept for edge in node["inputs"])

    @staticmethod
    def _key(edge):
        return edge if ":" in edge else edge + ":0"

    # --------------------------------------------------------------- execution
    @staticmethod
    def _evaluate(op, inputs, attrs):
        if op == "FusedGroupNorm":
            shape = tuple(int(value) for value in np.asarray(inputs[1]).reshape(-1).tolist())
            axes = tuple(int(value) for value in np.asarray(inputs[2]).reshape(-1).tolist())
            epsilon = float(np.asarray(inputs[3]).reshape(-1)[0])
            out_shape = tuple(int(value) for value in np.asarray(inputs[4]).reshape(-1).tolist())
            x = inputs[0].reshape(shape)
            mean = x.mean(axis=axes, keepdims=True)
            variance = ((x - mean) ** 2).mean(axis=axes, keepdims=True)
            return [((x - mean) / np.sqrt(variance + np.float32(epsilon))).reshape(out_shape)]
        if op == "FusedAttention":
            qkv = inputs[0]
            sizes = np.asarray(inputs[1]).reshape(-1).astype(np.int64)
            score = qkv[..., :sizes[0]]
            value = qkv[..., sizes[0]:sizes[0] + sizes[1]]
            gate = np.maximum(qkv[..., sizes[0] + sizes[1]:], np.float32(0.0))
            scores = score - score.max(axis=2, keepdims=True)
            weights = np.exp(scores)
            weights /= weights.sum(axis=2, keepdims=True)
            return [gate * (weights * value).sum(axis=2, keepdims=True)]
        return FastTail._evaluate(op, inputs, attrs)


def main():
    import argparse
    import json
    from pathlib import Path
    import time

    parser = argparse.ArgumentParser(description="Fused tail self-check")
    parser.add_argument("--graph", type=Path, default=Path("ps_tail_numpy.json"))
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--features-npy", type=Path, required=True)
    parser.add_argument("--reference-logits", type=Path)
    parser.add_argument("--repeat", type=int, default=5)
    args = parser.parse_args()

    started = time.time()
    tail = FusedTail(args.graph, args.weights)
    load_seconds = time.time() - started
    features = np.load(str(args.features_npy), allow_pickle=False)
    logits = tail.predict(features)[0]
    started = time.time()
    for _ in range(args.repeat):
        logits = tail.predict(features)[0]
    elapsed = (time.time() - started) / args.repeat
    result = {"class_index": int(np.argmax(logits)), "numpy_version": np.__version__,
              "folded": tail.folded, "fused_ops": tail.fused_ops,
              "nodes": len(tail.nodes), "load_s": round(load_seconds, 3),
              "ms_per_frame": round(elapsed * 1000, 2)}
    if args.reference_logits:
        reference = np.load(str(args.reference_logits), allow_pickle=False).reshape(-1)
        result["max_absolute_logit_error"] = float(np.max(np.abs(logits - reference)))
        result["self_check_passed"] = bool(np.allclose(logits, reference, rtol=1e-4, atol=1e-4))
    print(json.dumps(result, indent=2))
    if args.reference_logits and not result["self_check_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
