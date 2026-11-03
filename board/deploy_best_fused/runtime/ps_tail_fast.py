"""Faster NumPy tail: folds BN/BiasAdd into convolutions at load time.

Same graph JSON/NPZ as ps_tail_numpy.py, same input/output.  At load time it
removes the 11 FusedBatchNormV3, folds the 40 BiasAdd into the convolutions and
drops the identity nodes, so the per-frame loop only runs conv/attention ops.
Convolutions use a contiguous im2col + matmul (or direct matmul for 1x1);
depthwise convolutions accumulate over the kernel taps.

Usage (board or PC):
    python3 ps_tail_fast.py --graph ps_tail_numpy.json \
        --features-npy sample_features_fp32.npy \
        --reference-logits sample_reference_logits.npy
"""
from __future__ import print_function

import argparse
from collections import Counter
import json
from pathlib import Path
import time

import numpy as np


def _edge_name(edge):
    return edge.split(":")[0]


def _tensor_key(edge):
    return edge if ":" in edge else edge + ":0"


def _conv(x, weights, bias, attrs, depthwise=False):
    n, height, width, channels = x.shape
    kh, kw, weight_channels, last = weights.shape
    if weight_channels != channels:
        raise ValueError("Convolution input channel mismatch")
    strides = attrs["strides"]
    dilations = attrs["dilations"]
    if strides[0] != 1 or strides[3] != 1 or dilations[0] != 1 or dilations[3] != 1:
        raise ValueError("Unsupported batch/channel stride or dilation")
    sh, sw = strides[1], strides[2]
    dh, dw = dilations[1], dilations[2]
    effective_h, effective_w = (kh - 1) * dh + 1, (kw - 1) * dw + 1
    padding = attrs["padding"]
    if padding == "SAME":
        oh, ow = (height + sh - 1) // sh, (width + sw - 1) // sw
        total_h = max((oh - 1) * sh + effective_h - height, 0)
        total_w = max((ow - 1) * sw + effective_w - width, 0)
        pads = [[0, 0], [total_h // 2, total_h - total_h // 2],
                [total_w // 2, total_w - total_w // 2], [0, 0]]
    elif padding in ("VALID", "EXPLICIT"):
        flat = attrs.get("explicit_paddings", []) if padding == "EXPLICIT" else [0] * 8
        pads = np.asarray(flat).reshape(4, 2).tolist()
        oh = (height + sum(pads[1]) - effective_h) // sh + 1
        ow = (width + sum(pads[2]) - effective_w) // sw + 1
    else:
        raise ValueError("Unsupported padding: {}".format(padding))
    if any(a or b for a, b in pads):
        x = np.pad(x, pads, mode="constant")
    if oh <= 0 or ow <= 0:
        raise ValueError("Non-positive output shape")

    if depthwise:
        accumulator = np.zeros((n, oh, ow, channels, last), dtype=np.float32)
        for i in range(kh):
            for j in range(kw):
                window = x[:, i * dh:i * dh + (oh - 1) * sh + 1:sh,
                           j * dw:j * dw + (ow - 1) * sw + 1:sw, :]
                accumulator += window[..., np.newaxis] * weights[i, j]
        out = accumulator.reshape(n, oh, ow, channels * last)
    elif kh == 1 and kw == 1:
        sampled = np.ascontiguousarray(x[:, :oh * sh:sh, :ow * sw:sw, :])
        out = np.matmul(sampled.reshape(-1, channels), weights[0, 0]).reshape(n, oh, ow, last)
    else:
        windows = np.lib.stride_tricks.as_strided(
            x,
            shape=(n, oh, ow, kh, kw, channels),
            strides=(x.strides[0], x.strides[1] * sh, x.strides[2] * sw,
                     x.strides[1] * dh, x.strides[2] * dw, x.strides[3]),
            writeable=False)
        columns = np.ascontiguousarray(windows).reshape(n * oh * ow, kh * kw * channels)
        out = np.matmul(columns, weights.reshape(kh * kw * channels, last)).reshape(n, oh, ow, last)
    if bias is not None:
        out = out + bias
    return out


class FastTail(object):
    def __init__(self, graph_path, weights_path=None):
        graph_path = Path(graph_path)
        self.graph = json.loads(graph_path.read_text(encoding="utf-8"))
        if self.graph["format_version"] != 1:
            raise ValueError("Unsupported NumPy graph format")
        weights_path = Path(weights_path) if weights_path else graph_path.parent / self.graph["weights_file"]
        with np.load(str(weights_path), allow_pickle=False) as archive:
            self.weights = {name: archive[name] for name in archive.files}
        self.folded = {"bn": 0, "bias": 0, "identity": 0}
        self.nodes = self._optimize(list(self.graph["nodes"]))
        self.uses = Counter(_tensor_key(edge) for node in self.nodes for edge in node["inputs"])

    def _const_of(self, nodes, edge, by_name):
        node = by_name.get(_edge_name(edge))
        for _ in range(20):
            if node is None:
                break
            if node["op"] == "Const":
                return node
            if node["op"] != "Identity" or not node["inputs"]:
                break
            node = by_name.get(_edge_name(node["inputs"][0]))
        raise ValueError("Expected Const behind {}".format(edge))

    def _add_bias(self, node, bias):
        key = "bias_{:04d}".format(len(self.weights))
        self.weights[key] = np.asarray(bias, dtype=np.float32).reshape(-1)
        if node.get("bias_key"):
            self.weights[key] = (self.weights[node["bias_key"]] + self.weights[key]).astype(np.float32)
        node["bias_key"] = key

    def _optimize(self, nodes):
        by_name = {node["name"]: node for node in nodes}
        consumers = Counter(_edge_name(edge) for node in nodes for edge in node["inputs"])
        replace = {}
        conv_ops = ("Conv2D", "DepthwiseConv2dNative")

        def resolve(name):
            for _ in range(20):
                if name not in replace:
                    break
                name = replace[name]
            return name

        def producer(edge):
            node = by_name.get(_edge_name(edge))
            for _ in range(10):
                if node is None or node["op"] != "Identity":
                    break
                node = by_name.get(_edge_name(node["inputs"][0]))
            return node

        def fold(conv, gamma, beta, mean, variance, epsilon, extra_bias):
            scale = (gamma.astype(np.float32) / np.sqrt(
                variance.astype(np.float32) + np.float32(epsilon))).astype(np.float32)
            weight_node = self._const_of(nodes, conv["inputs"][1], by_name)
            weights = self.weights[weight_node["weight_key"]].astype(np.float32)
            if weights.shape[-1] == scale.size:
                weights = weights * scale.reshape((1,) * (weights.ndim - 1) + (-1,))
            else:
                weights = weights * scale.reshape((1, 1) + weights.shape[2:])
            key = "weight_{:04d}".format(len(self.weights))
            self.weights[key] = weights.astype(np.float32)
            conv["weight_key"] = key
            base = np.zeros(scale.size, dtype=np.float32) if extra_bias is None else extra_bias.astype(np.float32)
            conv["bias_key"] = None
            self._add_bias(conv, (base - mean.astype(np.float32)) * scale + beta.astype(np.float32))
            conv["inputs"] = [conv["inputs"][0]]

        # 1) BatchNorm (optionally through a BiasAdd) into the conv
        for node in nodes:
            if node["op"] != "FusedBatchNormV3":
                continue
            source = producer(node["inputs"][0])
            if source is None:
                continue
            conv = None
            extra_bias = None
            bias_add = None
            if source["op"] in conv_ops:
                conv = source
            elif source["op"] == "BiasAdd":
                inner = producer(source["inputs"][0])
                if inner is not None and inner["op"] in conv_ops and consumers[source["name"]] == 1:
                    conv = inner
                    bias_add = source
                    bias_node = self._const_of(nodes, source["inputs"][1], by_name)
                    extra_bias = self.weights[bias_node["weight_key"]]
            if conv is None or consumers[conv["name"]] != 1:
                continue
            gamma = self.weights[self._const_of(nodes, node["inputs"][1], by_name)["weight_key"]]
            beta = self.weights[self._const_of(nodes, node["inputs"][2], by_name)["weight_key"]]
            mean = self.weights[self._const_of(nodes, node["inputs"][3], by_name)["weight_key"]]
            variance = self.weights[self._const_of(nodes, node["inputs"][4], by_name)["weight_key"]]
            fold(conv, gamma, beta, mean, variance, node["attrs"]["epsilon"], extra_bias)
            replace[node["name"]] = conv["name"]
            if bias_add is not None:
                replace[bias_add["name"]] = conv["name"]
                self.folded["bias"] += 1
            self.folded["bn"] += 1

        # 2) remaining BiasAdd -> conv bias
        for node in nodes:
            if node["op"] != "BiasAdd" or node["name"] in replace:
                continue
            source = producer(node["inputs"][0])
            if source is None or source["op"] not in conv_ops:
                continue
            if consumers[source["name"]] != 1:
                continue
            bias_node = self._const_of(nodes, node["inputs"][1], by_name)
            self._add_bias(source, self.weights[bias_node["weight_key"]])
            replace[node["name"]] = source["name"]
            self.folded["bias"] += 1

        # 3) Identity / StopGradient
        for node in nodes:
            if node["op"] in ("Identity", "StopGradient") and node["inputs"]:
                replace[node["name"]] = _edge_name(node["inputs"][0])
                self.folded["identity"] += 1

        kept = [node for node in nodes if node["name"] not in replace]
        for node in kept:
            inputs = []
            for edge in node["inputs"]:
                name = resolve(_edge_name(edge))
                inputs.append(name + (":" + edge.split(":", 1)[1] if ":" in edge else ""))
            node["inputs"] = inputs
        # Intermediate split outputs may be Identity or a folded BN/BiasAdd.
        # Preserve the exported output port while resolving its eliminated node.
        output_name, output_port = self.graph["output_tensor"].rsplit(":", 1)
        self.graph["output_tensor"] = resolve(output_name) + ":" + output_port
        return kept
    def predict(self, features):
        features = np.asarray(features)
        if features.dtype.kind != "f":
            raise ValueError("Dequantized floating-point features are required")
        expected = tuple(self.graph["input_shape"])
        if features.shape == expected[1:]:
            features = features[np.newaxis, ...]
        if features.shape != expected:
            raise ValueError("Expected NHWC {}; got {}".format(expected, features.shape))
        features = np.ascontiguousarray(features, dtype=np.float32)
        values = {}
        uses = self.uses.copy()
        for node in self.nodes:
            op = node["op"]
            name = node["name"]
            if op == "Const":
                outputs = [self.weights[node["weight_key"]]]
            elif op == "Placeholder":
                outputs = [features]
            else:
                inputs = [values[_tensor_key(edge)] for edge in node["inputs"]]
                if op == "Conv2D" or op == "DepthwiseConv2dNative":
                    weights = self.weights[node["weight_key"]] if node.get("weight_key") else inputs[1]
                    bias = self.weights[node["bias_key"]] if node.get("bias_key") else None
                    outputs = [_conv(inputs[0], weights, bias, node["attrs"],
                                     depthwise=op == "DepthwiseConv2dNative")]
                else:
                    outputs = self._evaluate(op, inputs, node["attrs"])
            for port, value in enumerate(outputs):
                key = name + ":" + str(port)
                if uses[key] > 0 or key == self.graph["output_tensor"]:
                    values[key] = value
            for edge in node["inputs"]:
                key = _tensor_key(edge)
                uses[key] -= 1
                if uses[key] == 0 and key != self.graph["output_tensor"]:
                    del values[key]
        logits = np.asarray(values[self.graph["output_tensor"]], dtype=np.float32)
        if logits.shape != tuple(self.graph["output_shape"]):
            raise ValueError("Invalid output logits shape {}".format(logits.shape))
        return logits

    @staticmethod
    def _evaluate(op, inputs, attrs):
        x = inputs[0]
        if op == "BiasAdd":
            return [x + inputs[1]]
        if op == "FusedBatchNormV3":
            scale, offset, mean, variance = inputs[1:5]
            inverse = scale / np.sqrt(variance + np.float32(attrs["epsilon"]))
            return [(x - mean) * inverse + offset]
        if op in ("Add", "AddV2"):
            return [x + inputs[1]]
        if op == "Sub":
            return [x - inputs[1]]
        if op == "Mul":
            return [x * inputs[1]]
        if op == "SquaredDifference":
            difference = x - inputs[1]
            return [difference * difference]
        if op == "Rsqrt":
            return [np.float32(1.0) / np.sqrt(x)]
        if op == "Relu":
            return [np.maximum(x, np.float32(0.0))]
        if op in ("Mean", "Sum"):
            axes = tuple(np.asarray(inputs[1]).reshape(-1).tolist())
            reducer = np.mean if op == "Mean" else np.sum
            return [reducer(x, axis=axes, keepdims=attrs["keep_dims"])]
        if op == "Reshape":
            return [np.reshape(x, tuple(np.asarray(inputs[1]).reshape(-1).tolist()))]
        if op == "Transpose":
            return [np.transpose(x, axes=np.asarray(inputs[1]).tolist())]
        if op == "Pad":
            return [np.pad(x, np.asarray(inputs[1]).tolist(), mode="constant")]
        if op == "Softmax":
            exponential = np.exp(x - np.max(x, axis=-1, keepdims=True))
            return [exponential / np.sum(exponential, axis=-1, keepdims=True)]
        if op == "SplitV":
            sizes = np.asarray(inputs[1]).astype(np.int64).reshape(-1)
            axis = int(np.asarray(inputs[2]).item())
            if np.count_nonzero(sizes == -1) == 1:
                sizes = sizes.copy()
                sizes[sizes == -1] = x.shape[axis] - int(sizes[sizes >= 0].sum())
            return np.split(x, np.cumsum(sizes)[:-1], axis=axis)
        if op == "MatMul":
            return [np.matmul(x, inputs[1])]
        raise ValueError("Unsupported op: {}".format(op))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, default=Path("ps_tail_numpy.json"))
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--features-npy", type=Path, required=True)
    parser.add_argument("--reference-logits", type=Path)
    parser.add_argument("--repeat", type=int, default=5)
    args = parser.parse_args()

    started = time.time()
    tail = FastTail(args.graph, args.weights)
    load_seconds = time.time() - started
    features = np.load(str(args.features_npy), allow_pickle=False)
    logits = tail.predict(features)[0]
    started = time.time()
    for _ in range(args.repeat):
        logits = tail.predict(features)[0]
    elapsed = (time.time() - started) / args.repeat
    result = {"class_index": int(np.argmax(logits)), "logits": logits.tolist(),
              "numpy_version": np.__version__, "folded": tail.folded,
              "nodes": len(tail.nodes), "load_s": round(load_seconds, 3),
              "ms_per_frame": round(elapsed * 1000, 2)}
    if args.reference_logits:
        reference = np.load(str(args.reference_logits), allow_pickle=False).reshape(-1)
        result["max_absolute_logit_error"] = float(np.max(np.abs(logits - reference)))
        result["self_check_passed"] = bool(np.allclose(logits, reference, rtol=1e-4, atol=1e-4)
                                           and np.argmax(logits) == np.argmax(reference))
    print(json.dumps(result, indent=2))
    if args.reference_logits and not result["self_check_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
