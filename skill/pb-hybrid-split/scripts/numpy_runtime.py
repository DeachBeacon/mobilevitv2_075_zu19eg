"""Single-input/output FP32 NumPy segment, no TensorFlow dependency."""
from collections import Counter
import json
from pathlib import Path
import numpy as np

def tensor_key(edge):
    return edge if ":" in edge else edge + ":0"


def _convolution(x, weights, attrs, depthwise=False):
    n, height, width, channels = x.shape
    kh, kw, weight_channels, last = weights.shape
    if weight_channels != channels:
        raise ValueError("Convolution input channel mismatch")
    strides = attrs["strides"]
    dilations = attrs["dilations"]
    if strides[0] != 1 or strides[3] != 1 or dilations[0] != 1 or dilations[3] != 1:
        raise ValueError("Unsupported batch/channel stride or dilation")
    sh, sw = strides[1:3]
    dh, dw = dilations[1:3]
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
        if len(flat) != 8:
            raise ValueError("Invalid explicit padding")
        pads = np.asarray(flat).reshape(4, 2).tolist()
        if any(pads[0]) or any(pads[3]):
            raise ValueError("Unsupported batch/channel padding")
        oh = (height + sum(pads[1]) - effective_h) // sh + 1
        ow = (width + sum(pads[2]) - effective_w) // sw + 1
    else:
        raise ValueError("Unsupported padding: {}".format(padding))
    if oh <= 0 or ow <= 0:
        raise ValueError("Non-positive convolution output shape")
    if any(a or b for a, b in pads):
        x = np.pad(x, pads, mode="constant")
    if kh == 1 and kw == 1:
        sampled = x[:, :oh * sh:sh, :ow * sw:sw, :]
        if depthwise:
            return (sampled[..., np.newaxis] * weights[0, 0]).reshape(n, oh, ow, channels * last)
        return np.matmul(sampled, weights[0, 0])
    # as_strided is available in NumPy 1.17; sliding_window_view is not.
    window_shape = (n, oh, ow, kh, kw, channels)
    strides_bytes = (x.strides[0], x.strides[1] * sh, x.strides[2] * sw,
                     x.strides[1] * dh, x.strides[2] * dw, x.strides[3])
    windows = np.lib.stride_tricks.as_strided(x, shape=window_shape, strides=strides_bytes,
                                             writeable=False)
    if depthwise:
        result = np.einsum("nhwklc,klcm->nhwcm", windows, weights, optimize=True)
        return result.reshape(n, oh, ow, channels * last)
    return np.tensordot(windows, weights, axes=((3, 4, 5), (0, 1, 2)))


def evaluate_node(op, inputs, attrs):
    x = inputs[0]
    if op in ("Identity", "StopGradient"):
        return [x]
    if op in ("Conv2D", "DepthwiseConv2dNative"):
        return [_convolution(x, inputs[1], attrs, depthwise=op == "DepthwiseConv2dNative")]
    if op == "FusedBatchNormV3":
        scale, offset, mean, variance = inputs[1:5]
        inverse = scale / np.sqrt(variance + np.float32(attrs["epsilon"]))
        return [(x - mean) * inverse + offset]
    if op in ("BiasAdd", "Add", "AddV2"):
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
        if np.any(sizes < 0) or int(sizes.sum()) != x.shape[axis] or len(sizes) != attrs["num_split"]:
            raise ValueError("Invalid SplitV sizes")
        return np.split(x, np.cumsum(sizes)[:-1], axis=axis)
    if op == "MatMul":
        a = x.swapaxes(-1, -2) if attrs["transpose_a"] else x
        b = inputs[1].swapaxes(-1, -2) if attrs["transpose_b"] else inputs[1]
        return [np.matmul(a, b)]
    raise ValueError("Unsupported op: {}".format(op))


class NumpySegment(object):
    """Load the numeric weights once; predict each frame's dequantized features."""

    def __init__(self, graph_path, weights_path=None):
        graph_path = Path(graph_path)
        self.graph = json.loads(graph_path.read_text(encoding="utf-8"))
        if self.graph["format_version"] != 1:
            raise ValueError("Unsupported NumPy graph format")
        self.nodes = self.graph["nodes"]
        weights_path = Path(weights_path) if weights_path else graph_path.parent / self.graph["weights_file"]
        with np.load(str(weights_path), allow_pickle=False) as archive:
            self.weights = {name: archive[name] for name in archive.files}
        self.uses = Counter(tensor_key(edge) for node in self.nodes for edge in node["inputs"])

    def predict(self, features):
        features = np.asarray(features)
        if features.dtype.kind != "f":
            raise ValueError("Dequantized floating-point features are required, not DPU INT8 bytes")
        expected = tuple(self.graph["input_shape"])
        if features.shape == expected[1:]:
            features = features[np.newaxis, ...]
        if features.shape != expected:
            raise ValueError("Expected shape {}; got {}".format(expected, features.shape))
        features = np.asarray(features, dtype=np.float32)
        if not np.isfinite(features).all():
            raise ValueError("Non-finite input features")
        values = {}
        uses = self.uses.copy()
        for node in self.nodes:
            op = node["op"]
            name = node["name"]
            if op == "Const":
                outputs = [self.weights[node["weight_key"]]]
            elif op == "Placeholder":
                if name + ":0" != self.graph["input_tensor"]:
                    raise ValueError("Unexpected Placeholder: {}".format(name))
                outputs = [features]
            else:
                inputs = [values[tensor_key(edge)] for edge in node["inputs"]]
                try:
                    outputs = evaluate_node(op, inputs, node["attrs"])
                except Exception as error:
                    raise RuntimeError("{} [{}]: {}".format(name, op, error))
            for port, value in enumerate(outputs):
                key = name + ":" + str(port)
                if uses[key] > 0 or key == self.graph["output_tensor"]:
                    values[key] = value
            for edge in node["inputs"]:
                key = tensor_key(edge)
                uses[key] -= 1
                if uses[key] == 0 and key != self.graph["output_tensor"]:
                    del values[key]
        logits = np.asarray(values[self.graph["output_tensor"]], dtype=np.float32)
        if logits.shape != tuple(self.graph["output_shape"]) or not np.isfinite(logits).all():
            raise ValueError("Invalid output logits")
        return logits

