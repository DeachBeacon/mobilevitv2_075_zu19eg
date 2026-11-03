"""PC-side conversion of this frozen TF tail to JSON + numeric NPZ weights."""
from collections import defaultdict, deque

import numpy as np
from tensorflow.python.framework import tensor_util

SUPPORTED_OPS = {
    "Placeholder", "Const", "Identity", "StopGradient", "Conv2D",
    "DepthwiseConv2dNative", "FusedBatchNormV3", "BiasAdd", "Relu",
    "Add", "AddV2", "Sub", "Mul", "SquaredDifference", "Rsqrt",
    "Mean", "Sum", "Reshape", "Transpose", "SplitV", "Softmax", "Pad", "MatMul",
}


def edge_name(edge):
    return edge.lstrip("^").split(":")[0]


def topological_nodes(graph_def):
    nodes = {n.name: n for n in graph_def.node}
    children = defaultdict(list)
    degree = {}
    for node in graph_def.node:
        parents = {edge_name(edge) for edge in node.input}
        for parent in parents:
            if parent not in nodes:
                raise ValueError("Missing parent {} for {}".format(parent, node.name))
            children[parent].append(node.name)
        degree[node.name] = len(parents)
    ready = deque(n.name for n in graph_def.node if degree[n.name] == 0)
    result = []
    while ready:
        name = ready.popleft()
        result.append(nodes[name])
        for child in children[name]:
            degree[child] -= 1
            if degree[child] == 0:
                ready.append(child)
    if len(result) != len(nodes):
        raise ValueError("The exported graph must be acyclic")
    return result


def prepare_numpy_export(graph_def, input_tensor, output_tensor):
    unsupported = sorted({n.op for n in graph_def.node} - SUPPORTED_OPS)
    if unsupported:
        raise ValueError("NumPy tail does not implement these ops: {}".format(unsupported))
    weights = {}
    records = []
    for node in topological_nodes(graph_def):
        record = {"name": node.name, "op": node.op,
                  "inputs": [x for x in node.input if not x.startswith("^")], "attrs": {}}
        attrs = record["attrs"]
        if node.op == "Const":
            array = tensor_util.MakeNdarray(node.attr["value"].tensor)
            if array.dtype.kind not in "biuf":
                raise ValueError("Non-numeric constant: {}".format(node.name))
            key = "weight_{:04d}".format(len(weights))
            weights[key] = array
            record["weight_key"] = key
        elif node.op in ("Conv2D", "DepthwiseConv2dNative"):
            for key in ("strides", "dilations", "explicit_paddings"):
                attrs[key] = list(node.attr[key].list.i) if key in node.attr else []
            if not attrs["dilations"]:
                attrs["dilations"] = [1, 1, 1, 1]
            attrs["padding"] = node.attr["padding"].s.decode("ascii")
            attrs["data_format"] = node.attr["data_format"].s.decode("ascii") or "NHWC"
        elif node.op == "FusedBatchNormV3":
            attrs["epsilon"] = float(node.attr["epsilon"].f)
            attrs["is_training"] = bool(node.attr["is_training"].b)
            attrs["data_format"] = node.attr["data_format"].s.decode("ascii") or "NHWC"
            if attrs["is_training"]:
                raise ValueError("Training BatchNorm is not supported: {}".format(node.name))
        elif node.op == "BiasAdd":
            attrs["data_format"] = node.attr["data_format"].s.decode("ascii") or "NHWC"
        elif node.op in ("Mean", "Sum"):
            attrs["keep_dims"] = bool(node.attr["keep_dims"].b)
        elif node.op == "MatMul":
            attrs["transpose_a"] = bool(node.attr["transpose_a"].b)
            attrs["transpose_b"] = bool(node.attr["transpose_b"].b)
        elif node.op == "SplitV":
            attrs["num_split"] = int(node.attr["num_split"].i)
        if "data_format" in attrs and attrs["data_format"] != "NHWC":
            raise ValueError("Only NHWC is implemented: {}".format(node.name))
        records.append(record)
    # Inference BatchNorm's auxiliary outputs have no consumers in this model.
    bn_names = {n.name for n in graph_def.node if n.op == "FusedBatchNormV3"}
    for node in graph_def.node:
        for edge in node.input:
            if not edge.startswith("^") and edge_name(edge) in bn_names and ":" in edge and edge.rsplit(":", 1)[1] != "0":
                raise ValueError("BatchNorm auxiliary output is needed: {}".format(edge))
    graph = {"format_version": 1, "input_tensor": input_tensor, "output_tensor": output_tensor,
             "input_shape": [1, 32, 32, 96], "output_shape": [1, 10],
             "weights_file": "ps_tail_numpy_weights.npz", "nodes": records}
    return graph, weights
