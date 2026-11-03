"""BN folding and positive channel equalization; adapted from the validated case core."""
from collections import defaultdict
import numpy as np
import tensorflow as tf
from tf_helpers import import_graph, cpu_config, tf1

def edge_node(edge):
    return edge.lstrip("^").split(":")[0]


def add_constant(graph_def, name, array):
    if any(n.name == name for n in graph_def.node):
        raise ValueError("Parameter name already exists: {}".format(name))
    node = graph_def.node.add()
    node.name, node.op = name, "Const"
    node.attr["dtype"].type = tf.float32.as_datatype_enum
    node.attr["value"].tensor.CopyFrom(tf.make_tensor_proto(np.asarray(array, dtype=np.float32)))
    return name


def fold_front(front_def, output_tensor, parameter_prefix='cle_params'):
    """Fold Conv/DWConv -> inference BN, cloning all changed parameters."""
    graph = import_graph(front_def)
    folded = tf1.GraphDef()
    folded.CopyFrom(front_def)
    nodes = {n.name: n for n in folded.node}
    layers = []
    with tf1.Session(graph=graph, config=cpu_config()) as session:
        for original in front_def.node:
            if original.op not in ("Conv2D", "DepthwiseConv2dNative"):
                continue
            op = graph.get_operation_by_name(original.name)
            if op.get_attr("data_format") != b"NHWC":
                raise ValueError("Only NHWC is supported")
            weight = session.run(op.inputs[1])
            consumers = list(op.outputs[0].consumers())
            batchnorms = [c for c in consumers if c.type in ("FusedBatchNormV3", "FusedBatchNormV2", "FusedBatchNorm")]
            if batchnorms and (len(batchnorms) != 1 or len(consumers) != 1):
                raise ValueError("BN folding would change another consumer: {}".format(op.name))
            channels = weight.shape[2] * weight.shape[3] if original.op == "DepthwiseConv2dNative" else weight.shape[3]
            bias = np.zeros(channels, dtype=np.float32)
            bn_name = None
            if batchnorms:
                bn = batchnorms[0]
                if bn.get_attr("is_training") or bn.inputs[0].name != op.outputs[0].name:
                    raise ValueError("Only direct inference BN is supported")
                if any(output.consumers() for output in bn.outputs[1:]):
                    raise ValueError("BN auxiliary outputs are in use: {}".format(bn.name))
                gamma, beta, mean, variance = session.run(bn.inputs[1:5])
                alpha = gamma / np.sqrt(variance + np.float32(bn.get_attr("epsilon")))
                factor = alpha.reshape(1, 1, weight.shape[2], weight.shape[3]) if original.op == "DepthwiseConv2dNative" else alpha.reshape(1, 1, 1, -1)
                weight = weight * factor
                bias = beta - mean * alpha
                bn_name = bn.name
            elif len(consumers) == 1 and consumers[0].type == "BiasAdd":
                affine = consumers[0]
                if affine.inputs[0].name != op.outputs[0].name:
                    raise ValueError("Unexpected BiasAdd layout")
                bias = session.run(affine.inputs[1])
                bn_name = affine.name
            index = len(layers)
            weight_name = add_constant(folded, parameter_prefix + "/layer_{:02d}/weights".format(index), weight)
            nodes[original.name].input[1] = weight_name
            bias_name = None
            if bn_name:
                bias_name = add_constant(folded, parameter_prefix + "/layer_{:02d}/bias".format(index), bias)
                replacement = nodes[bn_name]
                replacement.op = "BiasAdd"
                replacement.ClearField("input")
                replacement.input.extend([original.name, bias_name])
                replacement.ClearField("attr")
                replacement.attr["T"].type = tf.float32.as_datatype_enum
                replacement.attr["data_format"].s = b"NHWC"
            layers.append({"name": original.name, "op": original.op, "input": edge_node(original.input[0]),
                           "weight_node": weight_name, "bias_node": bias_name,
                           "weight": weight.astype(np.float64), "bias": bias.astype(np.float64)})
    folded = tf1.graph_util.extract_sub_graph(folded, [output_tensor.split(":")[0]])
    for node in folded.node:
        node.device = ""
    return folded, layers


class ChannelGroups(object):
    def __init__(self):
        self.parents = {}
        self.channels = {}

    def add(self, name, channels):
        if name not in self.parents:
            self.parents[name] = name
            self.channels[name] = channels

    def find(self, name):
        if self.parents[name] != name:
            self.parents[name] = self.find(self.parents[name])
        return self.parents[name]

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if self.channels[a] != self.channels[b]:
            raise ValueError("Channel mismatch at scale constraint")
        if a != b:
            first, second = sorted([a, b])
            self.parents[second] = first


def feature_groups(graph_def, layers, input_tensor, output_tensor):
    graph = import_graph(graph_def)
    groups = ChannelGroups()
    visited = set()
    additions = []

    def visit(name):
        if name in visited:
            return
        op = graph.get_operation_by_name(name)
        tensor = op.outputs[0]
        dims = tensor.shape.as_list()
        if tensor.dtype != tf.float32 or len(dims) != 4 or dims[-1] is None:
            raise ValueError("Expected fixed-channel NHWC feature: {}".format(name))
        groups.add(name, dims[-1])
        visited.add(name)
        if op.type == "Placeholder":
            if tensor.name != input_tensor:
                raise ValueError("Unexpected runtime input: {}".format(name))
        elif op.type in ("Conv2D", "DepthwiseConv2dNative"):
            visit(op.inputs[0].op.name)
        elif op.type in ("BiasAdd", "Identity", "Relu", "Pad"):
            parent = op.inputs[0].op.name
            visit(parent)
            groups.union(name, parent)
        elif op.type in ("Add", "AddV2"):
            branches = [t.op.name for t in op.inputs]
            for parent in branches:
                visit(parent)
                groups.union(name, parent)
            additions.append({"node": name, "branches": branches})
        else:
            raise ValueError("Unsupported feature operator: {} [{}]".format(name, op.type))

    visit(output_tensor.split(":")[0])
    members = defaultdict(list)
    for name in sorted(visited):
        members[groups.find(name)].append(name)
    pinned = {groups.find(input_tensor.split(":")[0]), groups.find(output_tensor.split(":")[0])}
    for layer in layers:
        layer["input_group"] = groups.find(layer["input"])
        layer["output_group"] = groups.find(layer["name"])
        if layer["input_group"] == layer["output_group"]:
            pinned.add(layer["input_group"])  # Conservative handling of a self-coupled residual group.
    return groups, dict(members), pinned, additions


def scaled_weight(layer, scales):
    incoming, outgoing = scales[layer["input_group"]], scales[layer["output_group"]]
    if layer["op"] == "DepthwiseConv2dNative":
        factor = outgoing.reshape(layer["weight"].shape[2:]) / incoming[:, None]
        return layer["weight"] * factor[None, None, :, :]
    return layer["weight"] * outgoing[None, None, None, :] / incoming[None, None, :, None]


def input_ranges(layer, weight):
    return np.max(np.abs(weight), axis=(0, 1, 3))


def output_ranges(layer, weight):
    if layer["op"] == "DepthwiseConv2dNative":
        return np.max(np.abs(weight), axis=(0, 1)).reshape(-1)
    return np.max(np.abs(weight), axis=(0, 1, 2))


def equalize(layers, groups, members, pinned, iterations=64, damping=0.5):
    scales = {name: np.ones(groups.channels[name], dtype=np.float64) for name in members}
    history = []
    for iteration in range(iterations):
        largest = 0.0
        for group in members:
            if group in pinned:
                continue
            producers = [l for l in layers if l["output_group"] == group]
            consumers = [l for l in layers if l["input_group"] == group]
            if not producers or not consumers:
                raise ValueError("Unanchored channel scale: {}".format(group))
            before = np.maximum.reduce([output_ranges(l, scaled_weight(l, scales)) for l in producers])
            after = np.maximum.reduce([input_ranges(l, scaled_weight(l, scales)) for l in consumers])
            valid = (before > 1e-12) & (after > 1e-12)
            log_adjustment = np.zeros_like(before)
            log_adjustment[valid] = 0.5 * damping * (np.log(after[valid]) - np.log(before[valid]))
            updated = np.clip(scales[group] * np.exp(np.clip(log_adjustment, -np.log(4), np.log(4))), 1e-4, 1e4)
            largest = max(largest, float(np.max(np.abs(np.log(updated / scales[group])))))
            scales[group] = updated
        history.append({"iteration": iteration + 1, "largest_log_scale_update": largest})
        if largest < 1e-4:
            break
    return scales, history


def apply_scales(graph_def, layers, scales):
    candidate = tf1.GraphDef()
    candidate.CopyFrom(graph_def)
    nodes = {n.name: n for n in candidate.node}
    stats = []
    for layer in layers:
        weight = np.asarray(scaled_weight(layer, scales), dtype=np.float32)
        bias = np.asarray(layer["bias"] * scales[layer["output_group"]], dtype=np.float32)
        if not np.isfinite(weight).all() or not np.isfinite(bias).all():
            raise ValueError("Non-finite equalized parameters")
        nodes[layer["weight_node"]].attr["value"].tensor.CopyFrom(tf.make_tensor_proto(weight))
        if layer["bias_node"]:
            nodes[layer["bias_node"]].attr["value"].tensor.CopyFrom(tf.make_tensor_proto(bias))
        elif np.any(bias):
            raise ValueError("Final output unexpectedly requires a new bias")
        stats.append({"name": layer["name"], "op": layer["op"],
                      "before_weight_abs_max": float(np.max(np.abs(layer["weight"]))),
                      "after_weight_abs_max": float(np.max(np.abs(weight))),
                      "before_bias_abs_max": float(np.max(np.abs(layer["bias"]))),
                      "after_bias_abs_max": float(np.max(np.abs(bias)))})
    return candidate, stats


