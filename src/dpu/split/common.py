"""Shared I/O for the fixed MobileViTv2 two-part split (TensorFlow 1.15)."""
from __future__ import print_function

from collections import Counter
import hashlib
import os
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_paths import PROJECT, ROOT, SOURCE, FRONT, TAIL, NUMPY_GRAPH, CALIB, CLE, SAMPLES, SPLIT_MANIFEST, configure_imports
configure_imports()

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import numpy as np
import tensorflow as tf

tf1 = tf.compat.v1
DEFAULT_SOURCE = SOURCE
DEFAULT_ARTIFACT_DIR = ROOT / "models"
DEFAULT_CALIB_DIR = CALIB
SOURCE_INPUT = "input_1:0"
CUT_TENSOR = "stack3_block2_pre_2_conv/Conv2D:0"
TAIL_INPUT = "ps_features:0"
OUTPUT = "predictions/BiasAdd:0"
IMAGE_SHAPE = [1, 256, 256, 3]
FEATURE_SHAPE = [1, 32, 32, 96]
LOGITS_SHAPE = [1, 10]
MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def read_graph(path):
    data = Path(path).read_bytes()
    graph_def = tf1.GraphDef()
    graph_def.ParseFromString(data)
    return graph_def, data


def import_graph(graph_def):
    graph = tf.Graph()
    with graph.as_default():
        tf1.import_graph_def(graph_def, name="")
    return graph


def cpu_config():
    return tf1.ConfigProto(device_count={"GPU": 0}, allow_soft_placement=True,
                           intra_op_parallelism_threads=2, inter_op_parallelism_threads=2)


def validate_interface(graph, input_name, output_name, input_shape, output_shape):
    placeholders = [op.name for op in graph.get_operations()
                    if op.type in ("Placeholder", "PlaceholderWithDefault")]
    expected = [input_name.split(":")[0]]
    if placeholders != expected:
        raise ValueError("Expected only input {}; got {}".format(expected, placeholders))
    for name, expected_shape in [(input_name, input_shape), (output_name, output_shape)]:
        tensor = graph.get_tensor_by_name(name)
        if tensor.dtype != tf.float32 or tensor.shape.as_list() != expected_shape:
            raise ValueError("Unexpected interface {}: {} {}".format(name, tensor.dtype, tensor.shape))


def graph_info(graph_def, filename, input_name, output_name, input_shape, output_shape):
    data = graph_def.SerializeToString()
    return {
        "file": filename, "sha256": sha256(data), "bytes": len(data),
        "node_count": len(graph_def.node),
        "ops": dict(sorted(Counter(n.op for n in graph_def.node).items())),
        "input_tensor": input_name, "output_tensor": output_name,
        "input_shape": input_shape, "output_shape": output_shape,
        "dtype": "float32", "layout": "NHWC",
    }


class ImageReader(object):
    """Match the existing input_fn: decode RGB, require 256x256, mean/std."""

    def __init__(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.path = tf1.placeholder(tf.string, shape=[])
            self.rgb = tf.image.decode_image(tf.io.read_file(self.path), channels=3)
        self.session = tf1.Session(graph=self.graph, config=cpu_config())

    def read(self, path):
        rgb = self.session.run(self.rgb, {self.path: str(path)})
        if rgb.dtype != np.uint8 or rgb.shape != (256, 256, 3):
            raise ValueError("Expected 256x256 RGB uint8 image: {} ({}, {})".format(
                path, rgb.shape, rgb.dtype))
        image = rgb.astype(np.float32) / np.float32(255.0)
        image = ((image - MEAN) / STD).astype(np.float32)
        if not np.isfinite(image).all():
            raise ValueError("Non-finite image: {}".format(path))
        return image[np.newaxis, ...]

    def close(self):
        self.session.close()
