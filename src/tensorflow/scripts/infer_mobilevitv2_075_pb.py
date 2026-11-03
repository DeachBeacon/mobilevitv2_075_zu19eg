"""Infer one 256x256 image with the pure-inference MobileViTv2 PB in TF 1.15."""
import argparse
import os
from _paths import PROJECT, TF_ROOT, TF_RECORDS, TF_WEIGHTS, INFERENCE, DEPLOY, REFERENCE_PB

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf


DEFAULT_PB = str(INFERENCE)


def preprocess_rgb_uint8(image):
    if image.dtype != np.uint8 or image.shape != (256, 256, 3):
        raise ValueError(
            "Expected a 256x256 RGB image; got shape={}, dtype={}. "
            "No automatic resizing is applied.".format(image.shape, image.dtype)
        )
    values = image.astype(np.float32) / np.float32(255.0)
    mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
    return ((values - mean) / std)[None, ...]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, help="Path to one 256x256 image (for example JPG or PNG)")
    parser.add_argument("--pb", default=DEFAULT_PB, help="Pure-inference GraphDef; defaults to the new exported PB")
    args = parser.parse_args()
    if tuple(tf.__version__.split(".")[:2]) != ("1", "15"):
        parser.error("Run this script with the specified TensorFlow 1.15 environment")
    for path in (args.image, args.pb):
        if not os.path.isfile(path):
            parser.error("File does not exist: {}".format(path))

    graph_def = tf.compat.v1.GraphDef()
    with tf.io.gfile.GFile(args.pb, "rb") as source:
        graph_def.ParseFromString(source.read())
    graph = tf.Graph()
    with graph.as_default():
        tf.import_graph_def(graph_def, name="")
        image_path = tf.compat.v1.placeholder(tf.string, shape=[], name="inference_image_path")
        decoded_rgb = tf.image.decode_image(tf.io.read_file(image_path), channels=3, name="inference_decode_rgb")
    input_tensor = graph.get_tensor_by_name("input_1:0")
    output_tensor = graph.get_tensor_by_name("predictions/BiasAdd:0")
    if input_tensor.dtype != tf.float32 or input_tensor.shape.as_list() != [None, 256, 256, 3]:
        raise ValueError("Unexpected PB input: {} {}".format(input_tensor.shape, input_tensor.dtype))
    if output_tensor.dtype != tf.float32 or output_tensor.shape.as_list() != [None, 10]:
        raise ValueError("Unexpected PB output: {} {}".format(output_tensor.shape, output_tensor.dtype))

    config = tf.compat.v1.ConfigProto(device_count={"GPU": 0}, allow_soft_placement=True)
    # Frozen weights need no H5, Keras model, variable initialization or training flag.
    with tf.compat.v1.Session(graph=graph, config=config) as session:
        rgb = session.run(decoded_rgb, feed_dict={image_path: os.path.abspath(args.image)})
        inputs = preprocess_rgb_uint8(rgb)
        logits = session.run(output_tensor, feed_dict={input_tensor: inputs})
    if logits.dtype != np.float32 or logits.shape != (1, 10) or not np.isfinite(logits).all():
        raise RuntimeError("PB returned invalid logits")
    # PB returns logits. Softmax is only for displaying classification probabilities.
    scores = np.exp(logits[0] - np.max(logits[0]))
    probabilities = scores / np.sum(scores)
    class_index = int(np.argmax(logits[0]))
    print("TensorFlow: {}".format(tf.__version__))
    print("PB: {}".format(os.path.abspath(args.pb)))
    print("Image: {}".format(os.path.abspath(args.image)))
    print("Input: shape={}, dtype={}".format(inputs.shape, inputs.dtype))
    print("Class index: {}".format(class_index))
    print("Softmax probability: {:.6f}".format(float(probabilities[class_index])))
    print("Logits: {}".format(logits[0].tolist()))
    print("Probabilities (index 0..9): {}".format(probabilities.tolist()))


if __name__ == "__main__":
    main()
