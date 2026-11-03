"""Calibration input for the frozen 256x256 RGB MobileViTv2 classifier."""
import os

import numpy as np
import tensorflow as tf

CALIB_IMAGE_DIR = os.environ.get("MOBILEVIT_CALIB_DIR")
if not CALIB_IMAGE_DIR or not os.path.isdir(CALIB_IMAGE_DIR):
    raise ValueError("Set MOBILEVIT_CALIB_DIR to the real dataset directory.")

# The supplied dataset contains 300 JPG files under c0 ... c9.
IMAGE_FILES = sorted(
    os.path.join(directory, filename)
    for directory, _, filenames in os.walk(CALIB_IMAGE_DIR)
    for filename in filenames
    if filename.lower().endswith((".jpg", ".jpeg"))
)
if len(IMAGE_FILES) != 300:
    raise ValueError(
        "Expected the supplied 300 calibration images, found {}."
        .format(len(IMAGE_FILES))
    )

MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)

# Keep decoding operations out of the model/quantizer default graph.
_DECODE_GRAPH = tf.Graph()
with _DECODE_GRAPH.as_default():
    _IMAGE_PATH = tf.placeholder(tf.string, shape=[])
    _DECODED_RGB = tf.image.decode_image(tf.read_file(_IMAGE_PATH), channels=3)
_DECODE_SESSION = tf.Session(
    graph=_DECODE_GRAPH,
    config=tf.ConfigProto(device_count={"GPU": 0}, allow_soft_placement=True),
)


def calib_input(iter):
    if iter < 0 or iter >= len(IMAGE_FILES):
        raise IndexError("Calibration iteration out of range: {}".format(iter))

    path = IMAGE_FILES[iter]
    rgb = _DECODE_SESSION.run(_DECODED_RGB, feed_dict={_IMAGE_PATH: path})
    if rgb.dtype != np.uint8 or rgb.shape != (256, 256, 3):
        raise ValueError(
            "Expected 256x256 RGB uint8 image: {} (shape={}, dtype={})"
            .format(path, rgb.shape, rgb.dtype)
        )

    image = rgb.astype(np.float32) / np.float32(255.0)
    image = ((image - MEAN) / STD).astype(np.float32)
    if not np.isfinite(image).all():
        raise ValueError("Non-finite calibration input: {}".format(path))
    return {"input_1": image[np.newaxis, ...]}
