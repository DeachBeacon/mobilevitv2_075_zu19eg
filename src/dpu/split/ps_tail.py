"""Execute the PS tail with a compatible TensorFlow CPU installation.

The input must be dequantized FP32 NHWC features, not raw DPU INT8 bytes.
This entry point can be tested on Windows tf115 before an ARM backend is ready.
"""
from __future__ import print_function

import argparse
import json
from pathlib import Path

import numpy as np
from common import (TAIL, DEFAULT_ARTIFACT_DIR, FEATURE_SHAPE, LOGITS_SHAPE, OUTPUT,
                    TAIL_INPUT, cpu_config, import_graph, read_graph, tf1, validate_interface)


class PSTail(object):
    """Load once, then call predict for each feature tensor."""

    def __init__(self, model_path):
        graph_def, _ = read_graph(model_path)
        self.graph = import_graph(graph_def)
        validate_interface(self.graph, TAIL_INPUT, OUTPUT, FEATURE_SHAPE, LOGITS_SHAPE)
        self.input = self.graph.get_tensor_by_name(TAIL_INPUT)
        self.output = self.graph.get_tensor_by_name(OUTPUT)
        self.session = tf1.Session(graph=self.graph, config=cpu_config())

    def predict(self, features):
        features = np.asarray(features)
        if features.dtype.kind != "f":
            raise ValueError("PS tail requires real floating-point features; dequantize DPU INT8 first")
        if features.shape == tuple(FEATURE_SHAPE[1:]):
            features = features[np.newaxis, ...]
        if features.shape != tuple(FEATURE_SHAPE):
            raise ValueError("Expected NHWC {}; got {}".format(FEATURE_SHAPE, features.shape))
        features = np.asarray(features, dtype=np.float32)
        if not np.isfinite(features).all():
            raise ValueError("Non-finite input features")
        logits = self.session.run(self.output, {self.input: features})
        if not np.isfinite(logits).all():
            raise ValueError("Non-finite output logits")
        return logits

    def close(self):
        self.session.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=TAIL)
    parser.add_argument("--features-npy", type=Path, required=True)
    parser.add_argument("--probabilities", action="store_true", help="Also calculate final classification softmax")
    args = parser.parse_args()
    features = np.load(str(args.features_npy), allow_pickle=False)
    with PSTail(args.model) as tail:
        logits = tail.predict(features)
    result = {"class_index": int(np.argmax(logits[0])), "logits": logits[0].tolist()}
    if args.probabilities:
        probabilities = np.exp(logits[0] - np.max(logits[0]))
        probabilities /= np.sum(probabilities)
        result["probabilities"] = probabilities.tolist()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
