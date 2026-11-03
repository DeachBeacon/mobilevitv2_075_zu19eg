"""Verify the saved front/tail PBs against the original float32 model on images."""
from __future__ import print_function

import argparse
import json
from pathlib import Path

import numpy as np
from common import (ROOT, SAMPLES, SPLIT_MANIFEST, CUT_TENSOR, DEFAULT_ARTIFACT_DIR, DEFAULT_CALIB_DIR, DEFAULT_SOURCE,
                    ImageReader, OUTPUT, SOURCE_INPUT, cpu_config, import_graph,
                    read_graph, sha256, tf1)
import tensorflow as tf
from ps_tail import PSTail
from ps_tail_numpy import NumpyPSTail


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-pb", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--calib-dir", type=Path, default=DEFAULT_CALIB_DIR)
    parser.add_argument("--samples", type=int, default=10, help="Evenly spaced images; default 10")
    parser.add_argument("--verification-dir", type=Path, help="Override sample/report output directory")
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("--samples must be positive")
    artifact_dir = args.artifact_dir.resolve()
    organized = artifact_dir == DEFAULT_ARTIFACT_DIR.resolve()
    manifest_path = SPLIT_MANIFEST if organized else artifact_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    original, original_data = read_graph(args.source_pb)
    if sha256(original_data) != manifest["source"]["sha256"]:
        raise ValueError("Source PB differs from the exported manifest")
    for part in ["dpu_front", "ps_tail"]:
        data = (artifact_dir / manifest[part]["file"]).read_bytes()
        if sha256(data) != manifest[part]["sha256"]:
            raise ValueError("Exported PB changed: {}".format(part))
    front_def, _ = read_graph(artifact_dir / manifest["dpu_front"]["file"])
    numpy_info = manifest["ps_tail_numpy"]
    for key in ("graph", "weights"):
        if sha256((artifact_dir / numpy_info[key + "_file"]).read_bytes()) != numpy_info[key + "_sha256"]:
            raise ValueError("NumPy export changed: {}".format(key))
    numpy_tail = NumpyPSTail(artifact_dir / numpy_info["graph_file"])
    full_graph = import_graph(original)
    front_graph = import_graph(front_def)
    calib_dir = args.calib_dir.resolve()
    images = sorted(p for p in calib_dir.rglob("*") if p.suffix.lower() in (".jpg", ".jpeg"))
    if not images:
        raise ValueError("No calibration images in {}".format(calib_dir))
    indices = np.linspace(0, len(images) - 1, min(args.samples, len(images)), dtype=int)
    verification_dir = args.verification_dir or (SAMPLES if organized else artifact_dir / "verification")
    verification_dir.mkdir(parents=True, exist_ok=True)
    results = []
    reader = ImageReader()
    try:
        with tf1.Session(graph=full_graph, config=cpu_config()) as full_session, \
                tf1.Session(graph=front_graph, config=cpu_config()) as front_session, \
                PSTail(artifact_dir / manifest["ps_tail"]["file"]) as tail:
            for number, index in enumerate(indices):
                image = reader.read(images[index])
                reference_features, reference_logits = full_session.run(
                    [CUT_TENSOR, OUTPUT], {SOURCE_INPUT: image})
                features = front_session.run(CUT_TENSOR, {SOURCE_INPUT: image})
                split_logits = tail.predict(features)
                numpy_logits = numpy_tail.predict(features)
                finite = bool(np.isfinite(reference_logits).all())
                feature_error = float(np.max(np.abs(features - reference_features)))
                logit_error = float(np.max(np.abs(split_logits - reference_logits)))
                numpy_error = float(np.max(np.abs(numpy_logits - reference_logits)))
                numpy_passed = bool(np.allclose(numpy_logits, reference_logits, rtol=1e-4, atol=1e-4)
                                    and np.argmax(numpy_logits[0]) == np.argmax(reference_logits[0]))
                passed = bool(finite and np.allclose(features, reference_features, rtol=1e-5, atol=1e-5)
                              and np.allclose(split_logits, reference_logits, rtol=1e-5, atol=1e-5)
                              and numpy_passed)
                results.append({
                    "image": str(images[index].relative_to(calib_dir)),
                    "max_feature_absolute_error": feature_error,
                    "max_logit_absolute_error": logit_error,
                    "numpy_max_logit_absolute_error": numpy_error,
                    "reference_class_index": int(np.argmax(reference_logits[0])),
                    "split_class_index": int(np.argmax(split_logits[0])), "passed": passed,
                    "numpy_class_index": int(np.argmax(numpy_logits[0])),
                })
                print("[{}/{}] {} feature_error={} TF_error={} NumPy_error={} passed={}".format(
                    number + 1, len(indices), results[-1]["image"], feature_error, logit_error, numpy_error, passed))
                if number == 0:
                    np.save(str(verification_dir / "sample_input_fp32.npy"), image, allow_pickle=False)
                    np.save(str(verification_dir / "sample_features_fp32.npy"), features, allow_pickle=False)
                    np.save(str(verification_dir / "sample_reference_logits.npy"), reference_logits, allow_pickle=False)
                    np.save(str(verification_dir / "sample_split_logits.npy"), split_logits, allow_pickle=False)
                    np.save(str(verification_dir / "sample_numpy_logits.npy"), numpy_logits, allow_pickle=False)
    finally:
        reader.close()
    report = {
        "tensorflow_version": tf.__version__, "source_sha256": sha256(original_data),
        "front_sha256": manifest["dpu_front"]["sha256"], "tail_sha256": manifest["ps_tail"]["sha256"],
        "sample_count": len(results), "passed": all(item["passed"] for item in results),
        "tolerance": {"rtol": 1e-5, "atol": 1e-5}, "results": results,
        "numpy_tolerance": {"rtol": 1e-4, "atol": 1e-4}, "numpy_version": np.__version__,
        "scope": "Saved FP32 PBs executed on the host CPU; no DPU/INT8/ARM/performance validation",
    }
    (verification_dir / "verification.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if not report["passed"]:
        raise SystemExit("FAIL: see verification/verification.json")
    print("PASS: saved front + saved tail match the original model on {} images".format(len(results)))


if __name__ == "__main__":
    main()
