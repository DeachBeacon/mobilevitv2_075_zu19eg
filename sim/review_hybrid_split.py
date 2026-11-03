"""Read-only model/code audit; write results only to diagnostics/hybrid_review_20261029.json."""
from __future__ import print_function
import sys
from pathlib import Path
_PROJECT=Path(__file__).resolve().parents[1]
for _path in (_PROJECT/'src/dpu',_PROJECT/'src/dpu/split',_PROJECT/'board/deploy_best_fused/runtime'):
    sys.path.insert(0,str(_path))


from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import struct
import sys
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
sys.dont_write_bytecode = True
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_paths import PROJECT, ROOT, SOURCE, FRONT, TAIL, NUMPY_GRAPH, CALIB, CLE, SAMPLES, SPLIT_MANIFEST, configure_imports
configure_imports()


import numpy as np
import tensorflow as tf
from common import CUT_TENSOR, OUTPUT, SOURCE_INPUT, ImageReader, cpu_config, import_graph, read_graph
from export_numpy import prepare_numpy_export
from ps_tail import PSTail
from ps_tail_numpy import NumpyPSTail
from split_model import split_graph


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def elf_info(path):
    data = path.read_bytes()
    if data[:6] != b"\x7fELF\x02\x01":
        raise ValueError("Expected a 64-bit little-endian ELF: {}".format(path))
    header = struct.unpack_from("<16sHHIQQQIHHHHHH", data)
    headers = [struct.unpack_from("<IIQQQQIIQQ", data, header[6] + i * header[11])
               for i in range(header[12])]
    names_header = headers[header[13]]
    names = data[names_header[4]:names_header[4] + names_header[5]]
    sections = {}
    for section in headers:
        name = names[section[0]:].split(b"\0", 1)[0].decode("ascii")
        if section[1] != 8:  # SHT_NOBITS has no stored payload.
            sections[name] = data[section[4]:section[4] + section[5]]
    return {"path": str(path.relative_to(ROOT)), "sha256": digest(path),
            "type": header[1], "machine": header[2],
            "dpu_sections": {k: {"bytes": len(v), "sha256": hashlib.sha256(v).hexdigest()}
                             for k, v in sections.items() if "dpu" in k.lower() or "deephi" in k.lower()}}, sections


def main():
    artifacts = ROOT / "models"
    source = SOURCE
    manifest = json.loads((SPLIT_MANIFEST).read_text(encoding="utf-8"))
    report = {"python": sys.version, "tensorflow": tf.__version__, "numpy": np.__version__}
    report["source_hash_matches_manifest"] = digest(source) == manifest["source"]["sha256"]
    paths = {"dpu_front_fp32.pb": FRONT, "ps_tail_fp32.pb": TAIL,
             "ps_tail_numpy.json": NUMPY_GRAPH,
             "ps_tail_numpy_weights.npz": NUMPY_GRAPH.parent / "ps_tail_numpy_weights.npz"}
    board = ROOT / "deploy_in_board/baseline/model"
    report["artifact_copies"] = {}
    for name, path in paths.items():
        copies = [path] + ([board / name] if (board / name).is_file() else [])
        report["artifact_copies"][name] = {str(p.relative_to(PROJECT)): digest(p) for p in copies}
    report["all_artifact_copies_equal"] = all(len(set(v.values())) == 1 for v in report["artifact_copies"].values())
    report["artifact_hashes_match_manifest"] = (
        digest(FRONT) == manifest["dpu_front"]["sha256"]
        and digest(TAIL) == manifest["ps_tail"]["sha256"]
        and digest(NUMPY_GRAPH) == manifest["ps_tail_numpy"]["graph_sha256"]
        and digest(NUMPY_GRAPH.parent / "ps_tail_numpy_weights.npz") == manifest["ps_tail_numpy"]["weights_sha256"])
    original, _ = read_graph(source)
    front, tail = split_graph(original)
    report["fresh_split_matches_saved"] = {
        # Protobuf map entries need not serialize in the same order; compare messages.
        name: graph == read_graph(paths[name])[0]
        for name, graph in [("dpu_front_fp32.pb", front), ("ps_tail_fp32.pb", tail)]}
    fresh_graph, fresh_weights = prepare_numpy_export(tail, "ps_features:0", OUTPUT)
    saved_graph = json.loads((NUMPY_GRAPH).read_text(encoding="utf-8"))
    report["numpy_topology_matches_fresh_export"] = fresh_graph == saved_graph
    with np.load(str(NUMPY_GRAPH.parent / "ps_tail_numpy_weights.npz"), allow_pickle=False) as archive:
        report["numpy_weights_match_fresh_export"] = set(archive.files) == set(fresh_weights) and all(
            np.array_equal(archive[key], fresh_weights[key]) for key in fresh_weights)
    report["graphs"] = {}
    for name, path in [("source", source), ("front", FRONT),
                       ("tail", TAIL),
                       ("quantized_deploy", ROOT / "models/front_baseline/int8/deploy_model.pb"),
                       ("quantized_eval", ROOT / "models/front_baseline/int8/quantize_eval_model.pb")]:
        graph, _ = read_graph(path)
        last = graph.node[-1]
        report["graphs"][name] = {"nodes": len(graph.node), "ops": dict(Counter(n.op for n in graph.node)),
                                  "last_node": last.name, "last_op": last.op}
        if name.startswith("quantized"):
            report["graphs"][name]["cut_nodes"] = [
                {"name": n.name, "op": n.op,
                 "quant_attrs": {k: str(v).strip() for k, v in n.attr.items()
                                 if k in ("ipos", "opos", "wpos", "bpos", "quantize_pos")}}
                for n in graph.node if CUT_TENSOR.split(":")[0] in n.name]
    elf, elf_sections = elf_info(ROOT / "models/front_baseline/compiled/dpu_mobilevitv2_075_front.elf")
    shared, shared_sections = elf_info(ROOT / "models/front_baseline/compiled/libdpumodelmobilevitv2_075_front.so")
    report["binaries"] = [elf, shared]
    report["dpu_sections_preserved_in_so"] = bool(elf["dpu_sections"]) and all(
        shared_sections.get(k) == elf_sections[k] for k in elf["dpu_sections"])
    full_graph = import_graph(original)
    front_graph = import_graph(read_graph(FRONT)[0])
    numpy_tail = NumpyPSTail(NUMPY_GRAPH)
    images = sorted((CALIB).rglob("*.jpg"))
    indices = np.linspace(0, len(images) - 1, 10, dtype=int)
    results = []
    reader = ImageReader()
    try:
        with tf.Session(graph=full_graph, config=cpu_config()) as full_session, \
                tf.Session(graph=front_graph, config=cpu_config()) as front_session, \
                PSTail(TAIL) as tf_tail:
            for index in indices:
                image = reader.read(images[index])
                features, reference = full_session.run([CUT_TENSOR, OUTPUT], {SOURCE_INPUT: image})
                split_features = front_session.run(CUT_TENSOR, {SOURCE_INPUT: image})
                split_logits = tf_tail.predict(split_features)
                started = time.perf_counter()
                numpy_logits = numpy_tail.predict(split_features)
                elapsed = time.perf_counter() - started
                result = {"image": str(images[index].relative_to(CALIB)),
                          "feature_max_abs_error": float(np.max(np.abs(features - split_features))),
                          "tf_logits_max_abs_error": float(np.max(np.abs(reference - split_logits))),
                          "numpy_logits_max_abs_error": float(np.max(np.abs(reference - numpy_logits))),
                          "reference_class": int(reference.argmax()), "numpy_class": int(numpy_logits.argmax()),
                          "host_numpy_seconds": elapsed,
                          "passed": bool(np.array_equal(features, split_features)
                                         and np.allclose(reference, split_logits, rtol=1e-5, atol=1e-5)
                                         and np.allclose(reference, numpy_logits, rtol=1e-4, atol=1e-4)
                                         and reference.argmax() == numpy_logits.argmax())}
                results.append(result)
                print(json.dumps(result), flush=True)
            sample = np.load(str(SAMPLES / "sample_features_fp32.npy"), allow_pickle=False)
            reference = np.load(str(SAMPLES / "sample_reference_logits.npy"), allow_pickle=False)
            report["saved_sample_numpy_max_abs_error"] = float(np.max(np.abs(numpy_tail.predict(sample) - reference)))
            rng = np.random.RandomState(42)
            perturbed = sample + rng.normal(0, 0.01, sample.shape).astype(np.float32)
            report["perturbed_feature_numpy_max_abs_error"] = float(
                np.max(np.abs(numpy_tail.predict(perturbed) - tf_tail.predict(perturbed))))
    finally:
        reader.close()
    report["samples"] = results
    report["float_checks_passed"] = bool(report["source_hash_matches_manifest"]
        and report["all_artifact_copies_equal"] and report["artifact_hashes_match_manifest"]
        and all(report["fresh_split_matches_saved"].values())
        and report["numpy_topology_matches_fresh_export"] and report["numpy_weights_match_fresh_export"]
        and all(item["passed"] for item in results))
    report["scope"] = "Host CPU FP32 and static binary audit only; no DPU, ARM, or INT8 accuracy test."
    output = ROOT / "diagnostics/hybrid_review_current.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("Report:", output)
    if not report["float_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
