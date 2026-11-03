"""Standard-library-only provenance and regression gates for the CLE workflow."""
from __future__ import print_function

import argparse
import hashlib
import json
from pathlib import Path
import struct

NET = "mobilevitv2_075_front_cle"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check(directory, stage, max_drop=0.06, min_samples=300):
    directory = directory.resolve()
    structured = (directory / "fp32").is_dir()
    records = directory / "records" if structured else directory
    fp32 = directory / "fp32" if structured else directory
    quantized_dir = directory / "int8" if structured else directory / "quantize"
    compiled_dir = directory / "compiled" if structured else directory / "compile"
    manifest = read(records / "manifest.json")
    verification = read(records / "fp32_verification.json")
    require(manifest["fp32_verification_passed"] and verification["passed"], "FP32 equivalence has not passed")
    require(digest(fp32 / manifest["candidate_file"]) == manifest["candidate_sha256"], "Candidate PB changed after verification")
    if stage == "float":
        return
    quant_files = {"candidate_sha256": manifest["candidate_sha256"],
                   "eval_sha256": digest(quantized_dir / "quantize_eval_model.pb"),
                   "deploy_sha256": digest(quantized_dir / "deploy_model.pb")}
    if stage == "record-quantized":
        with (records / "quantization_manifest.json").open("x", encoding="utf-8") as stream:
            json.dump(quant_files, stream, indent=2)
        return
    require(read(records / "quantization_manifest.json") == quant_files, "Quantization artifacts do not match this candidate")
    if stage == "quantized":
        return
    native = read(records / "native_quant_accuracy.json")
    require(native["backend"] == "native", "A native FixNeuron evaluation is required")
    require(native["source_sha256"] == manifest["source_sha256"], "Evaluation used a different FP32 reference")
    require(native.get("tail_sha256") == manifest["tail_sha256"], "Evaluation used a different tail or an outdated evaluator")
    require(native["quantized_graph_sha256"] == quant_files["eval_sha256"], "Evaluation used a different quantized graph")
    require(native["sample_count"] >= min_samples, "Not enough evaluated images (need {})".format(min_samples))
    require(len(native["results"]) == native["sample_count"] and all(r["label"] in range(10) for r in native["results"]),
            "Evaluation requires a known c0 ... c9 label for every image")
    floating = native["accuracy"]["float_class"]
    quantized = native["accuracy"]["quantized_front_class"]
    require(floating is not None and quantized is not None, "Missing classification accuracy")
    measured_float = sum(r["float_class"] == r["label"] for r in native["results"]) / float(native["sample_count"])
    measured_quant = sum(r["quantized_front_class"] == r["label"] for r in native["results"]) / float(native["sample_count"])
    require(abs(floating - measured_float) < 1e-12 and abs(quantized - measured_quant) < 1e-12,
            "Accuracy summary is inconsistent with per-image results")
    accuracy_drop = floating - quantized
    # Inclusive boundary: 0.90 - 0.84 can be 0.06000000000000005 in binary FP.
    gate = {"passed": accuracy_drop <= max_drop + 1e-12, "float_accuracy": floating,
            "quantized_accuracy": quantized, "accuracy_drop": floating - quantized,
            "max_allowed_drop": max_drop, "sample_count": native["sample_count"],
            "scope": "Native regression gate; an independent validation set is still required for acceptance"}
    (records / "accuracy_gate.json").write_text(json.dumps(gate, indent=2), encoding="utf-8")
    require(gate["passed"], "Native INT8 accuracy drop {:.2%} exceeds allowed {:.2%}; compilation stopped".format(
        gate["accuracy_drop"], max_drop))
    if stage == "native":
        return
    elf_path = compiled_dir / ("dpu_" + NET + ".elf")
    compile_files = {"deploy_sha256": quant_files["deploy_sha256"], "elf_sha256": digest(elf_path), "kernel": NET}
    if stage == "record-compiled":
        topology = (compiled_dir / (NET + "_kernel_graph.gv")).read_text(encoding="utf-8")
        require(topology.count("type: DPUKernel") == 1 and "CPUKernel" not in topology, "Expected one DPU kernel and no CPU kernel")
        with (records / "compile_manifest.json").open("x", encoding="utf-8") as stream:
            json.dump(compile_files, stream, indent=2)
        return
    require(read(records / "compile_manifest.json") == compile_files, "Compiled ELF does not match recorded artifacts")
    if stage == "compiled":
        return
    shared = compiled_dir / ("libdpumodel" + NET + ".so")
    data = shared.read_bytes()
    require(data[:6] == b"\x7fELF\x02\x01" and struct.unpack_from("<HH", data, 16) == (3, 183),
            "Expected an AArch64 shared library")
    (records / "linked_manifest.json").write_text(json.dumps(
        dict(compile_files, shared_sha256=digest(shared)), indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--stage", choices=("float", "record-quantized", "quantized", "native",
                                            "record-compiled", "compiled", "record-linked"), required=True)
    parser.add_argument("--max-drop", type=float, default=0.06)
    parser.add_argument("--min-samples", type=int, default=300)
    args = parser.parse_args()
    if not 0 <= args.max_drop <= 1 or args.min_samples < 1:
        parser.error("Invalid --max-drop or --min-samples")
    try:
        check(args.artifact_dir, args.stage, args.max_drop, args.min_samples)
    except (ValueError, OSError, KeyError) as error:
        raise SystemExit("CLE gate failed: {}".format(error))
    print("CLE gate passed:", args.stage)


if __name__ == "__main__":
    main()
