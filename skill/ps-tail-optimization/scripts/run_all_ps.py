"""Pure-PS (CPU-only) reference pipeline for the final MobileViTv2-0.75 deployment.

Runs every segment with NumPy on the board CPU:
    front -> ps1 -> mv2_4 -> ps2 -> mv2_5 -> ps3

This is the FP32 reference used to compare against the DPU+PS split pipeline.
It uses the same graphs as the split version but executes all of them with NumPy.

Usage:
    python3 runtime/run_all_ps.py --image-dir ./test1_p --labels-from-folders \
        --output-dir batch_ps_300
"""
from __future__ import print_function

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

from ps_tail_numpy import NumpyPSTail, preprocess_rgb

HERE = Path(__file__).resolve().parents[1]
STAGE_NAMES = ["front", "ps1", "mv2_4", "ps2", "mv2_5", "ps3"]
EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
CLASS_COUNT = 10


def load_image(path):
    import cv2
    bgr = cv2.imread(str(path))
    if bgr is None:
        raise ValueError("Cannot decode {}".format(path))
    return preprocess_rgb(bgr[..., ::-1])


def build_tails(model_dir, impl):
    if impl == "fused":
        from ps_tail_fused import FusedTail as factory
    elif impl == "fast":
        from ps_tail_fast import FastTail as factory
    else:
        factory = NumpyPSTail
    tails = []
    for name in STAGE_NAMES:
        graph = model_dir / (name + ".json")
        if not graph.is_file():
            raise FileNotFoundError("Missing graph: {}".format(graph))
        tails.append((name, factory(graph)))
    return tails


def run(args):
    directory = Path(args.image_dir).resolve()
    files = sorted(p for p in directory.rglob("*") if p.is_file() and p.suffix.lower() in EXTENSIONS)
    if args.limit:
        files = files[:args.limit]
    if not files:
        raise ValueError("No images: " + str(directory))

    labels = []
    names = {"c{}".format(i): i for i in range(CLASS_COUNT)}
    for path in files:
        parts = path.relative_to(directory).parts
        if args.labels_from_folders and (len(parts) < 2 or parts[0] not in names):
            raise ValueError("Expected c0...c9 directly below image root: " + str(path))
        labels.append(names[parts[0]] if args.labels_from_folders else None)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir / "summary.json").exists():
        raise ValueError("Choose a new output directory")

    tails = build_tails(Path(args.model_dir), args.tail_impl)
    totals = {name: 0.0 for name in ["preprocess"] + STAGE_NAMES + ["numpy_tail", "total"]}
    confusion = np.zeros((CLASS_COUNT, CLASS_COUNT), dtype=np.int64)
    success = failed = correct = 0
    warmup = max(0, args.warmup)
    rows = []
    status = "failed"
    error = None
    wall_start = wall_end = None
    try:
        with (output_dir / "predictions.csv").open("x", encoding="utf-8", newline="") as stream:
            fields = ["image", "status", "error", "true_class", "predicted_class", "correct"] + \
                     [name + "_seconds" for name in ["preprocess"] + STAGE_NAMES] + \
                     ["total_seconds"] + ["logit_{}".format(i) for i in range(CLASS_COUNT)]
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for index, path in enumerate(files):
                record = {"image": str(path.relative_to(directory)), "status": "ok", "error": "",
                          "true_class": labels[index] if labels[index] is not None else ""}
                try:
                    started = time.perf_counter()
                    value = load_image(path)
                    record["preprocess_seconds"] = time.perf_counter() - started
                    for name, tail in tails:
                        stage_started = time.perf_counter()
                        value = tail.predict(value)
                        record[name + "_seconds"] = time.perf_counter() - stage_started
                    logits = np.asarray(value).reshape(-1)
                    if logits.shape != (CLASS_COUNT,) or not np.isfinite(logits).all():
                        raise RuntimeError("Invalid logits")
                    record["total_seconds"] = time.perf_counter() - started
                    predicted = int(logits.argmax())
                    record["predicted_class"] = predicted
                    record["correct"] = int(labels[index] == predicted) if labels[index] is not None else ""
                    for i in range(CLASS_COUNT):
                        record["logit_{}".format(i)] = float(logits[i])
                    success += 1
                    if labels[index] is not None and predicted == labels[index]:
                        correct += 1
                    if labels[index] is not None:
                        confusion[labels[index]][predicted] += 1
                    if index >= warmup:
                        for key in ["preprocess"] + STAGE_NAMES:
                            totals[key] += record[key + "_seconds"]
                        totals["total"] += record["total_seconds"]
                except Exception as exc:  # noqa: BLE001 - keep the batch running
                    record.update({"status": "failed", "error": str(exc)})
                    if index >= warmup:
                        failed += 1
                writer.writerow(record)
                rows.append(record)
                if wall_start is None and index >= warmup:
                    wall_start = time.perf_counter()
        wall_end = time.perf_counter()
        status = "completed"
    except BaseException as exc:
        error = str(exc)
        raise
    finally:
        counted = max(1, success)
        totals["numpy_tail"] = sum(totals[name] for name in STAGE_NAMES)
        mean_seconds = {key: round(totals[key] / counted, 6) for key in totals}
        summary = {
            "mode": "all_ps",
            "image_dir": str(directory),
            "model_dir": str(Path(args.model_dir).resolve()),
            "selected_images": len(files),
            "tail_impl": {"fused": "ps_tail_fused", "fast": "ps_tail_fast",
                          "numpy": "ps_tail_numpy"}[args.tail_impl],
            "warmup_runs": warmup,
            "status": status,
            "error": error,
            "successful_images": success,
            "failed_images": failed,
            "unprocessed_images": max(0, len(files) - success - failed),
            "correct_images": correct,
            "accuracy": (correct / success) if success else None,
            "processing_wall_seconds_excluding_warmup": round((wall_end - wall_start), 6) if wall_start else None,
            "model_loading_and_warmup_included": False,
            "mean_seconds": mean_seconds,
            "confusion_matrix_true_rows_predicted_columns": confusion.tolist() if args.labels_from_folders else None,
        }
        (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=HERE / "model")
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("batch_ps"))
    parser.add_argument("--labels-from-folders", action="store_true")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--fast-tail", dest="fast", action="store_true", default=True)
    parser.add_argument("--no-fast-tail", dest="fast", action="store_false")
    parser.add_argument("--fuse", action="store_true")
    args = parser.parse_args()
    if args.fuse:
        args.tail_impl = "fused"
    elif args.fast:
        args.tail_impl = "fast"
    else:
        args.tail_impl = "numpy"
    run(args)


if __name__ == "__main__":
    main()
