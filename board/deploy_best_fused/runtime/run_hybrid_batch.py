"""Reuse three DPU tasks; report accuracy and per-stage wall times, excluding warmup."""
from __future__ import print_function
import argparse
import csv
import json
from pathlib import Path
import time
import numpy as np
from run_hybrid_board import HybridRunner, add_options, load_image
from model_libraries import ensure_library_search_path

EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp'}
TIMES = ['preprocess', 'front_dpu', 'ps1', 'mv2_4', 'ps2', 'mv2_5', 'ps3',
         'dpu_with_transfer', 'numpy_tail', 'total']


def run_batch(args):
    directory = args.image_dir.resolve()
    files = sorted(p for p in directory.rglob('*') if p.is_file() and p.suffix.lower() in EXTENSIONS)
    if args.limit:
        files = files[:args.limit]
    if not files:
        raise ValueError('No images: ' + str(directory))
    labels = []
    names = {'c{}'.format(i): i for i in range(10)}
    for path in files:
        parts = path.relative_to(directory).parts
        if args.labels_from_folders and (len(parts) < 2 or parts[0] not in names):
            raise ValueError('Expected c0...c9 directly below image root: ' + str(path))
        labels.append(names[parts[0]] if args.labels_from_folders else None)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if any((args.output_dir / name).exists() for name in ('summary.json', 'predictions.csv')):
        raise ValueError('Choose a new output directory')
    totals = np.zeros(len(TIMES), dtype=np.float64)
    confusion = np.zeros((10, 10), dtype=np.int64)
    success, failed, correct, warmed = 0, 0, 0, False
    warmup_seconds, processing_start = 0., None
    runner = None
    summary = {'selected_images': len(files), 'image_dir': str(directory),
        'mv2_backend': args.mv2_backend,
        'tail_impl': 'ps_tail_fused' if getattr(args, 'fuse', False) else ('ps_tail_fast' if args.fast_tail else 'ps_tail_numpy'),
        'warmup_runs': args.warmup, 'status': 'failed', 'error': None,
        'input_range_policy': getattr(args, 'input_range_policy', 'error'),
        'timing_note': 'DPU times include FP32 transfer, quantization/dequantization; total includes image decode/preprocess',
        'model_loading_and_warmup_included': False}
    fields = ['image', 'status', 'error', 'true_class', 'predicted_class', 'correct'] + [
        t + '_seconds' for t in TIMES] + ['logit_{}'.format(i) for i in range(10)]
    try:
        with HybridRunner(args.config, args.mv2_backend, args.fast_tail, getattr(args, 'fuse', False),
                input_range_policy=getattr(args, 'input_range_policy', 'error'),
                trace_dpu_input=getattr(args, 'trace_dpu_input', False),
                input_dump_dir=args.output_dir / 'input_debug') as runner, \
                (args.output_dir / 'predictions.csv').open('x', encoding='utf-8', newline='') as stream:
            summary['mv2_transform'] = runner.config.get('mv2_transform')
            summary['dpu_kernels'] = [runner.config['front']['kernel']] + [
                s['kernel'] for s in runner.config['stages'] if s['backend'] == 'dpu' and args.mv2_backend == 'dpu']
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            processing_start = time.perf_counter()
            for index, (path, label) in enumerate(zip(files, labels), 1):
                row = {'image': path.relative_to(directory).as_posix(), 'true_class': label}
                started = time.perf_counter()
                try:
                    image = load_image(path)
                except (ValueError, OSError) as error:
                    failed += 1
                    row.update(status='image_error', error=str(error))
                    writer.writerow(row)
                    stream.flush()
                    continue
                preprocess = time.perf_counter() - started
                if not warmed:
                    started = time.perf_counter()
                    try:
                        for _ in range(args.warmup):
                            runner.predict(image)
                    finally:
                        warmup_seconds += time.perf_counter() - started
                    warmed = True
                started = time.perf_counter()
                result = runner.predict(image)
                times = dict(result['seconds'], preprocess=preprocess,
                             total=preprocess + time.perf_counter() - started)
                prediction = result['class_index']
                row.update(status='ok', error='', predicted_class=prediction,
                           correct=int(prediction == label) if label is not None else '')
                row.update({t + '_seconds': times[t] for t in TIMES})
                row.update({'logit_{}'.format(i): x for i, x in enumerate(result['logits'])})
                writer.writerow(row)
                stream.flush()
                success += 1
                totals += [times[t] for t in TIMES]
                if label is not None:
                    correct += int(prediction == label)
                    confusion[label, prediction] += 1
                print('[{}/{}] {} -> c{} ({:.4f}s)'.format(index, len(files), row['image'], prediction, times['total']), flush=True)
            summary['status'] = 'completed' if not failed else 'completed_with_image_errors'
    except BaseException as error:
        summary['error'] = '{}: {}'.format(type(error).__name__, error)
        raise
    finally:
        if runner is not None:
            runner.save_input_report(args.output_dir)
        wall = max(0., time.perf_counter() - processing_start - warmup_seconds) if processing_start else 0.
        summary.update(successful_images=success, failed_images=failed,
            unprocessed_images=len(files)-success-failed,
            correct_images=correct if args.labels_from_folders else None,
            accuracy=correct/float(success) if args.labels_from_folders and success else None,
            processing_wall_seconds_excluding_warmup=wall,
            processing_images_per_second=success/wall if wall else None,
            mean_seconds=dict(zip(TIMES, (totals/success).tolist() if success else [None]*len(TIMES))),
            confusion_matrix_true_rows_predicted_columns=confusion.tolist() if args.labels_from_folders else None)
        (args.output_dir / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_options(parser)
    parser.add_argument('--image-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, default=Path('batch_mv2_dpu'))
    parser.add_argument('--labels-from-folders', action='store_true')
    parser.add_argument('--warmup', type=int, default=1)
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    if args.warmup < 0 or (args.limit is not None and args.limit < 1):
        parser.error('Invalid warmup/limit')
    ensure_library_search_path(args.config)
    import cv2
    if run_batch(args)['failed_images']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
