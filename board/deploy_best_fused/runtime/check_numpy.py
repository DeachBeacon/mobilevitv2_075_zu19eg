"""No-DPU smoke test and NumPy per-stage timing, using a saved FP32 feature."""
from __future__ import print_function
import argparse
import json
from pathlib import Path
import time
import numpy as np
from ps_tail_fast import FastTail

HERE = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repeat', type=int, default=3)
    parser.add_argument('--features-npy', type=Path, default=HERE/'samples/sample_front_features_fp32.npy')
    args = parser.parse_args()
    if args.repeat < 1:
        parser.error('repeat must be positive')
    config = json.loads((HERE/'runtime_config.json').read_text(encoding='utf-8'))
    stages = [(s['name'], FastTail(HERE/'model'/s['graph'])) for s in config['stages']]
    baseline = FastTail(HERE/'model/ps_tail_numpy.json')
    features = np.load(str(args.features_npy),allow_pickle=False)
    expected = baseline.predict(features)
    value = features
    for _,runner in stages:
        value = runner.predict(value)
    passed = bool(np.allclose(value,expected,rtol=1e-4,atol=1e-4) and value.argmax() == expected.argmax())
    totals = {name:0. for name,_ in stages}
    baseline_seconds = 0.
    for _ in range(args.repeat):
        start = time.perf_counter()
        baseline.predict(features)
        baseline_seconds += time.perf_counter()-start
        value = features
        for name,runner in stages:
            start = time.perf_counter()
            value = runner.predict(value)
            totals[name] += time.perf_counter()-start
    times = {k:v/args.repeat for k,v in totals.items()}
    print(json.dumps({'passed':passed,'numpy_version':np.__version__,
        'max_logit_abs_error':float(np.abs(value-expected).max()), 'class_index':int(value.argmax()),
        'baseline_full_tail_seconds':baseline_seconds/args.repeat,'split_numpy_mean_seconds':times,
        'mv2_numpy_seconds':times['mv2_4']+times['mv2_5'],
        'scope':'Saved FP32 feature; no DPU execution. Actual speedup also includes two new DPU tasks and transfer.'},indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
