"""CLE front -> PS1 -> MV2_4 -> PS2 -> MV2_5 -> PS3; N2Cube + NumPy."""
from __future__ import print_function
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from dpu_runtime import DpuStage, check_status, load_image
from ps_tail_fast import FastTail
from ps_tail_fused import FusedTail
from ps_tail_numpy import NumpyPSTail
from model_libraries import ensure_library_search_path, inspect_libraries

HERE = Path(__file__).resolve().parents[1]


class HybridRunner(object):
    def __init__(self, config_path=HERE / 'runtime_config.json', mv2_backend='dpu',
                 fast_tail=True, fused_tail=False, api=None, input_range_policy='error',
                 trace_dpu_input=False, input_dump_dir=None):
        if mv2_backend not in ('dpu', 'numpy'):
            raise ValueError('Unknown MV2 backend')
        self.root = Path(config_path).resolve().parent
        self.config = json.loads(Path(config_path).read_text(encoding='utf-8'))
        self.mv2_backend = mv2_backend
        self.tasks, self.stages, self.opened = [], [], False
        self.dpu_input_report = {'scope': 'All DPU submissions, including warmup and any rejected input. '
            'Outside counts use the conservative FP32 endpoint interval, not a simulation of runtime rounding.',
            'policy': input_range_policy, 'stages': []}
        front = self.config['front']
        if self.config.get('format_version') != 1 or [s['name'] for s in self.config['stages']] != [
                'ps1', 'mv2_4', 'ps2', 'mv2_5', 'ps3']:
            raise ValueError('Wrong execution graph')
        for stage in [front] + [s for s in self.config['stages'] if s['backend'] == 'dpu' and mv2_backend == 'dpu']:
            if not all(stage.get(k) for k in ('kernel', 'input_node', 'output_node', 'so_sha256')):
                raise ValueError('Missing compiled runtime metadata for {}; run Docker steps 5/6 and copy '
                                 'the generated runtime_config.json with model/'.format(stage.get('name', 'front')))
            so = self.root / 'model' / ('libdpumodel' + stage['kernel'] + '.so')
            if not so.is_file():
                for cand in (self.root / 'model').glob('*.so'):
                    try:
                        from model_libraries import elf_kernel_names
                        if stage['kernel'] in elf_kernel_names(cand.read_bytes()):
                            so = cand
                            break
                    except Exception:
                        pass
            if not so.is_file():
                raise ValueError('Missing SO: {}. Finish Docker steps 5/6 and copy the generated model/ directory.'.format(so))
            actual_sha = hashlib.sha256(so.read_bytes()).hexdigest()
            if actual_sha != stage['so_sha256']:
                try:
                    from model_libraries import elf_kernel_names
                    if stage['kernel'] in elf_kernel_names(so.read_bytes()):
                        print('[warning] SO hash differs for {}: {} vs expected {}. Proceeding with matching embedded kernel.'
                              .format(stage['kernel'], actual_sha, stage['so_sha256']), file=sys.stderr)
                    else:
                        raise ValueError('Shared library differs from runtime config: ' + str(so))
                except Exception as e:
                    if isinstance(e, ValueError):
                        raise
                    raise ValueError('Shared library differs from runtime config: ' + str(so))
        if fused_tail:
            factory = FusedTail
        elif fast_tail:
            factory = FastTail
        else:
            factory = NumpyPSTail
        previous = front['output_shape']
        for stage in self.config['stages']:
            if stage['input_shape'] != previous:
                raise ValueError('Disconnected stage ' + stage['name'])
            previous = stage['output_shape']
            runner = None
            if stage['backend'] == 'numpy' or mv2_backend == 'numpy':
                runner = factory(self.root / 'model' / stage['graph'])
                if runner.graph['input_shape'] != stage['input_shape'] or runner.graph['output_shape'] != stage['output_shape']:
                    raise ValueError('NumPy boundary mismatch: ' + stage['name'])
            self.stages.append((stage, runner))
        if api is None:
            inspection = inspect_libraries(config_path, mv2_backend)
            if not inspection['passed']:
                raise ValueError('; '.join(r['stage'] + ': ' + r['error'] for r in
                                          inspection['libraries'] if not r['passed']))
            from dnndk import n2cube
            api = n2cube
        self.api = api
        check_status('dpuOpen', api.dpuOpen())
        self.opened = True
        try:
            stage_options = dict(input_range_policy=input_range_policy, trace_input=trace_dpu_input,
                                 input_dump_dir=input_dump_dir)
            self.front = DpuStage(api, front, front['input_shape'], front['output_shape'], **stage_options)
            self.tasks.append(self.front)
            self.dpu_input_report['stages'].append(self.front.input_statistics)
            for i, (stage, runner) in enumerate(self.stages):
                if runner is None:
                    runner = DpuStage(api, stage, stage['input_shape'], stage['output_shape'], **stage_options)
                    self.tasks.append(runner)
                    self.dpu_input_report['stages'].append(runner.input_statistics)
                    self.stages[i] = (stage, runner)
        except BaseException:
            self.close()
            raise

    def predict(self, image):
        if not self.opened:
            raise RuntimeError('Runner is closed')
        times = {}
        start = time.perf_counter()
        value = self.front.predict(image)
        times['front_dpu'] = time.perf_counter() - start
        for stage, runner in self.stages:
            start = time.perf_counter()
            value = runner.predict(value)
            times[stage['name']] = time.perf_counter() - start
        logits = np.asarray(value).reshape(-1)
        if logits.shape != (10,) or not np.isfinite(logits).all():
            raise RuntimeError('Invalid final logits')
        times['dpu_with_transfer'] = times['front_dpu'] + sum(
            times[s['name']] for s, _ in self.stages if s['backend'] == 'dpu' and self.mv2_backend == 'dpu')
        times['numpy_tail'] = sum(times[s['name']] for s, _ in self.stages if
            s['backend'] == 'numpy' or self.mv2_backend == 'numpy')
        return {'class_index': int(logits.argmax()), 'logits': logits.tolist(), 'seconds': times}

    def save_input_report(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / 'dpu_input_ranges.json').write_text(json.dumps(self.dpu_input_report, indent=2), encoding='utf-8')

    def close(self):
        tasks, opened = self.tasks, self.opened
        self.tasks, self.opened = [], False
        try:
            for runner in reversed(tasks):
                runner.close()
        finally:
            if opened:
                self.api.dpuClose()

    def __enter__(self):
        return self

    def __exit__(self, *unused):
        self.close()


def add_options(parser):
    parser.add_argument('--config', type=Path, default=HERE / 'runtime_config.json')
    parser.add_argument('--mv2-backend', choices=('dpu', 'numpy'), default='dpu')
    parser.add_argument('--fast-tail', dest='fast_tail', action='store_true', default=True,
                        help='Fast NumPy implementation (default)')
    parser.add_argument('--no-fast-tail', dest='fast_tail', action='store_false')
    parser.add_argument('--fuse', dest='fuse', action='store_true',
                        help='Fuse GroupNorm/attention in the NumPy stages (ps1/ps2/ps3)')
    parser.add_argument('--input-range-policy', choices=('error', 'clip'), default='error',
                        help='Stop with range diagnostics (default), or explicitly saturate inputs using each DPU scale')
    parser.add_argument('--trace-dpu-input', action='store_true', help='Print every DPU input range, including warmup')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_options(parser)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--image', type=Path)
    source.add_argument('--input-npy', type=Path)
    args = parser.parse_args()
    ensure_library_search_path(args.config)
    image = np.load(str(args.input_npy), allow_pickle=False) if args.input_npy else load_image(args.image)
    with HybridRunner(args.config, args.mv2_backend, args.fast_tail, args.fuse,
                      input_range_policy=args.input_range_policy, trace_dpu_input=args.trace_dpu_input) as runner:
        print(json.dumps(runner.predict(image), indent=2))


if __name__ == '__main__':
    main()
