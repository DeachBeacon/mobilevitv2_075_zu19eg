"""Shared N2Cube task with validated HWC interfaces and FP32 transfer."""
import numpy as np
import json
from pathlib import Path
import sys
from ps_tail_numpy import preprocess_rgb

def check_status(name, status):
    if status not in (None, 0):
        raise RuntimeError('{} failed: {}'.format(name, status))


def load_image(path):
    import cv2
    bgr = cv2.imread(str(path))
    if bgr is None:
        raise ValueError('Cannot decode {}'.format(path))
    return preprocess_rgb(bgr[..., ::-1])


class DpuStage(object):
    """Shares one opened N2Cube context with the other stage."""
    def __init__(self, api, config, input_shape, output_shape,
                 input_range_policy='error', trace_input=False, input_dump_dir=None):
        self.api, self.config = api, config
        self.kernel, self.task = None, None
        self.input_shape, self.output_shape = tuple(input_shape), tuple(output_shape)
        if input_range_policy not in ('error', 'clip'):
            raise ValueError('input_range_policy must be error or clip')
        self.input_range_policy, self.trace_input = input_range_policy, trace_input
        self.input_dump_dir = Path(input_dump_dir) if input_dump_dir is not None else None
        self.input_statistics = {'kernel': config['kernel'], 'input_node': config['input_node'],
            'policy': input_range_policy, 'calls': 0, 'elements': 0, 'below_range': 0,
            'above_range': 0, 'calls_outside_range': 0, 'clipped_elements': 0,
            'input_min': None, 'input_max': None, 'last': None}
        try:
            self.kernel = api.dpuLoadKernel(config['kernel'])
            if not self.kernel:
                raise RuntimeError('Cannot load {} (check matching libdpumodel*.so)'.format(config['kernel']))
            self.task = api.dpuCreateTask(self.kernel, 0)
            if not self.task:
                raise RuntimeError('Cannot create task')
            for kind, shape in [('Input', self.input_shape), ('Output', self.output_shape)]:
                node = config[kind.lower() + '_node']
                actual = tuple(int(getattr(api, 'dpuGet' + kind + 'Tensor' + dim)(self.task, node))
                               for dim in ('Height', 'Width', 'Channel'))
                size = int(getattr(api, 'dpuGet' + kind + 'TensorSize')(self.task, node))
                if actual != shape[1:] or size != int(np.prod(shape)):
                    raise ValueError('{} {} boundary mismatch: {} size {}, expected {}'.format(
                        config['kernel'], kind, actual, size, shape))
            self.input_scale = float(api.dpuGetInputTensorScale(self.task, config['input_node']))
            if not np.isfinite(self.input_scale) or self.input_scale <= 0:
                raise ValueError('{}: invalid DPU input scale {}'.format(config['kernel'], self.input_scale))
            expected_scale = config.get('expected_input_scale')
            if expected_scale is not None and not np.isclose(self.input_scale, float(expected_scale), rtol=1e-6, atol=0):
                raise ValueError('{}: loaded DPU input scale {} differs from build expectation {}. '
                    'Check deploy Placeholder.opos, recompile/relink, and copy the matching SO/config together.'.format(
                        config['kernel'], self.input_scale, expected_scale))
            # Keep the runtime's existing FP32 -> INT8 rounding implementation.
            # Only bound the FP32 input when the caller explicitly selects clip.
            self.input_lower = np.float32(-128.0 / self.input_scale)
            self.input_upper = np.float32(127.0 / self.input_scale)
            if (not np.isfinite(self.input_lower) or not np.isfinite(self.input_upper)
                    or self.input_lower >= self.input_upper):
                raise ValueError('{}: invalid representable input interval'.format(config['kernel']))
            if float(self.input_lower) * self.input_scale < -128.0:
                self.input_lower = np.nextafter(self.input_lower, np.float32(np.inf))
            if float(self.input_upper) * self.input_scale > 127.0:
                self.input_upper = np.nextafter(self.input_upper, np.float32(-np.inf))
            self.input_statistics.update(input_scale=self.input_scale,
                representable_input_min=float(self.input_lower), representable_input_max=float(self.input_upper))
        except BaseException:
            self.close()
            raise

    def predict(self, value):
        if self.task is None:
            raise RuntimeError('DPU stage is closed')
        value = np.asarray(value)
        if value.dtype.kind != 'f' or value.shape != self.input_shape or not np.isfinite(value).all():
            raise ValueError('Invalid {} input {}'.format(self.config['kernel'], value.shape))
        value = np.ascontiguousarray(value, dtype=np.float32).reshape(-1)
        if not np.isfinite(value).all():
            raise ValueError('{}: input cannot be represented as finite float32'.format(self.config['kernel']))
        minimum, maximum = float(value.min()), float(value.max())
        low = int(np.count_nonzero(value < self.input_lower)) if minimum < self.input_lower else 0
        high = int(np.count_nonzero(value > self.input_upper)) if maximum > self.input_upper else 0
        stats = self.input_statistics
        stats['calls'] += 1
        stats['elements'] += value.size
        stats['below_range'] += low
        stats['above_range'] += high
        stats['calls_outside_range'] += int(bool(low or high))
        stats['input_min'] = minimum if stats['input_min'] is None else min(stats['input_min'], minimum)
        stats['input_max'] = maximum if stats['input_max'] is None else max(stats['input_max'], maximum)
        stats['last'] = {'call': stats['calls'], 'input_min': minimum, 'input_max': maximum,
            'scaled_min': minimum * self.input_scale, 'scaled_max': maximum * self.input_scale,
            'below_range': low, 'above_range': high, 'outside_fraction': (low + high) / float(value.size)}
        if self.trace_input or stats['calls'] == 1 or low or high:
            print('[DPU input] kernel={} scale={} fp32=[{:.8g}, {:.8g}] '
                  'scaled=[{:.8g}, {:.8g}] outside={}/{} policy={}'.format(
                      self.config['kernel'], self.input_scale, minimum, maximum,
                      stats['last']['scaled_min'], stats['last']['scaled_max'],
                      low + high, value.size, self.input_range_policy), file=sys.stderr, flush=True)
        if low or high:
            if self.input_dump_dir is not None and stats['calls_outside_range'] == 1:
                self.input_dump_dir.mkdir(parents=True, exist_ok=True)
                path = self.input_dump_dir / (self.config['kernel'] + '_first_outside_input.npy')
                np.save(str(path), value.reshape(self.input_shape), allow_pickle=False)
                path.with_suffix('.json').write_text(json.dumps(stats, indent=2), encoding='utf-8')
            if self.input_range_policy == 'error':
                raise ValueError('{}: DPU input outside representable FP32 interval [{}, {}] '
                    '(scale={}, scaled range [{}, {}], outside={}/{}). '
                    'Inspect dpu_input_ranges.json and input_debug/. For an explicit saturation run, '
                    'use --input-range-policy clip and compare accuracy.'.format(
                        self.config['kernel'], self.input_lower, self.input_upper, self.input_scale,
                        stats['last']['scaled_min'], stats['last']['scaled_max'], low + high, value.size))
            value = np.clip(value, self.input_lower, self.input_upper)
            stats['clipped_elements'] += low + high
        check_status('dpuSetInputTensorInHWCFP32', self.api.dpuSetInputTensorInHWCFP32(
            self.task, self.config['input_node'], value, value.size))
        check_status('dpuRunTask', self.api.dpuRunTask(self.task))
        output = np.asarray(self.api.dpuGetOutputTensorInHWCFP32(
            self.task, self.config['output_node'], int(np.prod(self.output_shape))), dtype=np.float32)
        if output.size != int(np.prod(self.output_shape)) or not np.isfinite(output).all():
            raise RuntimeError('Invalid DPU output')
        # Copy before the next task can touch runtime-owned buffers. API already dequantizes.
        return output.reshape(self.output_shape).copy()

    def close(self):
        task, kernel = self.task, self.kernel
        self.task, self.kernel = None, None
        try:
            if task:
                self.api.dpuDestroyTask(task)
        finally:
            if kernel:
                self.api.dpuDestroyKernel(kernel)


