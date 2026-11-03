# -*- coding: utf-8 -*-
"""Convert this project's Apple MobileViTv2-0.75 pure .pt weights to Keras HDF5.

Run with py312. The script launches py312 for PyTorch export and tf115 for
TensorFlow conversion, so neither environment needs the other framework.
The TensorFlow 1.15 adapter executes selected definitions from the original
KECAM source. It does not modify that project or hand-reimplement the model.
Only the confirmed 0.75x / 10 classes / ReLU / 2x2 patch configuration is used.
"""
from __future__ import print_function

import argparse
import ast
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import types
import uuid

PROJECT = Path(__file__).resolve().parents[4]
DEFAULT_PT = str(PROJECT / 'data/weights/bestmodel_v1_only_weights.pt')
DEFAULT_APPLE = str(PROJECT / 'src/pytorch/model_source')
DEFAULT_KECAM = str(PROJECT / 'src/tensorflow/conversion/source')
DEFAULT_ENVS = Path(r'C:\ProgramData\anaconda3\envs')
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
ATTENTION_BLOCKS = {3: 2, 4: 4, 5: 3}


def read_json(path):
    with open(str(path), 'r', encoding='utf-8') as stream:
        return json.load(stream)


def write_json(path, value):
    with open(str(path), 'w', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)


def sha256(path):
    digest = hashlib.sha256()
    with open(str(path), 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def preprocess_rgb(images):
    """RGB uint8/float pixels in [0,255], HWC/NHWC -> float32 ImageNet inputs.

    Does not resize or crop. The saved weights do not embed preprocessing.
    """
    import numpy as np
    images = np.asarray(images)
    if images.ndim not in (3, 4) or images.shape[-1] != 3:
        raise ValueError('Expected RGB HWC or NHWC images with 3 channels.')
    if not np.all(np.isfinite(images)) or images.min() < 0 or images.max() > 255:
        raise ValueError('Expected finite raw RGB pixels in [0,255].')
    return (images.astype('float32') / 255.0 - np.array(MEAN, dtype='float32')) / np.array(STD, dtype='float32')


def target_name(source):
    """Explicit semantic names; never pair unrelated layers by list position."""
    if source == 'conv_1.block.conv':
        return 'stem_conv'
    if source == 'conv_1.block.norm':
        return 'stem_bn'
    if source == 'classifier.1':
        return 'predictions'
    match = re.fullmatch(r'layer_(\d+)\.(\d+)\.block\.(exp_1x1|conv_3x3|red_1x1)\.block\.(conv|norm)', source)
    if match:
        stage, block, unit, kind = match.groups()
        unit_id = {'exp_1x1': 1, 'conv_3x3': 2, 'red_1x1': 3}[unit]
        tail = 'bn' if kind == 'norm' else ('dw_conv' if unit_id == 2 else 'conv')
        return 'stack{}_block{}_deep_{}_{}'.format(stage, int(block)+1, unit_id, tail)
    match = re.fullmatch(r'layer_([345])\.1\.local_rep\.([01])\.block\.(conv|norm)', source)
    if match:
        stage, unit, kind = match.groups()
        tail = 'bn' if kind == 'norm' else ('dw_conv' if unit == '0' else 'conv')
        return 'stack{}_block2_pre_{}_{}'.format(stage, int(unit)+1, tail)
    match = re.fullmatch(r'layer_([345])\.1\.global_rep\.(\d+)(?:\.(.+))?', source)
    if match:
        stage, block, tail = match.groups()
        stage, block = int(stage), int(block)
        if tail is None and block == ATTENTION_BLOCKS[stage]:
            return 'stack{}_block{}_post_group_norm'.format(stage, block+1)
        tails = {
            'pre_norm_attn.0': 'attn_group_norm',
            'pre_norm_attn.1.qkv_proj.block.conv': 'attn_mhsa_qkv_conv',
            'pre_norm_attn.1.out_proj.block.conv': 'attn_mhsa_outputconv',
            'pre_norm_ffn.0': 'mlp_group_norm',
            'pre_norm_ffn.1.block.conv': 'mlp_Conv_0',
            'pre_norm_ffn.3.block.conv': 'mlp_Conv_1',
        }
        if block < ATTENTION_BLOCKS[stage] and tail in tails:
            return 'stack{}_block{}_{}'.format(stage, block+2, tails[tail])
    match = re.fullmatch(r'layer_([345])\.1\.conv_proj\.block\.(conv|norm)', source)
    if match:
        stage, kind = match.groups()
        return 'stack{}_block{}_post_1_{}'.format(stage, ATTENTION_BLOCKS[int(stage)]+1, 'conv' if kind == 'conv' else 'bn')
    raise ValueError('No explicit Keras name mapping for {}'.format(source))


def export_torch(config):
    import numpy as np
    import torch
    from torch import nn
    sys.path.insert(0, config['apple_root'])
    from cvnets.models.classification.mobilevit_v2 import MobileViTv2
    from cvnets.layers import LinearLayer
    torch.set_num_threads(config['threads'])
    torch.manual_seed(20261026)
    state = torch.load(config['weights'], map_location='cpu', weights_only=True)
    if not isinstance(state, dict) or not state or not all(torch.is_tensor(v) for v in state.values()):
        raise ValueError('Expected a pure state_dict, not a full training checkpoint.')
    clean_state = {}
    for name, value in state.items():
        clean_name = name[7:] if name.startswith('module.') else name
        if clean_name in clean_state:
            raise ValueError('Duplicate state key after stripping module.: '+clean_name)
        clean_state[clean_name] = value
    state = clean_state
    if tuple(state['conv_1.block.conv.weight'].shape) != (24, 3, 3, 3):
        raise ValueError('Stem does not match MobileViTv2-0.75.')
    if tuple(state['classifier.1.weight'].shape) != (10, 384):
        raise ValueError('Classifier does not match the confirmed 10-class model.')
    opts = argparse.Namespace(**{
        'model.classification.n_classes': 10,
        'model.classification.gradient_checkpointing': False,
        'model.classification.enable_layer_wise_lr_decay': False,
        'model.classification.layer_wise_lr_decay_rate': 1.0,
        'model.classification.freeze_batch_norm': False,
        'model.classification.classifier_dropout': 0.1,
        'model.classification.activation.name': 'relu',
        'model.classification.activation.inplace': False,
        'model.classification.activation.neg_slope': 0.1,
        'model.activation.name': 'relu',
        'model.activation.inplace': False,
        'model.activation.neg_slope': 0.1,
        'model.normalization.name': 'batch_norm',
        'model.normalization.momentum': 0.1,
        'model.normalization.eps': 1e-5,
        'model.normalization.groups': 32,
        'model.layer.conv_init': 'kaiming_normal',
        'model.layer.linear_init': 'trunc_normal',
        'model.layer.global_pool': 'mean',
        'model.classification.mitv2.width_multiplier': 0.75,
        'model.classification.mitv2.attn_dropout': 0.1,
        'model.classification.mitv2.ffn_dropout': 0.1,
        'model.classification.mitv2.dropout': 0.2,
        'model.classification.mitv2.attn_norm_layer': 'layer_norm_2d',
        'common.enable_coreml_compatible_module': False,
    })
    model = MobileViTv2(opts).cpu().eval()
    model.load_state_dict(state, strict=True)
    arrays, records, consumed = {}, [], set()
    modules = []
    for name, layer in model.named_modules():
        if not isinstance(layer, (nn.Conv2d, nn.BatchNorm2d, nn.GroupNorm, nn.Linear, LinearLayer)):
            continue
        record = {'source': name, 'target': target_name(name), 'weights': [], 'source_keys': []}
        if isinstance(layer, nn.Conv2d):
            record['kind'] = 'depthwise' if layer.groups == layer.in_channels and layer.in_channels > 1 else 'conv'
            if record['kind'] == 'conv' and layer.groups != 1:
                raise ValueError('Unexpected grouped convolution: '+name)
            local_keys = ['weight'] + (['bias'] if layer.bias is not None else [])
        elif isinstance(layer, nn.BatchNorm2d):
            record['kind'] = 'bn'
            local_keys = ['weight', 'bias', 'running_mean', 'running_var']
        elif isinstance(layer, nn.GroupNorm):
            if layer.num_groups != 1 or abs(layer.eps-1e-5) > 1e-12:
                raise ValueError('Expected GroupNorm(groups=1, eps=1e-5): '+name)
            record['kind'] = 'gn'
            local_keys = ['weight', 'bias']
        else:
            record['kind'] = 'dense'
            local_keys = ['weight'] + (['bias'] if layer.bias is not None else [])
        for key in local_keys:
            source_key = name+'.'+key
            value = state[source_key].detach().cpu().numpy()
            if value.dtype != np.float32 or not np.isfinite(value).all():
                raise ValueError('Expected finite float32 weights: '+source_key)
            array_key = 'weight_{:03d}_{:02d}'.format(len(records), len(record['weights']))
            arrays[array_key] = value.copy()
            record['weights'].append(array_key)
            record['source_keys'].append(source_key)
            consumed.add(source_key)
        records.append(record)
        modules.append(layer)
    skipped = sorted(k for k in state if k.endswith('.num_batches_tracked'))
    remaining = set(state) - consumed - set(skipped)
    if remaining or len(records) != 111 or len(set(r['target'] for r in records)) != 111:
        raise ValueError('Incomplete or ambiguous mapping: remaining={}, layers={}'.format(sorted(remaining), len(records)))
    size = config['image_size']
    random = np.random.RandomState(20261026)
    yy, xx = np.meshgrid(np.arange(size), np.arange(size), indexing='ij')
    gradient = np.stack([xx % 256, yy % 256, (xx+yy) % 256], axis=-1).astype('uint8')
    raw_cases = [
        ('random_rgb', random.randint(0, 256, size=(1, size, size, 3)).astype('uint8')),
        ('black_rgb', np.zeros((1, size, size, 3), dtype='uint8')),
        ('gradient_rgb_batch2', np.stack([gradient, gradient[::-1, ::-1, ::-1]], axis=0)),
    ]
    handles = []
    def make_hook(index):
        def hook(_module, _inputs, output):
            arrays['feature_{:03d}'.format(index)] = output.detach().cpu().numpy().copy()
        return hook
    for index, layer in enumerate(modules):
        handles.append(layer.register_forward_hook(make_hook(index)))
    cases = []
    for index, (name, raw) in enumerate(raw_cases):
        x = preprocess_rgb(raw)
        arrays['input_'+str(index)] = x
        with torch.inference_mode():
            output = model(torch.from_numpy(x.transpose(0, 3, 1, 2).copy())).detach().cpu().numpy()
        arrays['logits_'+str(index)] = output.copy()
        cases.append({'name': name, 'input': 'input_'+str(index), 'logits': 'logits_'+str(index)})
        if index == 0:
            for handle in handles:
                handle.remove()
    metadata = {
        'source_weights': config['weights'], 'source_sha256': sha256(config['weights']),
        'torch_version': torch.__version__, 'python_torch': sys.version,
        'state_entries': len(state), 'mapped_layers': records,
        'skipped_only': skipped, 'strict_load': True,
        'learnable_parameters': sum(p.numel() for p in model.parameters()),
        'apple_block_class': type(model.layer_3[1]).__module__+'.'+type(model.layer_3[1]).__name__,
        'cases': cases,
    }
    np.savez_compressed(str(Path(config['run_dir'])/'torch_export.npz'), **arrays)
    write_json(Path(config['run_dir'])/'torch_export.json', metadata)
    print('[PyTorch] Strict load passed; {} state entries, {} mapped layers, {} ignored BN counters.'.format(len(state), len(records), len(skipped)), flush=True)


def build_tf_model(keras_project=DEFAULT_KECAM, image_size=256):
    """Build using the original KECAM definitions under TF 1.15, without import-all.

    This narrow compatibility adapter avoids unrelated models and modern imports.
    Conv2D's unsupported groups=1 keyword is removed; GroupNorm(1) is supplied;
    symbolic TensorFlow operations are wrapped in native Keras Lambda layers.
    Weight-bearing layer names and the original computation are preserved.
    """
    import numpy as np
    import tensorflow as tf
    tf.compat.v1.enable_v2_tensorshape()
    tf.keras.backend.set_image_data_format('channels_last')
    class GroupNormalization(tf.keras.layers.Layer):
        def __init__(self, groups=1, axis=-1, epsilon=1e-5, **kwargs):
            if groups != 1 or axis != -1:
                raise ValueError('This adapter supports only GN(1), channels_last.')
            self.groups, self.axis, self.epsilon = groups, axis, epsilon
            super(GroupNormalization, self).__init__(**kwargs)
        def build(self, input_shape):
            channels = int(input_shape[-1])
            self.gamma = self.add_weight(name='gamma', shape=(channels,), initializer='ones')
            self.beta = self.add_weight(name='beta', shape=(channels,), initializer='zeros')
            super(GroupNormalization, self).build(input_shape)
        def call(self, inputs):
            mean, variance = tf.nn.moments(inputs, axes=[1, 2, 3], keep_dims=True)
            return (inputs-mean)*tf.math.rsqrt(variance+self.epsilon)*self.gamma+self.beta
        def get_config(self):
            result = super(GroupNormalization, self).get_config()
            result.update(groups=self.groups, axis=self.axis, epsilon=self.epsilon)
            return result
    class LayerAdapter(object):
        def __getattr__(self, name):
            return getattr(tf.keras.layers, name)
        def Conv2D(self, *args, **kwargs):
            if kwargs.pop('groups', 1) != 1:
                raise ValueError('Unsupported grouped Conv2D in target model.')
            return tf.keras.layers.Conv2D(*args, **kwargs)
        def BatchNormalization(self, *args, **kwargs):
            # TF1.15 fused_batch_norm clamps epsilon to 1.001e-5. PyTorch
            # uses exactly 1e-5; fused=False preserves the trained equation.
            kwargs['fused'] = False
            return tf.keras.layers.BatchNormalization(*args, **kwargs)
    layer_api = LayerAdapter()
    layer_api.GroupNormalization = GroupNormalization
    class FunctionalAdapter(object):
        def __getattr__(self, name):
            op = getattr(tf, name)
            def wrapped(inputs, *args, **kwargs):
                return layer_api.Lambda(lambda x: op(x, *args, **kwargs))(inputs)
            return wrapped
        def relu(self, inputs):
            return layer_api.Activation('relu')(inputs)
        def resize(self, inputs, size, method='bilinear'):
            raise ValueError('Internal resize is not allowed in the validated 256/multiple-of-64 configuration.')
    backend = types.SimpleNamespace(
        is_torch_backend=False, is_tensorflow_backend=True,
        image_data_format=lambda: 'channels_last',
        register_keras_serializable=lambda **kwargs: lambda obj: obj,
    )
    def no_pretrained(_model, _dictionary, _release, pretrained):
        if pretrained is not None:
            raise ValueError('Pretrained downloads are disabled; use the specified .pt.')
    namespace = dict(
        __name__='mobilevit_tf115_compat', math=math, np=np, backend=backend,
        layers=layer_api, models=tf.keras.models, functional=FunctionalAdapter(),
        initializers=tf.keras.initializers, image_data_format=lambda: 'channels_last',
        BATCH_NORM_EPSILON=1e-5, LAYER_NORM_EPSILON=1e-5, TF_BATCH_NORM_EPSILON=0.001,
        register_model=lambda obj: obj,
        # These convenience attributes do not participate in the model graph.
        # Expose training-specific preprocessing separately with preprocess_rgb().
        add_pre_post_process=lambda *args, **kwargs: None,
        reload_model_weights=no_pretrained,
    )
    project = Path(keras_project)
    root = project/'keras_cv_attention_models'
    if not root.is_dir() and project.name == 'keras_cv_attention_models' and (project/'backend.py').is_file():
        root = project
    hashes = {}
    def load_definitions(relative_path, names):
        source_path = root/relative_path
        text = source_path.read_text(encoding='utf-8')
        tree = ast.parse(text, filename=str(source_path))
        selected, found = [], set()
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names:
                selected.append(node)
                found.add(node.name)
            elif isinstance(node, ast.Assign):
                matching = [t.id for t in node.targets if isinstance(t, ast.Name) and t.id in names]
                if matching:
                    selected.append(node)
                    found.update(matching)
        if set(names) != found:
            raise ValueError('Required original definitions missing from {}: {}'.format(source_path, sorted(set(names)-found)))
        module = ast.parse('')
        module.body = selected
        exec(compile(module, str(source_path), 'exec'), namespace)
        hashes[str(source_path)] = sha256(source_path)
    load_definitions('backend.py', ['align_input_shape_by_image_data_format'])
    backend.align_input_shape_by_image_data_format = namespace['align_input_shape_by_image_data_format']
    load_definitions('common_layers.py', ['activation_by_name', 'batchnorm_with_activation', 'conv2d_no_bias', 'depthwise_conv2d_no_bias', 'group_norm', 'layer_norm', 'make_divisible', 'drop_block', 'drop_connect_rates_split', 'output_block'])
    load_definitions('mlp_family/res_mlp.py', ['ChannelAffine'])
    load_definitions('convnext/convnext.py', ['add_with_layer_scale_and_drop_block'])
    load_definitions('mlp_family/mlp_mixer.py', ['mlp_block'])
    load_definitions('mobilevit/mobilevit.py', ['PRETRAINED_DICT', 'bottle_in_linear_out_block', 'linear_self_attention', 'linear_mhsa_mlp_block', 'transformer_pre_process', 'transformer_post_process', 'MobileViT'])
    load_definitions('mobilevit/mobilevit_v2.py', ['MobileViT_V2', 'get_mobilevit_v2_width', 'MobileViT_V2_075'])
    model = namespace['MobileViT_V2_075'](
        input_shape=(image_size, image_size, 3), num_classes=10,
        activation='relu', classifier_activation=None, pretrained=None)
    model.conversion_source_sha256 = hashes
    return model


def normalize_hdf5_string_attributes(path):
    """TF1.15 expects byte attributes; h5py 3 reads variable strings as str."""
    import h5py
    import numpy as np
    def normalize(group):
        for key in list(group.attrs.keys()):
            value = group.attrs[key]
            replacement = None
            if isinstance(value, str):
                replacement = np.bytes_(value.encode('utf-8'))
            elif isinstance(value, np.ndarray) and value.dtype.kind in ('O', 'U'):
                flat = value.reshape(-1).tolist()
                if all(isinstance(item, (str, bytes)) for item in flat):
                    replacement = np.asarray([item.encode('utf-8') if isinstance(item, str) else item for item in flat], dtype='S').reshape(value.shape)
            if replacement is not None:
                del group.attrs[key]
                group.attrs.create(key, replacement)
    with h5py.File(str(path), 'r+') as handle:
        normalize(handle)
        handle.visititems(lambda _name, group: normalize(group))


def load_tf_weights(weights_path, keras_project=DEFAULT_KECAM, image_size=256):
    """Load the converted HDF5 for inference in tf115 (no partial loading)."""
    model = build_tf_model(keras_project, image_size)
    model.load_weights(str(weights_path))
    return model

def compare_arrays(reference, actual, atol, rtol):
    import numpy as np
    if reference.shape != actual.shape:
        raise ValueError('Output shape mismatch: {} vs {}'.format(reference.shape, actual.shape))
    error = np.abs(reference-actual)
    return {
        'shape': list(reference.shape), 'max_abs': float(error.max()),
        'mean_abs': float(error.mean()),
        'allclose': bool(np.allclose(reference, actual, atol=atol, rtol=rtol)),
    }


def tensorflow_convert(config):
    # Frameworks are imported only in their designated child process.
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    os.environ['KECAM_BACKEND'] = 'tensorflow'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    import numpy as np
    import tensorflow as tf
    if not tf.__version__.startswith('1.15.'):
        raise RuntimeError('Expected TensorFlow 1.15.x in tf115; got '+tf.__version__)
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    def session():
        tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(
            intra_op_parallelism_threads=config['threads'], inter_op_parallelism_threads=config['threads'], device_count={'GPU': 0})))
    session()
    export = read_json(Path(config['run_dir'])/'torch_export.json')
    archive = np.load(str(Path(config['run_dir'])/'torch_export.npz'), allow_pickle=False)
    model = build_tf_model(config['keras_root'], config['image_size'])
    source_hashes = model.conversion_source_sha256
    actual_layer_names = {layer.name for layer in model.layers if layer.weights}
    expected_names = {record['target'] for record in export['mapped_layers']}
    if actual_layer_names != expected_names:
        raise ValueError('Target layer coverage mismatch. Missing={}, extra={}'.format(sorted(expected_names-actual_layer_names), sorted(actual_layer_names-expected_names)))
    mapping_report = []
    for record in export['mapped_layers']:
        layer = model.get_layer(record['target'])
        values = [archive[key] for key in record['weights']]
        if record['kind'] == 'conv':
            values[0] = values[0].transpose(2, 3, 1, 0)   # OIHW -> HWIO
        elif record['kind'] == 'depthwise':
            values[0] = values[0].transpose(2, 3, 0, 1)   # C,1,H,W -> H,W,C,1
        elif record['kind'] == 'dense':
            values[0] = values[0].T                     # OI -> IO
        expected_shapes = [tuple(weight.shape.as_list()) for weight in layer.weights]
        got_shapes = [value.shape for value in values]
        if expected_shapes != got_shapes:
            raise ValueError('Weight shape mismatch for {}: expected={}, got={}'.format(record['target'], expected_shapes, got_shapes))
        layer.set_weights(values)
        # Read back every array to ensure set_weights performed the exact transfer.
        if not all(np.array_equal(want, got) for want, got in zip(values, layer.get_weights())):
            raise ValueError('Weight read-back mismatch: '+record['target'])
        mapping_report.append(dict(record, tensorflow_shapes=[list(shape) for shape in expected_shapes]))
    feature_model = tf.keras.Model(model.inputs, [model.get_layer(r['target']).output for r in export['mapped_layers']])
    first_input = archive[export['cases'][0]['input']]
    feature_values = feature_model.predict(first_input, batch_size=1, verbose=0)
    layer_checks = []
    for index, (record, value) in enumerate(zip(export['mapped_layers'], feature_values)):
        if value.ndim == 4:
            value = value.transpose(0, 3, 1, 2)          # NHWC / BPNC -> NCHW / BCPN
        check = compare_arrays(archive['feature_{:03d}'.format(index)], value, config['atol'], config['rtol'])
        check.update(source=record['source'], target=record['target'])
        layer_checks.append(check)
    failures = [item for item in layer_checks if not item['allclose']]
    if failures:
        write_json(Path(config['run_dir'])/'failed_layer_checks.json', layer_checks)
        raise ValueError('Per-layer verification failed: '+repr(failures[:3]))
    checks = []
    for case in export['cases']:
        value = model.predict(archive[case['input']], batch_size=2, verbose=0)
        check = compare_arrays(archive[case['logits']], value, config['atol'], config['rtol'])
        check.update(case=case['name'], pytorch_top1=archive[case['logits']].argmax(-1).tolist(), tensorflow_top1=value.argmax(-1).tolist())
        if not check['allclose'] or check['pytorch_top1'] != check['tensorflow_top1']:
            raise ValueError('Logits verification failed: '+repr(check))
        checks.append(check)
    pending = Path(config['run_dir'])/'validated_weights.h5'
    model.save_weights(str(pending))
    normalize_hdf5_string_attributes(pending)
    parameter_count = model.count_params()
    tf.keras.backend.clear_session()
    session()
    fresh = build_tf_model(config['keras_root'], config['image_size'])
    # Full topology load; no skip_mismatch and no by_name partial loading.
    fresh.load_weights(str(pending))
    reload_checks = []
    for case in export['cases']:
        value = fresh.predict(archive[case['input']], batch_size=2, verbose=0)
        check = compare_arrays(archive[case['logits']], value, config['atol'], config['rtol'])
        check['case'] = case['name']
        if not check['allclose']:
            raise ValueError('HDF5 reload verification failed: '+repr(check))
        reload_checks.append(check)
    output = Path(config['output_dir'])
    output.mkdir(parents=True, exist_ok=True)
    weight_path = output/'mobilevitv2_075_10cls_relu_weights.h5'
    report_path = output/'conversion_report.json'
    if not config['overwrite'] and (weight_path.exists() or report_path.exists()):
        raise FileExistsError('Output already exists; use --overwrite or another --output-dir.')
    report = {
        'success': True, 'source_weights': export['source_weights'], 'source_sha256': export['source_sha256'],
        'weights_file': str(weight_path), 'weights_sha256': sha256(pending),
        'architecture': 'MobileViTv2-0.75', 'num_classes': 10, 'activation': 'relu',
        'patch': [2, 2], 'input_shape': [None, config['image_size'], config['image_size'], 3],
        'output': '10 logits, no softmax',
        'preprocessing': {'embedded_in_model': False, 'layout': 'NHWC', 'color': 'RGB', 'formula': '(RGB/255 - mean)/std', 'mean': MEAN, 'std': STD},
        'python_pytorch': export['python_torch'], 'torch_version': export['torch_version'],
        'python_tensorflow': sys.version, 'tensorflow_version': tf.__version__,
        'apple_block_class': export['apple_block_class'],
        'torch_trainable_parameters': export['learnable_parameters'],
        'tensorflow_total_parameters_including_bn_statistics': parameter_count,
        'source_state_entries': export['state_entries'], 'mapped_layer_count': len(mapping_report),
        'ignored_only_bn_counters': export['skipped_only'],
        'verification': {'atol': config['atol'], 'rtol': config['rtol'], 'per_layer_random_input': layer_checks, 'logits': checks, 'hdf5_reload': reload_checks},
        'mapping': mapping_report, 'keras_source_sha256': source_hashes,
        'compatibility': 'Original KECAM definitions with TF1.15 inference adapter; no repository edits. Use build_tf_model() for loading under tf115.',
    }
    # The final HDF5 is published only after all numerical and reload checks pass.
    os.replace(str(pending), str(weight_path))
    write_json(report_path, report)
    archive.close()
    print('[TensorFlow] PASS: all {} mapped layers; 3 logits cases; HDF5 reload.'.format(len(mapping_report)), flush=True)
    print('[TensorFlow] Maximum logits error: {:.9g}'.format(max(item['max_abs'] for item in checks)), flush=True)
    print('WEIGHTS: '+str(weight_path), flush=True)
    print('REPORT: '+str(report_path), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', default=DEFAULT_PT)
    parser.add_argument('--apple-root', default=DEFAULT_APPLE, help='ml-cvnets root containing cvnets/')
    parser.add_argument('--keras-root', default=DEFAULT_KECAM)
    parser.add_argument('--torch-python', default=str(DEFAULT_ENVS/'py312'/'python.exe'))
    parser.add_argument('--tf-python', default=str(DEFAULT_ENVS/'tf115'/'python.exe'))
    parser.add_argument('--output-dir', default=str(PROJECT/'src/tensorflow/conversion/records/converted_new'))
    parser.add_argument('--work-dir', default=str(PROJECT/'src/tensorflow/conversion/records/work'))
    parser.add_argument('--image-size', type=int, default=256)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--atol', type=float, default=1e-3, help='Absolute tolerance for FP32 cross-framework inference; actual errors are always reported.')
    parser.add_argument('--rtol', type=float, default=1e-4)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--stage', choices=['all', 'export', 'tensorflow'], default='all', help=argparse.SUPPRESS)
    parser.add_argument('--config', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.stage != 'all':
        config = read_json(args.config)
        if args.stage == 'export':
            export_torch(config)
        else:
            tensorflow_convert(config)
        return
    if args.image_size <= 0 or args.image_size % 64:
        parser.error('--image-size must be a positive multiple of 64. Other sizes trigger the known interpolation difference.')
    if args.threads <= 0 or args.atol < 0 or args.rtol < 0:
        parser.error('Invalid threads or tolerances.')
    for path in (args.weights, args.torch_python, args.tf_python):
        if not Path(path).is_file():
            parser.error('File not found: '+path)
    if not (Path(args.apple_root)/'cvnets'/'models'/'classification'/'mobilevit_v2.py').is_file():
        parser.error('--apple-root must contain cvnets/models/classification/mobilevit_v2.py')
    output_dir = Path(args.output_dir).resolve()
    if not args.overwrite and any((output_dir/name).exists() for name in ('mobilevitv2_075_10cls_relu_weights.h5', 'conversion_report.json')):
        parser.error('Output exists. Use another --output-dir or --overwrite.')
    run_dir = Path(args.work_dir).resolve()/('conversion-'+uuid.uuid4().hex[:12])
    run_dir.mkdir(parents=True, exist_ok=False)
    config = vars(args).copy()
    for key in ('weights', 'apple_root', 'keras_root', 'torch_python', 'tf_python'):
        config[key] = str(Path(config[key]).resolve())
    config.update(run_dir=str(run_dir), output_dir=str(output_dir))
    config_path = run_dir/'run_config.json'
    write_json(config_path, config)
    env = os.environ.copy()
    env.update(PYTHONDONTWRITEBYTECODE='1', PYTHONIOENCODING='utf-8', CUDA_VISIBLE_DEVICES='-1', TF_CPP_MIN_LOG_LEVEL='3')
    script = str(Path(__file__).resolve())
    print('Config: MobileViTv2-0.75 / 10 classes / ReLU / ImageNet normalization / {}x{}'.format(args.image_size, args.image_size), flush=True)
    print('Intermediate diagnostics: '+str(run_dir), flush=True)
    for stage, python in [('export', config['torch_python']), ('tensorflow', config['tf_python'])]:
        print('Running '+stage+' with '+python, flush=True)
        subprocess.run([python, '-B', '-u', script, '--stage', stage, '--config', str(config_path)], env=env, check=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        import traceback
        traceback.print_exc()
        print('CONVERSION FAILED: {}: {}'.format(type(error).__name__, error), file=sys.stderr)
        sys.exit(1)
