"""Specialize integer shape subgraphs for batch-one deployment using TF 1.15."""
import collections
import copy
import json
import os
from _paths import PROJECT, TF_ROOT, TF_RECORDS, TF_WEIGHTS, INFERENCE, DEPLOY, REFERENCE_PB

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
import numpy as np
import tensorflow as tf
from inspect_mobilevitv2_pb import inspect_graph, sha256

HERE = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS = str(TF_RECORDS)
SOURCE = str(INFERENCE)
TARGET = str(DEPLOY / "mobilevitv2_075_10cls_relu_deploy_b1.pb")
REPORT = os.path.join(ARTIFACTS, 'deploy_b1_export_report.json')
IMAGES = os.environ.get('MOBILEVIT_TEST_DIR', str(PROJECT/'data/calibration/jpegimages'))
INPUT = 'input_1:0'
OUTPUT = 'predictions/BiasAdd:0'
INTEGER_OPS = {'Identity', 'StridedSlice', 'ConcatV2', 'Pack', 'Range',
               'Add', 'AddV2', 'Sub', 'Mul', 'FloorDiv', 'FloorMod',
               'GatherV2', 'Prod', 'Rank', 'Size'}


def load(path):
    result = tf.compat.v1.GraphDef()
    with tf.io.gfile.GFile(path, 'rb') as source:
        result.ParseFromString(source.read())
    return result


def import_graph(definition):
    result = tf.Graph()
    with result.as_default():
        tf.import_graph_def(definition, name='')
    return result


def config():
    return tf.compat.v1.ConfigProto(device_count={'GPU': 0}, allow_soft_placement=True)


def const_node(node, value, dtype):
    if not dtype.is_integer:
        raise AssertionError('Only integer constants may be introduced')
    result = tf.compat.v1.NodeDef(name=node.name, op='Const', device=node.device)
    result.attr['dtype'].type = dtype.as_datatype_enum
    result.attr['value'].tensor.CopyFrom(tf.make_tensor_proto(value, dtype=dtype))
    return result


def canonical(node):
    result = copy.deepcopy(node)
    if '_output_shapes' in result.attr:
        del result.attr['_output_shapes']
    return result


def main():
    if tuple(tf.__version__.split('.')[:2]) != ('1', '15'):
        raise RuntimeError('Use the specified TF1.15 interpreter')
    original = load(SOURCE)
    original_hash = sha256(SOURCE)
    if not inspect_graph(original)['pure_inference']:
        raise AssertionError('Source must be the pure-inference PB')
    old_graph = import_graph(original)
    concat_names = [n.name for n in original.node if n.op == 'ConcatV2']
    if len(concat_names) != 39:
        raise AssertionError('Expected exactly 39 ConcatV2 nodes')
    fixed = copy.deepcopy(original)
    for node in fixed.node:
        if node.name == 'input_1':
            node.attr['shape'].shape.CopyFrom(tf.TensorShape([1, 256, 256, 3]).as_proto())
        # Re-infer shapes from the fixed input instead of retaining batch=None caches.
        if '_output_shapes' in node.attr:
            del node.attr['_output_shapes']
    fixed_graph = import_graph(fixed)
    replacements = {}
    for node in fixed.node:
        if node.op == 'Shape':
            op = fixed_graph.get_operation_by_name(node.name)
            if not op.inputs[0].shape.is_fully_defined() or op.control_inputs:
                raise AssertionError('Shape is not provably static: ' + node.name)
            value = np.asarray(op.inputs[0].shape.as_list(), dtype=op.outputs[0].dtype.as_numpy_dtype)
            replacements[node.name] = const_node(node, value, op.outputs[0].dtype)
    for node in fixed.node:
        if node.name in replacements:
            node.CopyFrom(replacements[node.name])
    shape_graph = import_graph(fixed)
    proof_cache = {}

    def integer_constant_tree(tensor):
        # A closed whitelist and no data feeds prove input-value independence.
        if tensor.name in proof_cache:
            return proof_cache[tensor.name]
        op = tensor.op
        result = (tensor.dtype.is_integer and not op.control_inputs and
                  (op.type == 'Const' or (op.type in INTEGER_OPS and
                   all(integer_constant_tree(value) for value in op.inputs))))
        proof_cache[tensor.name] = result
        return result

    candidates = []
    for node in fixed.node:
        op = shape_graph.get_operation_by_name(node.name)
        if node.op != 'Const' and len(op.outputs) == 1 and integer_constant_tree(op.outputs[0]):
            candidates.append(op.outputs[0])
    with tf.compat.v1.Session(graph=shape_graph, config=config()) as session:
        values = session.run(candidates)  # No image feed: these are constant-only trees.
    for tensor, value in zip(candidates, values):
        node = next(n for n in fixed.node if n.name == tensor.op.name)
        replacements[node.name] = const_node(node, value, tensor.dtype)
    for node in fixed.node:
        if node.name in replacements:
            node.CopyFrom(replacements[node.name])
    if not all(name in replacements for name in concat_names):
        raise AssertionError('Not all 39 integer ConcatV2 nodes were proven constant')
    deployed = tf.compat.v1.graph_util.extract_sub_graph(fixed, [OUTPUT.split(':')[0]])
    new_graph = import_graph(deployed)
    new_nodes = {n.name: n for n in deployed.node}
    if new_graph.get_tensor_by_name(INPUT).shape.as_list() != [1, 256, 256, 3]:
        raise AssertionError('Deployment input is not fixed batch one')
    if not inspect_graph(deployed)['pure_inference'] or any(n.op == 'ConcatV2' for n in deployed.node):
        raise AssertionError('Deployment structural checks failed')
    # Every floating-point node (including ALL weights, GN and Softmax math)
    # must have equal NodeDefs (protobuf map serialization order is irrelevant),
    # apart from the input batch dimension and non-executable shape caches.
    float_nodes = [op.name for op in old_graph.get_operations()
                   if any(t.dtype.is_floating for t in op.outputs)]
    old_nodes = {n.name: n for n in original.node}
    for name in float_nodes:
        expected_node = copy.deepcopy(old_nodes[name])
        if name == 'input_1':
            expected_node.attr['shape'].shape.CopyFrom(tf.TensorShape([1, 256, 256, 3]).as_proto())
        if name not in new_nodes or canonical(expected_node) != canonical(new_nodes[name]):
            raise AssertionError('Floating math or weight changed: ' + name)
    concat_audit = []
    for name in concat_names:
        op = old_graph.get_operation_by_name(name)
        replacement = new_nodes[name]
        if not op.outputs[0].dtype.is_integer or replacement.op != 'Const':
            raise AssertionError('Concat was not replaced with integer Const: ' + name)
        concat_audit.append({
            'name': name, 'original_op': 'ConcatV2', 'dtype': op.outputs[0].dtype.name,
            'original_inputs': [t.name for t in op.inputs],
            'consumers': [consumer.name for consumer in op.outputs[0].consumers()],
            'category': 'group_norm_shape' if '_group_norm/' in name else 'softmax_transpose_permutation',
            'integer_value': tf.make_ndarray(replacement.attr['value'].tensor).tolist(),
        })
    print('Proved and replaced all 39 integer ConcatV2; floating math/weights unchanged.', flush=True)

    rng = np.random.RandomState(20261028)
    samples = [('zero', np.zeros((1, 256, 256, 3), dtype=np.float32), None)]
    for index in range(3):
        samples.append(('random_normal_{}'.format(index), rng.standard_normal((1, 256, 256, 3)).astype(np.float32), None))
    decode_graph = tf.Graph()
    with decode_graph.as_default():
        image_path = tf.compat.v1.placeholder(tf.string, [])
        decoded = tf.image.decode_image(tf.io.read_file(image_path), channels=3)
    with tf.compat.v1.Session(graph=decode_graph, config=config()) as session:
        for folder in sorted(os.listdir(IMAGES)):
            directory = os.path.join(IMAGES, folder)
            if not os.path.isdir(directory):
                continue
            paths = sorted(os.path.join(directory, f) for f in os.listdir(directory) if f.lower().endswith(('.jpg', '.jpeg', '.png')))
            if not paths:
                raise AssertionError('No real image in ' + directory)
            path = paths[0]
            rgb = session.run(decoded, {image_path: path})
            if rgb.shape != (256, 256, 3) or rgb.dtype != np.uint8:
                raise AssertionError('Unexpected image shape/type: ' + path)
            normalized = (rgb.astype(np.float32) / np.float32(255) - np.asarray([.485, .456, .406], np.float32)) / np.asarray([.229, .224, .225], np.float32)
            samples.append(('real_' + folder, normalized[None, ...], path))
    if len([s for s in samples if s[2]]) != 10:
        raise AssertionError('Expected one real image from each of ten class folders')
    # Serialize and reload BEFORE numerical validation, in independent TF sessions.
    temp = TARGET + '.tmp'
    with tf.io.gfile.GFile(temp, 'wb') as output:
        output.write(deployed.SerializeToString())
    loaded = load(temp)
    new_graph = import_graph(loaded)
    comparisons, old_logits, new_logits = [], [], []
    with tf.compat.v1.Session(graph=old_graph, config=config()) as before, tf.compat.v1.Session(graph=new_graph, config=config()) as after:
        concat_tensors = [old_graph.get_tensor_by_name(name + ':0') for name in concat_names]
        for label, inputs, path in samples:
            old_values = before.run([old_graph.get_tensor_by_name(OUTPUT)] + concat_tensors,
                                    {old_graph.get_tensor_by_name(INPUT): inputs})
            actual = after.run(new_graph.get_tensor_by_name(OUTPUT), {new_graph.get_tensor_by_name(INPUT): inputs})
            expected = old_values[0]
            for record, value in zip(concat_audit, old_values[1:]):
                if not np.array_equal(value, np.asarray(record['integer_value'], dtype=np.int32)):
                    raise AssertionError('Concat value mismatch: ' + record['name'])
            if actual.shape != (1, 10) or actual.dtype != np.float32 or not np.isfinite(actual).all():
                raise AssertionError('Invalid logits')
            passed = bool(np.allclose(actual, expected, rtol=1e-6, atol=1e-6))
            if not passed or not np.array_equal(np.argmax(actual, 1), np.argmax(expected, 1)):
                raise AssertionError('Logits mismatch: ' + label)
            import hashlib
            comparisons.append({'input': label, 'image_path': path,
                                'image_sha256': sha256(path) if path else None,
                                'normalized_input_sha256': hashlib.sha256(inputs.tobytes()).hexdigest(),
                                'max_abs_error': float(np.max(np.abs(actual - expected))),
                                'mean_abs_error': float(np.mean(np.abs(actual - expected))),
                                'allclose': passed, 'class_index': int(np.argmax(actual)),
                                'bitwise_equal': bool(np.array_equal(actual, expected))})
            old_logits.append(expected)
            new_logits.append(actual)
    if sha256(SOURCE) != original_hash:
        raise AssertionError('Original PB changed')
    os.replace(temp, TARGET)
    np.savez(os.path.join(ARTIFACTS, 'deploy_b1_validation_logits.npz'),
             original_logits=np.concatenate(old_logits), deployment_logits=np.concatenate(new_logits),
             sample_names=np.asarray([s[0] for s in samples]))
    report = {
        'tensorflow_version': tf.__version__, 'source_path': SOURCE, 'source_sha256': original_hash,
        'deployment_path': TARGET, 'deployment_sha256': sha256(TARGET),
        'input_tensor': INPUT, 'input_shape': [1, 256, 256, 3], 'output_tensor': OUTPUT,
        'output_shape': [1, 10], 'random_seed': 20261028, 'rtol': 1e-6, 'atol': 1e-6,
        'source_nodes': len(original.node), 'deployment_nodes': len(deployed.node),
        'concat_audit': concat_audit, 'concat_categories': dict(collections.Counter(r['category'] for r in concat_audit)),
        'folded_static_shape_nodes': sum(n.op == 'Shape' and n.name in replacements for n in original.node),
        'integer_replacements': [{'name': name, 'original_op': old_nodes[name].op,
                                  'dtype': tf.as_dtype(node.attr['dtype'].type).name,
                                  'value': tf.make_ndarray(node.attr['value'].tensor).tolist()}
                                 for name, node in sorted(replacements.items())],
        'all_float_node_definitions_equal_except_input_batch_and_shape_metadata': True,
        'preserved_float_node_count': len(float_nodes),
        'remaining_integer_nonconstant_ops': dict(collections.Counter(op.type for op in new_graph.get_operations()
            if op.type != 'Const' and any(t.dtype.is_integer for t in op.outputs))),
        'structure': inspect_graph(deployed), 'comparisons': comparisons,
        'vitis_ai_quantizer_executed': False,
    }
    with open(REPORT, 'w', encoding='utf-8') as output:
        json.dump(report, output, indent=2)
    print('Saved: ' + TARGET)
    print('Validated {} inputs. Max logits error: {}'.format(len(samples), max(r['max_abs_error'] for r in comparisons)))
    print('Report: ' + REPORT)


if __name__ == '__main__':
    main()
