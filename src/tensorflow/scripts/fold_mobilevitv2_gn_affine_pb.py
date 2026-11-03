"""Audit/fold the 21 GN channel affines into deployment Conv2D (TF 1.15).

Only a deployment copy is changed. Dynamic GN statistics stay in the graph.
The serialized candidate must pass structural, local-fold and logits checks
before it is published. This script does not update external quantizer files.
"""
import argparse
import collections
import copy
import hashlib
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
SOURCE = str(DEPLOY / "mobilevitv2_075_10cls_relu_deploy_b1.pb")
TARGET = str(DEPLOY / "mobilevitv2_075_10cls_relu_deploy_b1_gnfold.pb")
IMAGES = os.environ.get('MOBILEVIT_TEST_DIR', str(PROJECT/'data/calibration/jpegimages'))
INPUT, OUTPUT = 'input_1:0', 'predictions/BiasAdd:0'
RTOL, ATOL, SEED = 1e-4, 1e-4, 20261028


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def load(path):
    result = tf.GraphDef()
    with open(path, 'rb') as source:
        result.ParseFromString(source.read())
    return result


def import_graph(definition):
    result = tf.Graph()
    with result.as_default():
        tf.import_graph_def(definition, name='')
    return result


def config():
    return tf.ConfigProto(device_count={'GPU': 0}, allow_soft_placement=True)


def array_hash(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest().upper()


def producer(reference):
    return reference.lstrip('^').split(':')[0]


def same_attributes(left, right):
    a, b = copy.deepcopy(left), copy.deepcopy(right)
    del a.input[:]
    del b.input[:]
    return a == b


def fold(original):
    graph = import_graph(original)
    require(graph.get_tensor_by_name(INPUT).shape.as_list() == [1, 256, 256, 3],
            'Expected fixed deployment input [1,256,256,3]')
    require(graph.get_tensor_by_name(OUTPUT).shape.as_list() == [1, 10], 'Expected logits [1,10]')
    require(inspect_graph(original)['pure_inference'], 'Source is not pure inference')
    require(not original.library.function, 'Function bodies need a separate audit')
    source_nodes = {n.name: n for n in original.node}
    require(len(source_nodes) == len(original.node), 'Duplicate node name')
    result = copy.deepcopy(original)
    nodes = {n.name: n for n in result.node}
    control_users = collections.defaultdict(list)
    for node in original.node:
        for reference in node.input:
            if reference.startswith('^'):
                control_users[producer(reference)].append(node.name)

    def no_control(op):
        require(not op.control_inputs and not control_users[op.name],
                'Control dependency on folding path: ' + op.name)

    def sole_consumer(tensor):
        no_control(tensor.op)
        users = tensor.consumers()
        require(len(users) == 1, 'Unhandled branch at {}: {}'.format(
            tensor.name, [op.name for op in users]))
        no_control(users[0])
        require(sum(t is tensor for t in users[0].inputs) == 1,
                'Repeated data edge: ' + tensor.name)
        return users[0]

    def const(tensor, owner=None):
        # Frozen variable reads may be Identity -> Const. If an owner is given,
        # prove exclusive use before changing/removing the entire private chain.
        chain = []
        while True:
            op = tensor.op
            no_control(op)
            require(len(op.outputs) == 1, 'Multiple constant outputs: ' + op.name)
            if owner is not None:
                require(sole_consumer(tensor).name == owner.name,
                        'Shared parameter: ' + tensor.name)
            chain.append(op.name)
            if op.type == 'Const':
                return tf.make_ndarray(op.get_attr('value')), op.name, chain
            require(op.type == 'Identity' and len(op.inputs) == 1,
                    'Not an immutable constant chain: ' + op.name)
            owner = op if owner is not None else None
            tensor = op.inputs[0]

    prefixes = [op.name[:-len('/Rsqrt')] for op in graph.get_operations()
                if op.name.endswith('_group_norm/Rsqrt')]
    require(len(prefixes) == 21, 'Expected exactly 21 GroupNorms')
    require({n + '/add_1' for n in prefixes} == {
        op.name for op in graph.get_operations() if op.name.endswith('_group_norm/add_1')},
        'GroupNorm affine inventory mismatch')
    audits, removed, changes, added = [], set(), set(), set()

    def set_float_const(name, value):
        require(value.dtype == np.float32 and np.isfinite(value).all(),
                'Invalid folded float32 constant: ' + name)
        require(nodes[name].op == 'Const', 'Expected Const: ' + name)
        nodes[name].attr['value'].tensor.CopyFrom(tf.make_tensor_proto(value, dtype=tf.float32))
        changes.add(name)

    for prefix in prefixes:
        affine_mul = graph.get_operation_by_name(prefix + '/mul_1')
        affine_add = graph.get_operation_by_name(prefix + '/add_1')
        normalized = graph.get_tensor_by_name(prefix + '/Reshape_1:0')
        require(affine_mul.type == 'Mul' and affine_add.type in ('Add', 'AddV2'),
                'Unexpected affine ops: ' + prefix)
        require(len(affine_mul.inputs) == len(affine_add.inputs) == 2,
                'Unexpected affine arity: ' + prefix)
        require(affine_mul.inputs[0] is normalized and
                affine_add.inputs[0] is affine_mul.outputs[0], 'Unexpected affine wiring: ' + prefix)
        require(sole_consumer(normalized) is affine_mul and
                sole_consumer(affine_mul.outputs[0]) is affine_add, 'Unhandled affine branch')
        gamma, gamma_const, gamma_chain = const(affine_mul.inputs[1], affine_mul)
        beta, beta_const, beta_chain = const(affine_add.inputs[1], affine_add)
        channels = normalized.shape.as_list()[-1]
        require(gamma.shape == beta.shape == (channels,) and
                gamma.dtype == beta.dtype == np.float32 and
                np.isfinite(gamma).all() and np.isfinite(beta).all(),
                'Expected finite per-channel float32 gamma/beta: ' + prefix)

        # Exhaustive symbolic channel labels, one for EVERY tensor element.
        # Reshape may temporarily pack channels together with spatial elements;
        # only the final Conv2D input must have channel label exactly i at [:,:,:,i].
        labels = np.broadcast_to(np.arange(channels, dtype=np.int32),
                                 normalized.shape.as_list()).copy()
        path, current = [], affine_add.outputs[0]
        first_layout = None
        while True:
            op = sole_consumer(current)
            require(op.inputs[0] is current, 'Data is not the first path input: ' + op.name)
            if op.type == 'Conv2D':
                conv = op
                break
            require(op.type in ('Identity', 'Reshape', 'Transpose'),
                    'Nonlinear/unsupported op before Conv2D: ' + op.name + ' ' + op.type)
            require(len(op.outputs) == 1, 'Multiple path outputs: ' + op.name)
            record = {'name': op.name, 'op': op.type, 'input_tensor': current.name,
                      'input_shape': current.shape.as_list(),
                      'output_shape': op.outputs[0].shape.as_list(),
                      'data_consumers': [v.name for v in current.consumers()]}
            if op.type != 'Identity':
                require(len(op.inputs) == 2 and op.inputs[1].dtype.is_integer,
                        'Expected integer layout parameter: ' + op.name)
                parameter, _, _ = const(op.inputs[1])
                record['integer_parameter'] = parameter.tolist()
                if op.type == 'Reshape':
                    labels = np.reshape(labels, parameter.tolist())
                else:
                    require(sorted(parameter.tolist()) == list(range(labels.ndim)),
                            'Transpose is not a permutation: ' + op.name)
                    labels = np.transpose(labels, parameter.tolist())
            else:
                require(len(op.inputs) == 1, 'Unexpected Identity inputs')
            require(list(labels.shape) == op.outputs[0].shape.as_list(),
                    'Layout proof shape mismatch: ' + op.name)
            record['channel_labels_sha256'] = array_hash(labels)
            path.append(record)
            if first_layout is None:
                first_layout = op
            current = op.outputs[0]
        no_control(conv)
        require(conv.get_attr('data_format') == b'NHWC' and
                conv.get_attr('padding') == b'VALID' and
                list(conv.get_attr('strides')) == [1, 1, 1, 1] and
                list(conv.get_attr('dilations')) == [1, 1, 1, 1] and
                not list(conv.get_attr('explicit_paddings')),
                'Expected VALID NHWC unit-stride/dilation Conv2D: ' + conv.name)
        require(labels.ndim == 4 and labels.shape[-1] == channels,
                'Conv input channels changed: ' + conv.name)
        expected_labels = np.broadcast_to(np.arange(channels, dtype=np.int32), labels.shape)
        mismatches = int(np.count_nonzero(labels != expected_labels))
        require(mismatches == 0, 'Channel permutation/mixing before ' + conv.name)
        weight, weight_const, _ = const(conv.inputs[1], conv)
        require(weight.ndim == 4 and weight.shape[:3] == (1, 1, channels) and
                weight.dtype == np.float32, 'Expected [1,1,Cin,Cout] float32 kernel')
        out_channels = weight.shape[-1]
        conv_user = sole_consumer(conv.outputs[0])
        bias_added = conv_user.type != 'BiasAdd'
        if not bias_added:
            require(conv_user.inputs[0] is conv.outputs[0] and
                    conv_user.get_attr('data_format') == b'NHWC', 'Unexpected BiasAdd')
            bias, bias_const, _ = const(conv_user.inputs[1], conv_user)
            require(bias.shape == (out_channels,) and bias.dtype == np.float32,
                    'Unexpected original bias')
            endpoint = conv_user.outputs[0].name
        else:
            require(conv_user.type.startswith('FusedBatchNorm') and
                    not conv_user.get_attr('is_training') and
                    conv_user.inputs[0] is conv.outputs[0],
                    'Expected inference BN after bias-free post convolution')
            bias = np.zeros(out_channels, np.float32)
            bias_const = conv.name.rsplit('/', 1)[0] + '/gn_folded_bias'
            bias_add_name = conv.name.rsplit('/', 1)[0] + '/gn_folded_BiasAdd'
            require(bias_const not in nodes and bias_add_name not in nodes, 'Name collision')
            bias_node = tf.NodeDef(name=bias_const, op='Const', device=nodes[conv.name].device)
            bias_node.attr['dtype'].type = tf.float32.as_datatype_enum
            bias_add = tf.NodeDef(name=bias_add_name, op='BiasAdd', device=nodes[conv.name].device)
            bias_add.input.extend([conv.outputs[0].name, bias_const + ':0'])
            bias_add.attr['T'].type = tf.float32.as_datatype_enum
            bias_add.attr['data_format'].s = b'NHWC'
            for new in (bias_node, bias_add):
                result.node.add().CopyFrom(new)
                nodes[new.name] = result.node[-1]
                added.add(new.name)
            nodes[conv_user.name].input[0] = bias_add_name + ':0'
            changes.add(conv_user.name)
            endpoint = bias_add_name + ':0'

        # Use ORIGINAL W in beta's contraction. Accumulate in float64, then
        # round the mathematically equivalent deployment constants to float32.
        original_weight = weight.astype(np.float64)
        new_weight = (original_weight * gamma.astype(np.float64)[None, None, :, None]).astype(np.float32)
        bias_delta = np.sum(beta.astype(np.float64)[:, None] * original_weight[0, 0], axis=0)
        new_bias = (bias.astype(np.float64) + bias_delta).astype(np.float32)
        set_float_const(weight_const, new_weight)
        set_float_const(bias_const, new_bias)
        rewire = first_layout if first_layout is not None else conv
        nodes[rewire.name].input[0] = normalized.name
        changes.add(rewire.name)
        removed.update([affine_mul.name, affine_add.name] + gamma_chain + beta_chain)

        # Record the preserved dynamic-statistics contract explicitly.
        mean = graph.get_operation_by_name(prefix + '/moments/mean')
        variance = graph.get_operation_by_name(prefix + '/moments/variance')
        axes_mean, _, _ = const(mean.inputs[1])
        axes_variance, _, _ = const(variance.inputs[1])
        epsilon, _, _ = const(graph.get_operation_by_name(prefix + '/add').inputs[1])
        require(mean.type == variance.type == 'Mean' and
                mean.get_attr('keep_dims') and variance.get_attr('keep_dims') and
                axes_mean.tolist() == axes_variance.tolist() == [1, 3] and
                mean.inputs[0].shape.as_list()[2] == 1 and
                epsilon.shape == () and epsilon == np.float32(1e-5),
                'Unexpected GroupNorm statistics: ' + prefix)
        audit = {
            'group_norm': prefix, 'gamma_const': gamma_const, 'beta_const': beta_const,
            'channels': channels, 'normalized_tensor': normalized.name,
            'affine_nodes_removed': [affine_mul.name, affine_add.name],
            'path': path, 'conv': conv.name, 'kernel_const': weight_const,
            'kernel_shape': list(weight.shape), 'kernel_layout': 'HWIO',
            'bias_const': bias_const, 'bias_added': bias_added,
            'original_endpoint': conv_user.outputs[0].name if not bias_added else conv.outputs[0].name,
            'folded_endpoint': endpoint,
            'original_kernel_sha256': array_hash(weight), 'folded_kernel_sha256': array_hash(new_weight),
            'original_bias_sha256': array_hash(bias), 'folded_bias_sha256': array_hash(new_bias),
            'gamma_sha256': array_hash(gamma), 'beta_sha256': array_hash(beta),
            'bias_formula_uses_original_kernel': True, 'constant_accumulation_dtype': 'float64',
            'constant_storage_dtype': 'float32', 'checked_channel_elements': int(labels.size),
            'final_channel_label_mismatches': mismatches,
            'channel_mapping': 'original channel i -> Conv2D input channel i for every element',
            'no_nonlinearity_or_unhandled_branch': True,
            'statistics': {'mean_axes_in_grouped_tensor': axes_mean.tolist(),
                           'variance_axes_in_grouped_tensor': axes_variance.tolist(),
                           'grouped_shape': mean.inputs[0].shape.as_list(),
                           'effective_input_axes': [1, 2, 3], 'groups': 1,
                           'epsilon': float(epsilon), 'dynamic_mean_variance_preserved': True},
        }
        audits.append(audit)
        print('Audited/folded {} -> {}{}'.format(prefix, conv.name,
              ' (+ BiasAdd)' if bias_added else ''), flush=True)

    require(len({a['conv'] for a in audits}) == 21, 'Multiple GN folds target the same conv')
    require(sum(a['bias_added'] for a in audits) == 3, 'Expected 18 existing and 3 new biases')
    # Delete only the explicitly audited affine and private parameter nodes.
    retained = [copy.deepcopy(n) for n in result.node if n.name not in removed]
    del result.node[:]
    result.node.extend(retained)
    new_nodes = {n.name: n for n in result.node}
    require(removed == set(source_nodes) - set(new_nodes), 'Unexpected node deletion')
    require(added == set(new_nodes) - set(source_nodes), 'Unexpected node addition')
    unchanged = set(source_nodes) - removed - changes
    for name in unchanged:
        require(source_nodes[name] == new_nodes[name], 'Unexpected change: ' + name)
    # Rewiring BN data through the new bias is the sole change to these 3 BN
    # NodeDefs. All 25 BN attributes/parameters and the other BN edges stay exact.
    for name, node in source_nodes.items():
        if node.op.startswith('FusedBatchNorm'):
            require(same_attributes(node, new_nodes[name]) and
                    list(node.input[1:]) == list(new_nodes[name].input[1:]),
                    'BatchNorm attributes/parameters changed: ' + name)
    integer_constants = [n.name for n in original.node if n.op == 'Const' and
                         tf.as_dtype(n.attr['dtype'].type).is_integer]
    require(all(source_nodes[n] == new_nodes[n] for n in integer_constants),
            'Existing integer shape/permutation constants changed')
    for prefix in prefixes:
        for name, node in source_nodes.items():
            if name.startswith(prefix + '/') and name not in removed:
                require(node == new_nodes[name], 'Dynamic GroupNorm changed: ' + name)
    require(result.library == original.library and result.versions == original.versions,
            'Graph metadata changed')
    imported = import_graph(result)  # also reject dangling edges/type/shape errors
    require(inspect_graph(result)['pure_inference'], 'Folded graph is not pure inference')
    require(all(op.type == 'Const' or not any(t.dtype.is_integer for t in op.outputs)
                for op in imported.get_operations()), 'Nonconstant integer shape op introduced')
    require(not any(n.name.endswith(('_group_norm/mul_1', '_group_norm/add_1'))
                    for n in result.node), 'Independent GN affine remains')
    require(imported.get_tensor_by_name(INPUT).shape.as_list() == [1, 256, 256, 3] and
            imported.get_tensor_by_name(OUTPUT).shape.as_list() == [1, 10], 'I/O changed')
    preservation = {
        'removed_nodes': sorted(removed), 'added_nodes': sorted(added),
        'changed_existing_nodes': sorted(changes - added),
        'unchanged_existing_node_count': len(unchanged),
        'all_other_node_definitions_exactly_equal': True,
        'integer_const_count_preserved_exactly': len(integer_constants),
        'all_21_dynamic_group_norm_cores_exactly_equal': True,
        'all_9_softmax_node_definitions_exactly_equal': all(
            n == new_nodes[n.name] for n in original.node if n.op == 'Softmax'),
        'all_25_batch_norm_attributes_and_parameters_exactly_equal': True,
        'batch_norm_data_edges_redirected_through_new_bias': 3,
    }
    return result, audits, preservation


def samples(image_root):
    yield 'zero', np.zeros((1, 256, 256, 3), np.float32), None
    rng = np.random.RandomState(SEED)
    for index in range(3):
        yield 'random_normal_{}'.format(index), rng.standard_normal((1, 256, 256, 3)).astype(np.float32), None
    folders = sorted(p for p in os.listdir(image_root) if os.path.isdir(os.path.join(image_root, p)))
    require(len(folders) == 10, 'Expected 10 real-image class directories')
    decoder = tf.Graph()
    with decoder.as_default():
        filename = tf.placeholder(tf.string, [])
        rgb_tensor = tf.image.decode_image(tf.read_file(filename), channels=3)
    with tf.Session(graph=decoder, config=config()) as session:
        for folder in folders:
            directory = os.path.join(image_root, folder)
            files = sorted(os.path.join(directory, f) for f in os.listdir(directory)
                           if f.lower().endswith(('.jpg', '.jpeg', '.png')))
            require(files, 'No real image in ' + directory)
            path = files[0]
            rgb = session.run(rgb_tensor, {filename: path})
            require(rgb.shape == (256, 256, 3) and rgb.dtype == np.uint8,
                    'Expected original 256x256 RGB uint8: ' + path)
            value = (rgb.astype(np.float32) / np.float32(255) -
                     np.asarray([.485, .456, .406], np.float32)) / np.asarray([.229, .224, .225], np.float32)
            yield 'real_' + folder, value[None, ...], path


def errors(expected, actual):
    require(expected.shape == actual.shape and expected.dtype == actual.dtype == np.float32 and
            np.isfinite(expected).all() and np.isfinite(actual).all(), 'Invalid comparison tensors')
    delta = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    tolerance = ATOL + RTOL * np.abs(expected.astype(np.float64))
    return {'max_abs_error': float(delta.max()), 'mean_abs_error': float(delta.mean()),
            'max_error_over_tolerance': float((delta / tolerance).max()),
            'allclose': bool(np.allclose(actual, expected, rtol=RTOL, atol=ATOL)),
            'bitwise_equal': bool(np.array_equal(actual, expected))}


def validate(original, deployed, audits, image_root):
    before_graph, after_graph = import_graph(original), import_graph(deployed)
    before_input, after_input = [g.get_tensor_by_name(INPUT) for g in (before_graph, after_graph)]
    original_fetches = [before_graph.get_tensor_by_name(OUTPUT)]
    original_fetches += [before_graph.get_tensor_by_name(a['normalized_tensor']) for a in audits]
    original_fetches += [before_graph.get_tensor_by_name(a['original_endpoint']) for a in audits]
    local_inputs = [after_graph.get_tensor_by_name(a['normalized_tensor']) for a in audits]
    local_outputs = [after_graph.get_tensor_by_name(a['folded_endpoint']) for a in audits]
    for a in audits:
        a['local_equivalence_checks'] = []
    comparisons, original_logits, folded_logits = [], [], []
    with tf.Session(graph=before_graph, config=config()) as before, tf.Session(graph=after_graph, config=config()) as after:
        for label, value, path in samples(image_root):
            fetched = before.run(original_fetches, {before_input: value})
            expected = fetched[0]
            actual = after.run(after_graph.get_tensor_by_name(OUTPUT), {after_input: value})
            require(actual.shape == (1, 10), 'Unexpected logits shape')
            record = errors(expected, actual)
            record.update({'input': label, 'image_path': path,
                           'image_sha256': sha256(path) if path else None,
                           'normalized_input_sha256': array_hash(value),
                           'original_class_index': int(np.argmax(expected, axis=1)[0]),
                           'folded_class_index': int(np.argmax(actual, axis=1)[0]),
                           'class_equal': bool(np.array_equal(np.argmax(expected, axis=1), np.argmax(actual, axis=1))),
                           'original_logits': expected.tolist(), 'folded_logits': actual.tolist()})
            require(record['allclose'] and record['class_equal'], 'Logits validation failed: ' + label + ' ' + str(record))
            # Independently test EVERY fold with the SAME pre-affine normalized
            # input from the old graph. This excludes accumulated upstream error.
            local_actual = after.run(local_outputs, dict(zip(local_inputs, fetched[1:22])))
            for a, old, new in zip(audits, fetched[22:], local_actual):
                local = errors(old, new)
                local['input'] = label
                require(local['allclose'], 'Local fold validation failed: ' + a['group_norm'] + ' ' + str(local))
                a['local_equivalence_checks'].append(local)
            comparisons.append(record)
            original_logits.append(expected)
            folded_logits.append(actual)
            print('Validated {}: max_abs={:.9g}, mean_abs={:.9g}, class={}/{}'.format(
                label, record['max_abs_error'], record['mean_abs_error'],
                record['original_class_index'], record['folded_class_index']), flush=True)
    require(len(comparisons) == 14 and all(len(a['local_equivalence_checks']) == 14 for a in audits),
            'Incomplete numerical validation')
    return comparisons, np.concatenate(original_logits), np.concatenate(folded_logits)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', default=SOURCE)
    parser.add_argument('--target', default=TARGET)
    parser.add_argument('--images', default=IMAGES)
    parser.add_argument('--report-dir', default=ARTIFACTS, help='Directory for fold report and validation logits')
    args = parser.parse_args()
    require(tuple(tf.__version__.split('.')[:2]) == ('1', '15'), 'Use the existing TF1.15 interpreter')
    source, target = os.path.abspath(args.source), os.path.abspath(args.target)
    require(os.path.normcase(source) != os.path.normcase(target), 'Never overwrite original PB')
    require(not os.path.exists(target), 'Target already exists; select a fresh --target')
    original = load(source)
    source_hash = sha256(source)
    baseline_paths = [str(REFERENCE_PB), str(INFERENCE)]
    baseline_hashes = {p: sha256(p) for p in baseline_paths if os.path.exists(p)}
    deployed, audits, preservation = fold(original)
    temp = target + '.validating.tmp'
    require(not os.path.exists(temp), 'Validation temporary file already exists')
    with open(temp, 'wb') as output:
        output.write(deployed.SerializeToString())
    reloaded = load(temp)
    require(reloaded == deployed, 'Serialized/reloaded graph differs')
    comparisons, old_logits, new_logits = validate(original, reloaded, audits, args.images)
    require(sha256(source) == source_hash, 'Source PB changed')
    require(all(sha256(p) == h for p, h in baseline_hashes.items()), 'An original PB changed')
    os.makedirs(args.report_dir, exist_ok=True)
    target_stem = os.path.splitext(os.path.basename(target))[0]
    report_path = os.path.join(args.report_dir, target_stem + '_report.json')
    logits_path = os.path.join(args.report_dir, target_stem + '_validation_logits.npz')
    require(not os.path.exists(report_path) and not os.path.exists(logits_path), 'Output report/logits already exist')
    report = {
        'validation_passed': True, 'tensorflow_version': tf.__version__,
        'source_path': source, 'source_sha256': source_hash,
        'deployment_path': target, 'deployment_sha256': sha256(temp),
        'source_nodes': len(original.node), 'deployment_nodes': len(reloaded.node),
        'source_bytes': os.path.getsize(source), 'deployment_bytes': os.path.getsize(temp),
        'input_tensor': INPUT, 'input_shape': [1, 256, 256, 3],
        'output_tensor': OUTPUT, 'output_shape': [1, 10],
        'fold_count': len(audits), 'new_bias_add_count': sum(a['bias_added'] for a in audits),
        'weight_formula': "W'[0,0,i,o] = gamma[i] * original_W[0,0,i,o]",
        'bias_formula': "b'[o] = original_b[o] + sum_i(beta[i] * original_W[0,0,i,o])",
        'random_seed': SEED, 'rtol': RTOL, 'atol': ATOL,
        'preprocessing': 'RGB uint8 / 255; (x - [0.485,0.456,0.406]) / [0.229,0.224,0.225]; no resize',
        'fold_audit': audits, 'preservation': preservation, 'comparisons': comparisons,
        'max_logits_abs_error': max(r['max_abs_error'] for r in comparisons),
        'mean_logits_abs_error': float(np.mean([r['mean_abs_error'] for r in comparisons])),
        'all_classes_equal': all(r['class_equal'] for r in comparisons),
        'local_fold_checks': sum(len(a['local_equivalence_checks']) for a in audits),
        'original_pb_hashes_preserved': dict(baseline_hashes, **{source: source_hash}),
        'structure': inspect_graph(reloaded), 'training_performed': False,
        'vitis_ai_quantization_executed': False, 'full_dataset_accuracy_evaluated': False,
        'external_quantization_script_updated_by_exporter': False,
    }
    # Only validated bytes become the final deployment artifact.
    os.replace(temp, target)
    np.savez(logits_path, original_logits=old_logits, folded_logits=new_logits,
             sample_names=np.asarray([r['input'] for r in comparisons]))
    with open(report_path, 'w', encoding='utf-8') as output:
        json.dump(report, output, indent=2)
    print('Saved: ' + target)
    print('Report: ' + report_path)
    print('14/14 logits and 294/294 local-fold checks passed; maximum logits error: {}'.format(
        report['max_logits_abs_error']))


if __name__ == '__main__':
    main()
