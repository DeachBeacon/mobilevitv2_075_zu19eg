"""Strictly load H5 weights and compare the public TF1.15 model entry with cvnets."""
import json
import hashlib
import os
from _paths import PROJECT, TF_ROOT, TF_RECORDS, TF_WEIGHTS, INFERENCE, DEPLOY, REFERENCE_PB
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import h5py
import numpy as np

from keras_cv_attention_models import backend, mobilevit
from keras_cv_attention_models.common_layers import group_norm


HERE = os.path.dirname(os.path.abspath(__file__))
ARTIFACT_DIR = str(TF_RECORDS)
REFERENCE = os.path.join(ARTIFACT_DIR, "pytorch_reference.npz")
WEIGHTS = str(TF_WEIGHTS)


def _decode(value):
    return value.decode("utf-8") if hasattr(value, "decode") else value


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def inspect_loaded_h5(model):
    tensor_count = 0
    scalar_count = 0
    weighted_layer_count = 0
    max_weight_error = 0.0
    with h5py.File(WEIGHTS, "r") as h5_file:
        layer_names = [_decode(value) for value in h5_file.attrs["layer_names"]]
        for layer_name in layer_names:
            group = h5_file[layer_name]
            weight_names = [_decode(value) for value in group.attrs.get("weight_names", [])]
            if not weight_names:
                continue
            loaded = model.get_layer(layer_name).get_weights()
            if len(loaded) != len(weight_names):
                raise AssertionError("{} has {} model tensors, H5 lists {}".format(layer_name, len(loaded), len(weight_names)))
            weighted_layer_count += 1
            for weight_name, actual in zip(weight_names, loaded):
                expected = np.asarray(group[weight_name])
                if actual.shape != expected.shape:
                    raise AssertionError("{} shape mismatch: {} vs {}".format(weight_name, actual.shape, expected.shape))
                if actual.dtype != np.float32 or not np.array_equal(actual, expected):
                    difference = float(np.max(np.abs(actual.astype(np.float64) - expected.astype(np.float64))))
                    max_weight_error = max(max_weight_error, difference)
                    raise AssertionError("{} was not loaded exactly (max abs error {})".format(weight_name, difference))
                tensor_count += 1
                scalar_count += int(actual.size)
    if tensor_count != 244 or scalar_count != 2496523:
        raise AssertionError("Unexpected loaded H5 totals: {} tensors, {} scalars".format(tensor_count, scalar_count))
    return {"weighted_layers": weighted_layer_count, "tensors": tensor_count, "scalars": scalar_count, "max_weight_error": max_weight_error}


def main():
    model = mobilevit.MobileViT_V2_075_TF115(WEIGHTS)
    if tuple(model.input_shape) != (None, 256, 256, 3) or tuple(model.output_shape) != (None, 10):
        raise AssertionError("Unexpected model shapes: {} -> {}".format(model.input_shape, model.output_shape))
    classifier_activation = getattr(model.get_layer("predictions").activation, "__name__", "")
    if classifier_activation != "linear":
        raise AssertionError("The classifier must return raw logits, activation is {}".format(classifier_activation))
    if model.decode_predictions is not None:
        raise AssertionError("The 10-class model must not use ImageNet decode_predictions")
    if "keras_cv_attention_models.imagenet.eval_func" in sys.modules:
        raise AssertionError("Building the target model unexpectedly imported ImageNet evaluation code")
    if "keras_cv_attention_models.attention_layers" in sys.modules:
        raise AssertionError("Building the target model unexpectedly imported the aggregate attention package")
    resize_input = backend.layers.Input(shape=(2, 2, 1))
    resize_output = backend.functional.resize(resize_input, [4, 4], method="bilinear", antialias=True)
    resize_model = backend.models.Model(resize_input, resize_output)
    resize_shape = tuple(resize_model.predict(np.ones((1, 2, 2, 1), dtype=np.float32), verbose=0).shape)
    if resize_shape != (1, 4, 4, 1):
        raise AssertionError("TF1 resize wrapper returned unexpected shape {}".format(resize_shape))
    expected_layer_shapes = {
        "stem_conv": (3, 3, 3, 24),
        "stack1_block1_deep_3_conv": (1, 1, 48, 48),
        "stack2_block2_deep_3_conv": (1, 1, 192, 96),
        "stack3_block3_post_1_conv": (1, 1, 96, 192),
        "stack4_block5_post_1_conv": (1, 1, 144, 288),
        "stack5_block4_post_1_conv": (1, 1, 192, 384),
        "stack3_block2_attn_mhsa_qkv_conv": (1, 1, 96, 193),
        "stack4_block2_attn_mhsa_qkv_conv": (1, 1, 144, 289),
        "stack5_block2_attn_mhsa_qkv_conv": (1, 1, 192, 385),
        "stack3_block2_mlp_Conv_0": (1, 1, 96, 192),
        "stack4_block2_mlp_Conv_0": (1, 1, 144, 288),
        "stack5_block2_mlp_Conv_0": (1, 1, 192, 384),
        "predictions": (384, 10),
    }
    for layer_name, expected_shape in expected_layer_shapes.items():
        layer = model.get_layer(layer_name)
        weight_shape = tuple(layer.get_weights()[0].shape)
        if weight_shape != expected_shape:
            raise AssertionError("{} has weight shape {}, expected {}".format(layer_name, weight_shape, expected_shape))
    attention_block_names = {
        3: ["stack3_block2", "stack3_block3"],
        4: ["stack4_block2", "stack4_block3", "stack4_block4", "stack4_block5"],
        5: ["stack5_block2", "stack5_block3", "stack5_block4"],
    }
    for stage, block_names in attention_block_names.items():
        for block_name in block_names:
            model.get_layer(block_name + "_attn_mhsa_qkv_conv")
    if sum(len(block_names) for block_names in attention_block_names.values()) != 9:
        raise AssertionError("Unexpected MobileViTv2 attention block counts")
    if model.get_layer("avg_pool").__class__.__name__ != "GlobalAveragePooling2D":
        raise AssertionError("The expected global average pooling layer is missing")
    batch_norm_layers = [layer for layer in model.layers if isinstance(layer, backend.layers.BatchNormalization)]
    if not batch_norm_layers or any(abs(float(layer.epsilon) - 1e-5) > 1e-12 for layer in batch_norm_layers):
        raise AssertionError("BatchNorm layers must use epsilon=1e-5")
    custom_objects = backend.tf.keras.utils.get_custom_objects()
    for key in ("CompatibleGroupNormalization", "kecamCommon>CompatibleGroupNormalization"):
        if key not in custom_objects:
            raise AssertionError("TF1 Keras custom object registry is missing {}".format(key))
    bn_state_before = {layer.name: [value.copy() for value in layer.get_weights()[2:4]] for layer in batch_norm_layers}

    weight_report = inspect_loaded_h5(model)
    if len(model.weights) != 244:
        raise AssertionError("Expected 244 model weights, found {}".format(len(model.weights)))

    with np.load(REFERENCE) as reference:
        inputs = reference["inputs_nhwc"].astype(np.float32)
        pytorch_logits = reference["logits"].astype(np.float32)
        raw_rgb = reference["preprocess_rgb_uint8"]
        expected_preprocessed = reference["preprocess_rgb_nchw"].transpose(0, 2, 3, 1).astype(np.float32)

    if inputs.shape != (3, 256, 256, 3) or not np.isfinite(inputs).all():
        raise AssertionError("Invalid reference inputs")
    if not np.allclose(model.preprocess_input(raw_rgb), expected_preprocessed, rtol=0.0, atol=1e-6):
        raise AssertionError("RGB preprocessing differs from torchvision ToTensor + ImageNet Normalize")
    try:
        model.preprocess_input(inputs[2])
    except TypeError:
        pass
    else:
        raise AssertionError("The raw RGB preprocessor accepted already-standardized float input")

    batch_logits = model.predict(inputs, batch_size=3, verbose=0)
    if batch_logits.dtype != np.float32 or batch_logits.shape != (3, 10) or not np.isfinite(batch_logits).all():
        raise AssertionError("Invalid inference output: dtype={}, shape={}".format(batch_logits.dtype, batch_logits.shape))
    sample_logits = np.concatenate([model.predict(inputs[ii : ii + 1], batch_size=1, verbose=0) for ii in range(len(inputs))], axis=0)
    for layer in batch_norm_layers:
        state_after = layer.get_weights()[2:4]
        if any(not np.array_equal(before, after) for before, after in zip(bn_state_before[layer.name], state_after)):
            raise AssertionError("BatchNorm moving statistics changed during model.predict for {}".format(layer.name))
    batch_consistency_error = float(np.max(np.abs(batch_logits - sample_logits)))
    if not np.allclose(batch_logits, sample_logits, rtol=1e-4, atol=2e-4):
        raise AssertionError("Single and multi-sample inference differ by {}".format(batch_consistency_error))
    if not np.array_equal(np.argmax(batch_logits, axis=1), np.argmax(sample_logits, axis=1)):
        raise AssertionError("Single and multi-sample inference changed a predicted class")

    # Isolate GroupNorm from convolution kernel batch-size rounding differences.
    gn_input = backend.layers.Input(shape=(2, 3, 4))
    gn_output = group_norm(gn_input, groups=1, name="batch_independent_gn_")
    gn_model = backend.models.Model(gn_input, gn_output)
    gn_values = np.random.RandomState(41).standard_normal((3, 2, 3, 4)).astype(np.float32)
    gn_batch = gn_model.predict(gn_values, batch_size=3, verbose=0)
    gn_singles = np.concatenate([gn_model.predict(gn_values[ii : ii + 1], batch_size=1, verbose=0) for ii in range(3)], axis=0)
    gn_consistency_error = float(np.max(np.abs(gn_batch - gn_singles)))
    gn_weights = gn_model.get_layer("batch_independent_gn_group_norm").get_weights()
    if len(gn_weights) != 2 or tuple(gn_weights[0].shape) != (4,) or tuple(gn_weights[1].shape) != (4,):
        raise AssertionError("GroupNorm gamma/beta shapes are not (C,) in gamma-then-beta order")
    if not np.allclose(gn_batch, gn_singles, rtol=0.0, atol=1e-6):
        raise AssertionError("GroupNorm statistics depend on other batch samples: {}".format(gn_consistency_error))

    names = ["zero", "standard_normal", "normalized_rgb"]
    comparisons = []
    for index, name in enumerate(names):
        difference = np.abs(batch_logits[index] - pytorch_logits[index])
        single_difference = np.abs(sample_logits[index] - pytorch_logits[index])
        passed = bool(np.allclose(batch_logits[index], pytorch_logits[index], rtol=1e-4, atol=1e-4))
        single_passed = bool(np.allclose(sample_logits[index], pytorch_logits[index], rtol=1e-4, atol=1e-4))
        comparison = {
            "input": name,
            "max_abs_error": float(np.max(difference)),
            "mean_abs_error": float(np.mean(difference)),
            "single_max_abs_error": float(np.max(single_difference)),
            "single_mean_abs_error": float(np.mean(single_difference)),
            "pytorch_class": int(np.argmax(pytorch_logits[index])),
            "tensorflow_class": int(np.argmax(batch_logits[index])),
            "allclose": passed,
            "single_allclose": single_passed,
        }
        comparisons.append(comparison)
        if not passed or not single_passed:
            raise AssertionError("TF/PT mismatch for {}: {}".format(name, comparison))

    report = {
        "tensorflow_version": backend.tf.__version__,
        "python_version": sys.version.split()[0],
        "h5_weights_sha256": sha256(WEIGHTS),
        "model_entry": "from keras_cv_attention_models import mobilevit",
        "imagenet_eval_imported": False,
        "aggregate_attention_imported": False,
        "classifier_activation": classifier_activation,
        "tf1_resize_antialias_probe_shape": list(resize_shape),
        "strict_h5_load": weight_report,
        "stem_and_attention_and_classifier_shapes": {name: list(shape) for name, shape in expected_layer_shapes.items()},
        "transformer_attention_block_counts_by_stage": {stage: len(names) for stage, names in attention_block_names.items()},
        "batch_norm_layers": len(batch_norm_layers),
        "batch_norm_moving_stats_unchanged_during_predict": True,
        "preprocessing": "raw RGB uint8 -> float32 / 255 -> ImageNet mean/std; standardized float32 is passed directly",
        "preprocessing_max_abs_error_vs_torchvision": float(np.max(np.abs(model.preprocess_input(raw_rgb) - expected_preprocessed))),
        "batch_consistency_max_abs_error": batch_consistency_error,
        "group_norm_batch_consistency_max_abs_error": gn_consistency_error,
        "comparisons": comparisons,
        "image_dataset_or_full_validation_set_tested": False,
    }
    report_path = os.path.join(ARTIFACT_DIR, "tf115_verification.json")
    with open(report_path, "w", encoding="utf-8") as output:
        json.dump(report, output, indent=2)
    print(json.dumps(report, indent=2))
    print("Saved report: {}".format(report_path))


if __name__ == "__main__":
    main()
