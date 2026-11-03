"""Build deterministic CPU reference inputs and logits from the training cvnets model."""
import argparse
import hashlib
import json
import os
from _paths import PROJECT, TF_ROOT, TF_RECORDS, TF_WEIGHTS, INFERENCE, DEPLOY, REFERENCE_PB
import sys

import numpy as np
import torch
from torchvision import transforms


WORKSPACE = str(TF_ROOT)
CVNETS_ROOT = str(PROJECT / "src/pytorch/model_source")
WEIGHTS = str(PROJECT / "data/weights/bestmodel_v1_only_weights.pt")
OUTPUT_DIR = str(TF_RECORDS)

sys.path.insert(0, CVNETS_ROOT)
from cvnets.models.classification.mobilevit_v2 import MobileViTv2


def build_opts(num_classes=10, width_multiplier=0.75, activation="relu"):
    """Mirror build_opts from the training notebook for direct MobileViTv2(opts)."""
    opts = argparse.Namespace()
    attrs = {
        "model.classification.n_classes": num_classes,
        "model.classification.gradient_checkpointing": False,
        "model.classification.enable_layer_wise_lr_decay": False,
        "model.classification.layer_wise_lr_decay_rate": 1.0,
        "model.classification.freeze_batch_norm": False,
        "model.classification.pretrained": None,
        "model.classification.finetune_pretrained_model": False,
        "model.classification.n_pretrained_classes": None,
        "model.classification.classifier_dropout": 0.1,
        "model.classification.activation.name": activation,
        "model.classification.activation.inplace": False,
        "model.classification.activation.neg_slope": 0.1,
        "model.activation.name": activation,
        "model.activation.inplace": False,
        "model.activation.neg_slope": 0.1,
        "model.normalization.name": "batch_norm",
        "model.normalization.momentum": 0.1,
        "model.normalization.eps": 1e-5,
        "model.normalization.groups": 32,
        "model.layer.conv_init": "kaiming_normal",
        "model.layer.linear_init": "trunc_normal",
        "model.layer.pretrained": None,
        "model.layer.global_pool": "mean",
        "model.classification.mitv2.width_multiplier": width_multiplier,
        "model.classification.mitv2.attn_dropout": 0.1,
        "model.classification.mitv2.ffn_dropout": 0.1,
        "model.classification.mitv2.dropout": 0.2,
        "model.classification.mitv2.attn_norm_layer": "layer_norm_2d",
        "model.layer.deit": None,
        "common.enable_coreml_compatible_module": False,
        "model.segmentation.n_classes": None,
        "common.regnet.quantize": False,
        "common.mixed_precision": False,
    }
    for key, value in attrs.items():
        setattr(opts, key, value)
    return opts


def make_inputs():
    rng = np.random.RandomState(20261026)
    zero = np.zeros((3, 256, 256), dtype=np.float32)
    standard_normal = rng.standard_normal((3, 256, 256)).astype(np.float32)
    rgb_float01 = rng.uniform(0.0, 1.0, size=(3, 256, 256)).astype(np.float32)
    mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
    std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]
    normalized_rgb = (rgb_float01 - mean) / std
    rgb_uint8 = rng.randint(0, 256, size=(256, 256, 3)).astype(np.uint8)
    to_tensor_normalize = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
    )
    preprocessed_rgb_nchw = to_tensor_normalize(rgb_uint8).numpy()[None, ...].astype(np.float32)
    return np.stack([zero, standard_normal, normalized_rgb], axis=0).astype(np.float32), rgb_uint8, preprocessed_rgb_nchw


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    torch.set_num_threads(1)

    model = MobileViTv2(build_opts())
    state = torch.load(WEIGHTS, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or not all(torch.is_tensor(value) for value in state.values()):
        raise TypeError("Expected the input file to contain a pure state_dict")
    load_result = model.load_state_dict(state, strict=True)
    if load_result.missing_keys or load_result.unexpected_keys:
        raise RuntimeError("Strict state_dict load returned missing or unexpected keys")
    model.eval()

    inputs_nchw, preprocess_rgb_uint8, preprocess_rgb_nchw = make_inputs()
    with torch.no_grad():
        logits = model(torch.from_numpy(inputs_nchw)).cpu().numpy().astype(np.float32)
    inputs_nhwc = inputs_nchw.transpose(0, 2, 3, 1).copy()

    npz_path = os.path.join(OUTPUT_DIR, "pytorch_reference.npz")
    np.savez_compressed(
        npz_path,
        inputs_nchw=inputs_nchw,
        inputs_nhwc=inputs_nhwc,
        logits=logits,
        class_ids=logits.argmax(axis=1),
        preprocess_rgb_uint8=preprocess_rgb_uint8,
        preprocess_rgb_nchw=preprocess_rgb_nchw,
    )
    metadata = {
        "weights_path": WEIGHTS,
        "weights_sha256": sha256(WEIGHTS),
        "model": "MobileViTv2 0.75, 10 classes, ReLU, eval mode, CPU",
        "python_version": sys.version.split()[0],
        "pytorch_version": torch.__version__,
        "state_dict_tensors": len(state),
        "input_seed": 20261026,
        "input_shapes": {"nchw": list(inputs_nchw.shape), "nhwc": list(inputs_nhwc.shape)},
        "logits_shape": list(logits.shape),
        "class_ids": logits.argmax(axis=1).tolist(),
        "reference_npz": npz_path,
        "normalization": "(RGB_float32_[0,1] - [0.485,0.456,0.406]) / [0.229,0.224,0.225]",
    }
    metadata_path = os.path.join(OUTPUT_DIR, "pytorch_reference.json")
    with open(metadata_path, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)
    print("Saved reference: {}".format(npz_path))
    print("Strictly loaded {} state_dict tensors".format(len(state)))
    print("PT logits shape: {}, class ids: {}".format(tuple(logits.shape), metadata["class_ids"]))


if __name__ == "__main__":
    main()
