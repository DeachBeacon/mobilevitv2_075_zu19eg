from keras_cv_attention_models.mobilevit.mobilevit import MobileViT
from keras_cv_attention_models.models import register_model

import numpy as np


def MobileViT_V2(
    num_blocks=[1, 2, 3, 5, 4],
    out_channels=[64, 128, 256, 384, 512],
    attn_channels=0.5,  # Can be a list matching out_channels, or a float number for expansion ratio of out_channels
    expand_ratio=2,
    stem_width=32,
    resize_first=True,  # False for V1, True for V2
    use_depthwise=True,  # False for V1, True for V2
    use_fusion=False,  # True for V1, False for V2
    num_norm_groups=1,  # -1 or 0 for V1 using layer_norm, or 1 for V2 using group_norm
    use_linear_attention=True,  # False for V1, True for V2
    output_num_features=0,
    model_name="mobilevit_v2",
    **kwargs,
):
    kwargs.pop("kwargs", None)
    return MobileViT(**locals(), **kwargs)


def get_mobilevit_v2_width(multiplier=1.0):
    return int(32 * multiplier), [int(ii * multiplier) for ii in [64, 128, 256, 384, 512]]  # stem_width, out_channels


@register_model
def MobileViT_V2_050(input_shape=(256, 256, 3), num_classes=1000, activation="swish", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    stem_width, out_channels = get_mobilevit_v2_width(0.5)
    return MobileViT_V2(**locals(), model_name="mobilevit_v2_050", **kwargs)


@register_model
def MobileViT_V2_075(input_shape=(256, 256, 3), num_classes=1000, activation="swish", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    stem_width, out_channels = get_mobilevit_v2_width(0.75)
    return MobileViT_V2(**locals(), model_name="mobilevit_v2_075", **kwargs)


@register_model
def MobileViT_V2_100(input_shape=(256, 256, 3), num_classes=1000, activation="swish", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    return MobileViT_V2(**locals(), model_name="mobilevit_v2_100", **kwargs)


@register_model
def MobileViT_V2_125(input_shape=(256, 256, 3), num_classes=1000, activation="swish", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    stem_width, out_channels = get_mobilevit_v2_width(1.25)
    return MobileViT_V2(**locals(), model_name="mobilevit_v2_125", **kwargs)


@register_model
def MobileViT_V2_150(input_shape=(256, 256, 3), num_classes=1000, activation="swish", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    stem_width, out_channels = get_mobilevit_v2_width(1.5)
    return MobileViT_V2(**locals(), model_name="mobilevit_v2_150", **kwargs)


@register_model
def MobileViT_V2_175(input_shape=(256, 256, 3), num_classes=1000, activation="swish", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    stem_width, out_channels = get_mobilevit_v2_width(1.75)
    return MobileViT_V2(**locals(), model_name="mobilevit_v2_175", **kwargs)


@register_model
def MobileViT_V2_200(input_shape=(256, 256, 3), num_classes=1000, activation="swish", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    stem_width, out_channels = get_mobilevit_v2_width(2.0)
    return MobileViT_V2(**locals(), model_name="mobilevit_v2_200", **kwargs)


class MobileViT075RGBPreprocess:
    """Convert raw RGB uint8 images to the training pipeline's float32 input."""

    def __call__(self, images):
        images = np.asarray(images)
        if images.dtype != np.uint8:
            raise TypeError("preprocess_input expects raw RGB uint8 images; pass standardized float32 input directly to model.predict")
        if images.ndim == 3:
            images = images[None, ...]
        if images.ndim != 4 or tuple(images.shape[1:]) != (256, 256, 3):
            raise ValueError("Expected RGB image shape (256, 256, 3) or batch shape (B, 256, 256, 3)")
        images = images.astype(np.float32) / np.float32(255.0)
        mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32).reshape((1, 1, 1, 3))
        std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32).reshape((1, 1, 1, 3))
        return ((images - mean) / std).astype(np.float32)


def MobileViT_V2_075_TF115(weights_path):
    """Build the fixed 10-class ReLU model and strictly load its local legacy H5 weights."""
    model = MobileViT_V2_075(
        input_shape=(256, 256, 3),
        num_classes=10,
        activation="relu",
        classifier_activation=None,
        pretrained=None,
    )
    model.load_weights(weights_path, by_name=False)
    model.preprocess_input = MobileViT075RGBPreprocess()
    model.decode_predictions = None
    model.rescale_mode = "torch_imagenet_rgb"
    return model
