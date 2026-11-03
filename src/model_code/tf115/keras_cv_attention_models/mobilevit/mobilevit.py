from keras_cv_attention_models import backend
from keras_cv_attention_models.backend import layers, models, functional, image_data_format
from keras_cv_attention_models.models import register_model
from keras_cv_attention_models.common_layers import (
    activation_by_name,
    batchnorm_with_activation,
    conv2d_no_bias,
    depthwise_conv2d_no_bias,
    drop_block,
    drop_connect_rates_split,
    group_norm,
    layer_norm,
    make_divisible,
    output_block,
    add_pre_post_process,
)
from keras_cv_attention_models.download_and_load import reload_model_weights


def _static_dim(tensor, axis):
    shape = tensor.shape.as_list() if hasattr(tensor.shape, "as_list") else tensor.shape
    dim = shape[axis]
    dim = dim.value if hasattr(dim, "value") else dim
    if dim is None:
        raise ValueError("MobileViT requires a statically known dimension at axis {}".format(axis))
    return int(dim)


def _resize_align_corners(inputs, size):
    if backend.is_tensorflow_backend and str(backend.tf.__version__).split(".")[0] == "1":
        return backend.tf.compat.v1.image.resize_bilinear(inputs, size, align_corners=True)
    return functional.resize(inputs, size, method="bilinear")


@backend.register_keras_serializable(package="kecamMobileViT")
class ChannelAffine(layers.Layer):
    """Per-channel scale used by the MobileViT residual helper."""

    def __init__(self, use_bias=True, weight_init_value=1, axis=-1, **kwargs):
        super().__init__(**kwargs)
        self.use_bias, self.weight_init_value, self.axis = use_bias, weight_init_value, axis

    def build(self, input_shape):
        axis = self.axis if self.axis >= 0 else len(input_shape) + self.axis
        if axis != len(input_shape) - 1:
            raise ValueError("MobileViT ChannelAffine supports channels-last tensors")
        channels = input_shape[axis]
        channels = channels.value if hasattr(channels, "value") else channels
        initializer = self.weight_init_value
        if isinstance(initializer, (int, float)):
            initializer = "ones" if initializer == 1 else backend.initializers.Constant(initializer)
        self.ww = self.add_weight(name="weight", shape=(int(channels),), initializer=initializer)
        if self.use_bias:
            self.bb = self.add_weight(name="bias", shape=(int(channels),), initializer="zeros")
        super().build(input_shape)

    def call(self, inputs, **kwargs):
        return inputs * self.ww + self.bb if self.use_bias else inputs * self.ww

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update({"use_bias": self.use_bias, "weight_init_value": self.weight_init_value, "axis": self.axis})
        return config


def add_with_layer_scale_and_drop_block(short, deep, layer_scale=0, residual_scale=0, drop_rate=0, axis="auto", name=""):
    axis = (-1 if image_data_format() == "channels_last" else 1) if axis == "auto" else axis
    short = ChannelAffine(use_bias=False, weight_init_value=residual_scale, axis=axis, name=name + "res_gamma")(short) if residual_scale > 0 else short
    deep = ChannelAffine(use_bias=False, weight_init_value=layer_scale, axis=axis, name=name + "gamma")(deep) if layer_scale > 0 else deep
    deep = drop_block(deep, drop_rate=drop_rate, name=name)
    return layers.Add(name=name + "output")([short, deep])


def mlp_block(inputs, hidden_dim, output_channel=-1, drop_rate=0, use_conv=False, use_bias=True, activation="gelu", name=None):
    channel_axis = -1 if image_data_format() == "channels_last" else (1 if use_conv else -1)
    output_channel = output_channel if output_channel > 0 else _static_dim(inputs, channel_axis)
    if use_conv:
        nn = layers.Conv2D(hidden_dim, kernel_size=1, use_bias=use_bias, name=name and name + "Conv_0")(inputs)
    else:
        nn = layers.Dense(hidden_dim, use_bias=use_bias, name=name and name + "Dense_0")(inputs)
    nn = activation_by_name(nn, activation, name=name)
    nn = layers.Dropout(drop_rate)(nn) if drop_rate > 0 else nn
    if use_conv:
        nn = layers.Conv2D(output_channel, kernel_size=1, use_bias=use_bias, name=name and name + "Conv_1")(nn)
    else:
        nn = layers.Dense(output_channel, use_bias=use_bias, name=name and name + "Dense_1")(nn)
    return layers.Dropout(drop_rate)(nn) if drop_rate > 0 else nn


def multi_head_self_attention(*args, **kwargs):
    # V1 only; defer the aggregate attention import so the TF1 MobileViTv2 path stays small.
    from keras_cv_attention_models.attention_layers import multi_head_self_attention as attention

    return attention(*args, **kwargs)


LAYER_NORM_EPSILON = 1e-5
PRETRAINED_DICT = {
    "mobilevit_xxs": {"imagenet": "f9d1d4f7329726b4bb9069cd631a152e"},
    "mobilevit_xs": {"imagenet": "dcd8591668946aa4ddb0159dfe65cc2b"},
    "mobilevit_s": {"imagenet": "55f1051035ecd14e90ae3df80fd0c7f3"},
    "mobilevit_v2_050": {"imagenet": {256: "a842a40c0f49dc2bbe935493caed061b"}},
    "mobilevit_v2_075": {"imagenet": {256: "8588b3d6bf4aa750766ddc6d01824c67"}},
    "mobilevit_v2_100": {"imagenet": {256: "55d499bbc29f0f6379a4cc6f610e10e8"}},
    "mobilevit_v2_125": {"imagenet": {256: "b8af7b7668774796530f19dd5b6080fb"}},
    "mobilevit_v2_150": {
        "imagenet": {256: "065e7a07f7e2d0d74a33913195df9044"},
        "imagenet22k": {256: "cf3c4ec278154ece62e8967faa5c0391", 384: "cdcfaebb573f8cd1f41044ac0e958204"},
    },
    "mobilevit_v2_175": {
        "imagenet": {256: "627719428c6cb35f071a7ea69a6961c4"},
        "imagenet22k": {256: "b849708a6b2c1f115b8b8c366e1d1a19", 384: "d3feef5108b6195d1c5525fb185bf720"},
    },
    "mobilevit_v2_200": {
        "imagenet": {256: "1fe59d8bb2662761084d1c04259a778d"},
        "imagenet22k": {256: "931f0be1761bcf8443359ec1661bb6a7", 384: "1dc6cdafb187611e5a4819272d64fba7"},
    },
}


def bottle_in_linear_out_block(inputs, out_channel, strides=1, expand_ratio=4, use_shortcut=False, drop_rate=0, activation="swish", name=""):
    input_channel = _static_dim(inputs, -1 if image_data_format() == "channels_last" else 1)
    hidden_dim = int(input_channel * expand_ratio)
    deep = conv2d_no_bias(inputs, hidden_dim, kernel_size=1, strides=1, name=name + "deep_1_")
    deep = batchnorm_with_activation(deep, activation=activation, name=name + "deep_1_")
    deep = depthwise_conv2d_no_bias(deep, kernel_size=3, strides=strides, padding="same", name=name + "deep_2_")
    deep = batchnorm_with_activation(deep, activation=activation, name=name + "deep_2_")
    deep = conv2d_no_bias(deep, out_channel, kernel_size=1, strides=1, name=name + "deep_3_")
    deep = batchnorm_with_activation(deep, activation=None, name=name + "deep_3_")
    deep = drop_block(deep, drop_rate=drop_rate, name=name + "deep_")

    out = layers.Add()([inputs, deep]) if use_shortcut else deep
    return out


def linear_self_attention(inputs, qkv_bias=False, out_bias=False, attn_axis=2, attn_dropout=0, name=None):
    channel_axis = -1 if image_data_format() == "channels_last" else 1
    input_channel = _static_dim(inputs, channel_axis)
    qkv = conv2d_no_bias(inputs, 1 + input_channel * 2, kernel_size=1, use_bias=qkv_bias, name=name and name + "qkv_")
    query, key, value = functional.split(qkv, [1, input_channel, input_channel], axis=channel_axis)
    context_score = layers.Softmax(axis=attn_axis, name=name and name + "attention_scores")(query)  # on patch_hh * patch_ww dimension
    context_score = layers.Dropout(attn_dropout, name=name and name + "attn_drop")(context_score) if attn_dropout > 0 else context_score
    # print(f"{query.shape = }, {key.shape = }, {value.shape = }, {context_score.shape = }")

    context_vector = layers.Multiply()([key, context_score])  # [batch, height, width, input_channel]
    context_vector = functional.reduce_sum(context_vector, keepdims=True, axis=attn_axis)  # on patch_hh * patch_ww dimension

    out = functional.relu(value) * context_vector
    out = conv2d_no_bias(out, input_channel, kernel_size=1, use_bias=out_bias, name=name and name + "output")
    return out


def mhsa_mlp_block(
    inputs,
    out_channel,
    num_heads=4,
    qkv_bias=True,
    mlp_ratio=4,
    num_norm_groups=-1,  # -1 or 0 for V1 using layer_norm, or 1 for V2 using group_norm
    mlp_drop_rate=0,
    attn_drop_rate=0,
    drop_rate=0,
    layer_scale=-1,
    activation="gelu",
    name=None,
):
    # print(f"{inputs.shape = }")
    attn = group_norm(inputs, groups=num_norm_groups, axis=-1, name=name + "attn_") if num_norm_groups > 0 else layer_norm(inputs, axis=-1, name=name + "attn_")
    attn = multi_head_self_attention(attn, num_heads, qkv_bias=qkv_bias, out_bias=True, attn_dropout=attn_drop_rate, name=name and name + "attn_mhsa_")
    attn = add_with_layer_scale_and_drop_block(inputs, attn, layer_scale=layer_scale, drop_rate=drop_rate, axis=-1, name=name and name + "1_")

    mlp = group_norm(attn, groups=num_norm_groups, axis=-1, name=name + "mlp_") if num_norm_groups > 0 else layer_norm(attn, axis=-1, name=name + "mlp_")
    mlp = mlp_block(mlp, int(out_channel * mlp_ratio), drop_rate=mlp_drop_rate, activation=activation, name=name and name + "mlp_")
    return add_with_layer_scale_and_drop_block(attn, mlp, layer_scale=layer_scale, drop_rate=drop_rate, axis=-1, name=name and name + "2_")


def linear_mhsa_mlp_block(
    inputs,
    out_channel,
    num_heads=4,
    qkv_bias=True,
    mlp_ratio=4,
    num_norm_groups=-1,  # -1 or 0 for V1 using layer_norm, or 1 for V2 using group_norm
    mlp_drop_rate=0,
    attn_drop_rate=0,
    drop_rate=0,
    layer_scale=-1,
    activation="gelu",
    name=None,
):
    attn = group_norm(inputs, groups=num_norm_groups, name=name + "attn_") if num_norm_groups > 0 else layer_norm(inputs, name=name + "attn_")
    attn = layers.Reshape([_static_dim(attn, axis) for axis in range(1, len(attn.shape))])(attn)  # Keras 2.2 needs Python ints here
    attn = linear_self_attention(attn, qkv_bias=qkv_bias, out_bias=True, attn_dropout=attn_drop_rate, name=name and name + "attn_mhsa_")
    attn = add_with_layer_scale_and_drop_block(inputs, attn, layer_scale=layer_scale, drop_rate=drop_rate, name=name and name + "1_")

    mlp = group_norm(attn, groups=num_norm_groups, name=name + "mlp_") if num_norm_groups > 0 else layer_norm(attn, name=name + "mlp_")
    mlp = layers.Reshape([_static_dim(mlp, axis) for axis in range(1, len(mlp.shape))])(mlp)
    mlp = mlp_block(mlp, int(out_channel * mlp_ratio), drop_rate=mlp_drop_rate, use_conv=True, activation=activation, name=name and name + "mlp_")
    return add_with_layer_scale_and_drop_block(attn, mlp, layer_scale=layer_scale, drop_rate=drop_rate, name=name and name + "2_")


def transformer_pre_process(inputs, out_channel, patch_size=2, resize_first=False, use_depthwise=False, patches_to_batch=True, activation="swish", name=""):
    height_axis, width_axis, channel_axis = (1, 2, 3) if image_data_format() == "channels_last" else (2, 3, 1)
    nn = inputs

    if resize_first:  # V2
        input_height, input_width = _static_dim(nn, height_axis), _static_dim(nn, width_axis)
        patch_hh, patch_ww = (input_height + patch_size - 1) // patch_size, (input_width + patch_size - 1) // patch_size
        # print(f"transformer_pre_process before resize: {nn.shape = }")
        if patch_hh * patch_size != input_height or patch_ww * patch_size != input_width:
            nn = _resize_align_corners(nn, [patch_hh * patch_size, patch_ww * patch_size])

    if use_depthwise:  # V2
        nn = depthwise_conv2d_no_bias(nn, kernel_size=3, strides=1, padding="same", name=name + "pre_1_")
    else:  # V1
        nn = conv2d_no_bias(nn, _static_dim(nn, channel_axis), kernel_size=3, strides=1, padding="same", name=name + "pre_1_")
    nn = batchnorm_with_activation(nn, activation=activation, name=name + "pre_1_")
    nn = conv2d_no_bias(nn, out_channel, kernel_size=1, strides=1, name=name + "pre_2_")

    if not resize_first:  # V1
        input_height, input_width = _static_dim(nn, height_axis), _static_dim(nn, width_axis)
        patch_hh, patch_ww = (input_height + patch_size - 1) // patch_size, (input_width + patch_size - 1) // patch_size
        # print(f"transformer_pre_process before resize: {nn.shape = }")
        if patch_hh * patch_size != input_height or patch_ww * patch_size != input_width:
            nn = _resize_align_corners(nn, [patch_hh * patch_size, patch_ww * patch_size])

    # Extract patchs, limit transpose permute length <= 4
    # [batch, height, width, channel] -> [batch, height // 2, 2, width // 2, 2, channel] -> [batch * 4, height // 2, width // 2, channel]
    # print(f"transformer_pre_process after resize: {nn.shape = }")
    if image_data_format() == "channels_last":
        nn = functional.reshape(nn, [-1, patch_ww, patch_size, out_channel])  # [B * patch_hh * h_patch_size, patch_ww, w_patch_size, C]
        nn = functional.transpose(nn, [0, 2, 1, 3])  # [B * patch_hh * h_patch_size, w_patch_size, patch_ww, C]
        nn = functional.reshape(nn, [-1, patch_hh, patch_size * patch_size, patch_ww * out_channel])  # [B, patch_hh, h_patch_size * w_patch_size, patch_ww * C]
        nn = functional.transpose(nn, [0, 2, 1, 3])  # [B, h_patch_size * w_patch_size, patch_hh, patch_ww * C]
        extract_shape = [-1, patch_hh, patch_ww, out_channel] if patches_to_batch else [-1, patch_size * patch_size, patch_hh * patch_ww, out_channel]
        nn = functional.reshape(nn, extract_shape)  # channels_last
    else:  # [B, C, patch_hh * h_patch_size, patch_ww * w_patch_size]
        nn = functional.reshape(nn, [-1, patch_size, patch_ww, patch_size])  # [B * C * patch_hh, h_patch_size, patch_ww, w_patch_size]
        nn = functional.transpose(nn, [0, 2, 1, 3])  # [B * C * patch_hh, patch_ww, h_patch_size, w_patch_size]
        nn = functional.reshape(nn, [-1, out_channel, patch_hh * patch_ww, patch_size * patch_size])  # [B, C, patch_hh * patch_ww, h_patch_size * w_patch_size]
        if patches_to_batch:  # V1 -> channels_last, V2 keep channels_first, `patch_hh * patch_ww` -> attn_axis=2
            nn = functional.transpose(nn, [0, 3, 2, 1])  # [B, h_patch_size * w_patch_size, patch_hh * patch_ww, C]
            nn = functional.reshape(nn, [-1, patch_hh, patch_ww, out_channel])  # channels_last
    return nn


def transformer_post_process(inputs, pre_attn, out_channel, patch_size=2, patch_height=-1, activation="swish", name=""):
    height_axis, width_axis, channel_axis = (1, 2, 3) if image_data_format() == "channels_last" else (2, 3, 1)
    patches_to_batch = patch_height == -1
    if patches_to_batch:  # V1, [batch * 4, height // 2, width // 2, channel], channels_last
        patch_hh, patch_ww, channel = _static_dim(inputs, 1), _static_dim(inputs, 2), _static_dim(inputs, 3)
    else:  # V2, [batch, 4, height // 2 * width // 2, channel], channels_last or channels_first
        patch_hh, patch_ww, channel = int(patch_height), _static_dim(inputs, 2) // int(patch_height), _static_dim(inputs, channel_axis)
    # print(f"{patch_hh = }, {patch_ww = }, {channel = }, {inputs.shape = }")

    # [B * 4, height // 2, width // 2, C] -> [B, height // 2, 2, width // 2, 2, C] -> [B, height, width, C]
    if image_data_format() == "channels_last":
        nn = functional.reshape(inputs, [-1, patch_size * patch_size, patch_hh, patch_ww * channel])  # [B, h_patch_size * w_patch_size, patch_hh, patch_ww * C]
        nn = functional.transpose(nn, [0, 2, 1, 3])  # [B, patch_hh, h_patch_size * w_patch_size, patch_ww * C]
        nn = functional.reshape(nn, [-1, patch_size, patch_ww, channel])  # [B * patch_hh * h_patch_size, w_patch_size, patch_ww, C]
        nn = functional.transpose(nn, [0, 2, 1, 3])  # [B * patch_hh * h_patch_size, patch_ww, w_patch_size, C]
        nn = functional.reshape(nn, [-1, patch_hh * patch_size, patch_ww * patch_size, channel])
    else:
        nn = inputs
        if patches_to_batch:
            nn = functional.reshape(nn, [-1, patch_size * patch_size, patch_hh * patch_ww, channel])  # [B, h_patch_size * w_patch_size, patch_hh * patch_ww, C]
            nn = functional.transpose(nn, [0, 3, 2, 1])  # [B, C, patch_hh * patch_ww, h_patch_size * w_patch_size]
        nn = functional.reshape(nn, [-1, patch_ww, patch_size, patch_size])  # [B * C * patch_hh, patch_ww, h_patch_size, w_patch_size]
        nn = functional.transpose(nn, [0, 2, 1, 3])  # [B * C * patch_hh, h_patch_size, patch_ww, w_patch_size]
        nn = functional.reshape(nn, [-1, channel, patch_hh * patch_size, patch_ww * patch_size])

    # print(f"transformer_post_process before resize: {nn.shape = }")
    if pre_attn is not None:
        pre_height, pre_width = _static_dim(pre_attn, height_axis), _static_dim(pre_attn, width_axis)
        if _static_dim(nn, height_axis) != pre_height or _static_dim(nn, width_axis) != pre_width:
            nn = _resize_align_corners(nn, [pre_height, pre_width])
    # print(f"transformer_post_process after resize: {nn.shape = }")

    nn = conv2d_no_bias(nn, out_channel, kernel_size=1, strides=1, name=name + "post_1_")
    nn = batchnorm_with_activation(nn, activation=activation, name=name + "post_1_")
    if pre_attn is not None:  # V1
        nn = functional.concat([pre_attn, nn], axis=channel_axis)
        nn = conv2d_no_bias(nn, out_channel, kernel_size=3, strides=1, padding="same", name=name + "post_2_")
        nn = batchnorm_with_activation(nn, activation=activation, name=name + "post_2_")
    return layers.Activation("linear", name=name + "output")(nn)  # Identity, Just need a name here


def MobileViT(
    num_blocks=[1, 3, 3, 5, 4],
    out_channels=[32, 64, 96, 128, 160],
    attn_channels=[0, 0, 144, 192, 240],  # Can be a list matching out_channels, or a float number for expansion ratio of out_channels
    block_types=["conv", "conv", "transform", "transform", "transform"],
    strides=[1, 2, 2, 2, 2],
    expand_ratio=4,
    stem_width=16,
    patch_size=2,
    resize_first=False,  # False for V1, True for V2
    use_depthwise=False,  # False for V1, True for V2
    use_fusion=True,  # True for V1, False for V2
    num_norm_groups=-1,  # -1 or 0 for V1 using layer_norm, or 1 for V2 using group_norm
    use_linear_attention=False,  # False for V1, True for V2
    output_num_features=640,
    layer_scale=-1,
    input_shape=(256, 256, 3),
    num_classes=1000,
    activation="swish",
    drop_connect_rate=0,
    classifier_activation="softmax",
    dropout=0,
    pretrained=None,
    model_name="mobilevit",
    kwargs=None,
):
    # Regard input_shape as force using original shape if len(input_shape) == 4,
    # else assume channel dimension is the one with min value in input_shape, and put it first or last regarding image_data_format
    input_shape = backend.align_input_shape_by_image_data_format(input_shape)
    inputs = layers.Input(input_shape)
    nn = conv2d_no_bias(inputs, stem_width, kernel_size=3, strides=2, padding="same", name="stem_")
    nn = batchnorm_with_activation(nn, activation=activation, name="stem_")
    height_axis, width_axis, channel_axis = (1, 2, -1) if image_data_format() == "channels_last" else (2, 3, 1)

    # Save line width
    mhsa_mlp_block_common_kwargs = {
        "num_heads": 4,
        "qkv_bias": True,
        "mlp_ratio": 2,
        "num_norm_groups": num_norm_groups,
        "activation": activation,
    }

    """ stages """
    total_blocks = sum(num_blocks)
    global_block_id = 0
    post_activation = activation if use_fusion else None
    for id, (num_block, out_channel, block_type, stride) in enumerate(zip(num_blocks, out_channels, block_types, strides)):
        stack_name = "stack{}_".format(id + 1)
        is_conv_block = True if block_type[0].lower() == "c" else False
        attn_channel = attn_channels[id] if isinstance(attn_channels, (list, tuple)) else make_divisible(attn_channels * out_channel, divisor=8)
        for block_id in range(num_block):
            name = stack_name + "block{}_".format(block_id + 1)
            stride = stride if block_id == 0 else 1
            use_shortcut = False if stride != 1 or _static_dim(nn, channel_axis) != out_channel else True
            block_drop_rate = drop_connect_rate * global_block_id / total_blocks
            global_block_id += 1
            if is_conv_block or block_id == 0:  # First transformer block is also a conv block .
                nn = bottle_in_linear_out_block(nn, out_channel, stride, expand_ratio, use_shortcut, block_drop_rate, activation=activation, name=name)
            else:
                if block_id == 1:  # pre
                    pre_attn = nn if use_fusion else None
                    patches_to_batch = not use_linear_attention
                    patch_height = -1 if patches_to_batch else (_static_dim(nn, height_axis) + patch_size - 1) // patch_size
                    nn = transformer_pre_process(nn, attn_channel, patch_size, resize_first, use_depthwise, patches_to_batch, activation=activation, name=name)

                if use_linear_attention:  # channels_last for Tensorflow, channels_first for PyTorch
                    nn = linear_mhsa_mlp_block(nn, attn_channel, layer_scale=layer_scale, **mhsa_mlp_block_common_kwargs, name=name)
                else:  # channels_last for both Tensorflow or PyTorch
                    if block_id == 1:
                        block_height, block_width = [_static_dim(nn, axis) for axis in (1, 2)]
                        channel_axis = -1
                        nn = functional.reshape(nn, [-1, block_height * block_width, _static_dim(nn, -1)])  # Using 3D for attention inputs
                    nn = mhsa_mlp_block(nn, attn_channel, layer_scale=layer_scale, **mhsa_mlp_block_common_kwargs, name=name)
                    if block_id == num_block - 1:
                        channel_axis = -1 if image_data_format() == "channels_last" else 1
                        nn = functional.reshape(nn, [-1, block_height, block_width, _static_dim(nn, -1)])  # Revert 3D to 4D

                if block_id == num_block - 1:  # post
                    norm_axis = "auto" if use_linear_attention else -1
                    if use_linear_attention:
                        nn = group_norm(nn, groups=num_norm_groups, axis=norm_axis, name=name + "post_")
                    else:
                        nn = layer_norm(nn, axis=norm_axis, name=name + "post_")
                    nn = transformer_post_process(nn, pre_attn, out_channel, patch_size, patch_height, activation=post_activation, name=name)

    nn = output_block(nn, output_num_features, activation, num_classes, drop_rate=dropout, classifier_activation=classifier_activation)
    model = models.Model(inputs, nn, name=model_name)
    add_pre_post_process(model, rescale_mode="raw01")
    reload_model_weights(model, PRETRAINED_DICT, "mobilevit", pretrained)
    return model


@register_model
def MobileViT_XXS(input_shape=(256, 256, 3), num_classes=1000, activation="swish", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    num_blocks = [1, 3, 3, 5, 4]
    out_channels = [16, 24, 48, 64, 80]
    attn_channels = [0, 0, 64, 80, 96]
    output_num_features = 320
    expand_ratio = 2
    return MobileViT(**locals(), model_name="mobilevit_xxs", **kwargs)


@register_model
def MobileViT_XS(input_shape=(256, 256, 3), num_classes=1000, activation="swish", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    num_blocks = [1, 3, 3, 5, 4]
    out_channels = [32, 48, 64, 80, 96]
    attn_channels = 1.5
    output_num_features = 384
    return MobileViT(**locals(), model_name="mobilevit_xs", **kwargs)


@register_model
def MobileViT_S(input_shape=(256, 256, 3), num_classes=1000, activation="swish", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    num_blocks = [1, 3, 3, 5, 4]
    out_channels = [32, 64, 96, 128, 160]
    attn_channels = 1.5
    return MobileViT(**locals(), model_name="mobilevit_s", **kwargs)
