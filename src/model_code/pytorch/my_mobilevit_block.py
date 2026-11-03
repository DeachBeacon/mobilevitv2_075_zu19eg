#
# For licensing see accompanying license file.
# Copyright (C) 2023 Apple Inc. All Rights Reserved.
#

import math
from typing import Dict, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from cvnets.layers import ConvLayer2d, get_normalization_layer
from cvnets.modules.base_module import BaseModule
from cvnets.modules.transformer import LinearAttnFFN, TransformerEncoder

def _pixel_unshuffle(x: Tensor, downscale_factor: int) -> Tensor:
    """用 Reshape + Transpose 等价实现 pixel_unshuffle"""
    B, C, H, W = x.shape
    ps = downscale_factor
    Hp, Wp = H // ps, W // ps
    x = x.reshape(B, C, Hp, ps, Wp, ps)
    x = x.permute(0, 1, 3, 5, 2, 4)  # (B, C, ps, ps, Hp, Wp)
    x = x.reshape(B, C * ps * ps, Hp, Wp)

    return x


class my_MobileViTBlockv2(BaseModule):
    """
    MobileViTv2 block with unfold / fold implemented via pixel_unshuffle / pixel_shuffle.
    """

    def __init__(
        self,
        opts,
        in_channels: int,
        attn_unit_dim: int,
        ffn_multiplier: Optional[Union[Sequence[Union[int, float]], int, float]] = 2.0,
        n_attn_blocks: Optional[int] = 2,
        attn_dropout: Optional[float] = 0.0,
        dropout: Optional[float] = 0.0,
        ffn_dropout: Optional[float] = 0.0,
        patch_h: Optional[int] = 8,
        patch_w: Optional[int] = 8,
        conv_ksize: Optional[int] = 3,
        dilation: Optional[int] = 1,
        attn_norm_layer: Optional[str] = "layer_norm_2d",
        *args,
        **kwargs
    ) -> None:
        cnn_out_dim = attn_unit_dim

        conv_3x3_in = ConvLayer2d(
            opts=opts,
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=conv_ksize,
            stride=1,
            use_norm=True,
            use_act=True,
            dilation=dilation,
            groups=in_channels,
        )
        conv_1x1_in = ConvLayer2d(
            opts=opts,
            in_channels=in_channels,
            out_channels=cnn_out_dim,
            kernel_size=1,
            stride=1,
            use_norm=False,
            use_act=False,
        )

        super(my_MobileViTBlockv2, self).__init__()
        self.local_rep = nn.Sequential(conv_3x3_in, conv_1x1_in)

        self.global_rep, attn_unit_dim = self._build_attn_layer(
            opts=opts,
            d_model=attn_unit_dim,
            ffn_mult=ffn_multiplier,
            n_layers=n_attn_blocks,
            attn_dropout=attn_dropout,
            dropout=dropout,
            ffn_dropout=ffn_dropout,
            attn_norm_layer=attn_norm_layer,
        )

        self.conv_proj = ConvLayer2d(
            opts=opts,
            in_channels=cnn_out_dim,
            out_channels=in_channels,
            kernel_size=1,
            stride=1,
            use_norm=True,
            use_act=False,
        )

        self.patch_h = patch_h
        self.patch_w = patch_w
        self.patch_area = self.patch_w * self.patch_h

        self.cnn_in_dim = in_channels
        self.cnn_out_dim = cnn_out_dim
        self.transformer_in_dim = attn_unit_dim
        self.dropout = dropout
        self.attn_dropout = attn_dropout
        self.ffn_dropout = ffn_dropout
        self.n_blocks = n_attn_blocks
        self.conv_ksize = conv_ksize

        # patch 必须为方形才能使用 pixel_unshuffle / pixel_shuffle
        if self.patch_h != self.patch_w:
            raise ValueError(
                f"patch_h ({patch_h}) must equal patch_w ({patch_w}) for pixel_unshuffle/pixel_shuffle."
            )

    def _build_attn_layer(
        self,
        opts,
        d_model: int,
        ffn_mult: Union[Sequence, int, float],
        n_layers: int,
        attn_dropout: float,
        dropout: float,
        ffn_dropout: float,
        attn_norm_layer: str,
        *args,
        **kwargs
    ) -> Tuple[nn.Module, int]:
        if isinstance(ffn_mult, Sequence) and len(ffn_mult) == 2:
            ffn_dims = (
                np.linspace(ffn_mult[0], ffn_mult[1], n_layers, dtype=float) * d_model
            )
        elif isinstance(ffn_mult, Sequence) and len(ffn_mult) == 1:
            ffn_dims = [ffn_mult[0] * d_model] * n_layers
        elif isinstance(ffn_mult, (int, float)):
            ffn_dims = [ffn_mult * d_model] * n_layers
        else:
            raise NotImplementedError

        ffn_dims = [int((d // 16) * 16) for d in ffn_dims]

        global_rep = [
            LinearAttnFFN(
                opts=opts,
                embed_dim=d_model,
                ffn_latent_dim=ffn_dims[block_idx],
                attn_dropout=attn_dropout,
                dropout=dropout,
                ffn_dropout=ffn_dropout,
                norm_layer=attn_norm_layer,
            )
            for block_idx in range(n_layers)
        ]
        global_rep.append(
            get_normalization_layer(
                opts=opts, norm_type=attn_norm_layer, num_features=d_model
            )
        )

        return nn.Sequential(*global_rep), d_model

    def unfolding_pixel_unshuffle(self, feature_map: Tensor) -> Tuple[Tensor, Tuple[int, int]]:
        """
        用 _pixel_unshuffle 替换 F.unfold。
        输入:  (B, C, H, W)
        输出:  patches (B, C, P, N), 原始空间尺寸 (H, W)
        """
        batch_size, in_channels, img_h, img_w = feature_map.shape
        # (B, C * P, H/P, W/P)
        patches = _pixel_unshuffle(feature_map, self.patch_h)
        # 重塑为 (B, C, P, N)
        patches = patches.reshape(
            batch_size, in_channels, self.patch_h * self.patch_w, -1
        )
        return patches, (img_h, img_w)

    def folding_pixel_shuffle(self, patches: Tensor, output_size: Tuple[int, int]) -> Tensor:
        """
        用 F.pixel_shuffle 替换 F.fold。
        输入:  patches (B, C, P, N), output_size (H, W)
        输出:  特征图 (B, C, H, W)
        """
        batch_size, in_dim, patch_size, n_patches = patches.shape
        n_patches_h = output_size[0] // self.patch_h
        n_patches_w = output_size[1] // self.patch_w

        # (B, C * P, H/P, W/P)
        feature_map = patches.reshape(
            batch_size, in_dim * self.patch_h * self.patch_w, n_patches_h, n_patches_w
        )
        # pixel_shuffle 将通道维的空间信息还原到空间维
        feature_map = F.pixel_shuffle(feature_map, upscale_factor=self.patch_h)
        return feature_map

    def resize_input_if_needed(self, x: Tensor) -> Tensor:
        batch_size, in_channels, orig_h, orig_w = x.shape
        if orig_h % self.patch_h != 0 or orig_w % self.patch_w != 0:
            new_h = int(math.ceil(orig_h / self.patch_h) * self.patch_h)
            new_w = int(math.ceil(orig_w / self.patch_w) * self.patch_w)
            x = F.interpolate(
                x, size=(new_h, new_w), mode="bilinear", align_corners=True
            )
        return x

    def forward_spatial(self, x: Tensor, *args, **kwargs) -> Tensor:
        x = self.resize_input_if_needed(x)

        fm = self.local_rep(x)

        # 使用 pixel_unshuffle 进行 unfolding
        patches, output_size = self.unfolding_pixel_unshuffle(fm)

        # 全局表征学习
        patches = self.global_rep(patches)

        # 使用 pixel_shuffle 进行 folding
        fm = self.folding_pixel_shuffle(patches=patches, output_size=output_size)
        fm = self.conv_proj(fm)

        return fm

    def forward_temporal(
        self, x: Tensor, x_prev: Tensor, *args, **kwargs
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        x = self.resize_input_if_needed(x)

        fm = self.local_rep(x)

        patches, output_size = self.unfolding_pixel_unshuffle(fm)

        for global_layer in self.global_rep:
            if isinstance(global_layer, LinearAttnFFN):
                patches = global_layer(x=patches, x_prev=x_prev)
            else:
                patches = global_layer(patches)

        fm = self.folding_pixel_shuffle(patches=patches, output_size=output_size)
        fm = self.conv_proj(fm)

        return fm, patches

    def forward(
        self, x: Union[Tensor, Tuple[Tensor]], *args, **kwargs
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        if isinstance(x, Tuple) and len(x) == 2:
            return self.forward_temporal(x=x[0], x_prev=x[1])
        elif isinstance(x, Tensor):
            return self.forward_spatial(x)
        else:
            raise NotImplementedError

    def __repr__(self) -> str:
        repr_str = "{}(".format(self.__class__.__name__)

        repr_str += "\n\t Local representations"
        if isinstance(self.local_rep, nn.Sequential):
            for m in self.local_rep:
                repr_str += "\n\t\t {}".format(m)
        else:
            repr_str += "\n\t\t {}".format(self.local_rep)

        repr_str += "\n\t Global representations with patch size of {}x{}".format(
            self.patch_h,
            self.patch_w,
        )
        if isinstance(self.global_rep, nn.Sequential):
            for m in self.global_rep:
                repr_str += "\n\t\t {}".format(m)
        else:
            repr_str += "\n\t\t {}".format(self.global_rep)

        if isinstance(self.conv_proj, nn.Sequential):
            for m in self.conv_proj:
                repr_str += "\n\t\t {}".format(m)
        else:
            repr_str += "\n\t\t {}".format(self.conv_proj)

        repr_str += "\n)"
        return repr_str