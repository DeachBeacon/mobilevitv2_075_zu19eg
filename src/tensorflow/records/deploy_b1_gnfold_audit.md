# MobileViTv2 deployment GN affine folding audit

Validation: **PASS**, TensorFlow 1.15.2 CPU, 2026-10-28.

- Source: `C:\WorkSpace\Z19-Onborad\Model\keras_cv_attention_models\2board\artifacts\mobilevitv2_tf115\mobilevitv2_075_10cls_relu_deploy_b1.pb`
- Source SHA256: `78C15505EC9BC402EF5A07DC99DEBA942BAB549DC46ABEB0554230088D0C3132`
- New PB: `C:\WorkSpace\Z19-Onborad\Model\keras_cv_attention_models\2board\artifacts\mobilevitv2_tf115\mobilevitv2_075_10cls_relu_deploy_b1_gnfold.pb`
- New PB SHA256: `C205A47377EE7FFC8683F43F7A95B5A46A00F8B0B9282CEE5EBC147C23E0DE2D`
- Input: `input_1:0`, float32 `[1,256,256,3]`; output: `predictions/BiasAdd:0`, float32 `[1,10]` logits.
- Original deploy_b1, inference and frozen PBs are retained with verified SHA256 hashes. No retraining.
- Nodes: 1255 -> 1135 (126 deleted, 6 added); 21 kernels and 18 existing biases changed; 3 bias-free convolutions received a new bias Const and BiasAdd before their existing BN.

## Algebra and channel proof

For original HWIO kernel W[1,1,Cin,Cout]:

- W'[0,0,i,o] = gamma[i] * original_W[0,0,i,o]
- b'[o] = original_b[o] + sum_i(beta[i] * original_W[0,0,i,o])
- Original bias is zero for a bias-free convolution. Calculations/accumulations use float64; deployed constants are float32.
- A separate einsum implementation rechecked all 21 kernels and all 21 biases against the original PB; resulting constants match exactly.
- Each path is a single-consumer chain without control dependencies. Only Identity, Reshape and Transpose are allowed before a VALID 1x1 NHWC Conv2D with unit stride/dilation.
- Exhaustive per-element channel labels are transformed through every layout operation. All 21 final inputs preserve label i at Conv2D input channel i. Intermediate spatial/channel packing in the three post-GN paths is explicitly included.

## All 21 audited folds

Each row passed 14 independent local comparisons using exactly the same pre-affine normalized input from the original graph. Local errors exclude accumulated upstream error. Full per-operation shapes, integer parameters and per-sample errors are in the JSON report.

| # | GroupNorm | Layout operations before convolution | Target Conv2D / kernel shape | Bias | Elements checked / channel mismatches | Local max absolute error |
| ---: | --- | --- | --- | --- | --- | ---: |
| 1 | `stack3_block2_attn_group_norm` | `reshape_3/Reshape` (Reshape) | `stack3_block2_attn_mhsa_qkv_conv/Conv2D` / `[1, 1, 96, 193]` | Existing bias updated | 98304 / 0 | 4.76837158e-06 |
| 2 | `stack3_block2_mlp_group_norm` | `reshape_1_1/Reshape` (Reshape) | `stack3_block2_mlp_Conv_0/Conv2D` / `[1, 1, 96, 192]` | Existing bias updated | 98304 / 0 | 1.33514404e-05 |
| 3 | `stack3_block3_attn_group_norm` | `reshape_2_1/Reshape` (Reshape) | `stack3_block3_attn_mhsa_qkv_conv/Conv2D` / `[1, 1, 96, 193]` | Existing bias updated | 98304 / 0 | 7.62939453e-06 |
| 4 | `stack3_block3_mlp_group_norm` | `reshape_3_1/Reshape` (Reshape) | `stack3_block3_mlp_Conv_0/Conv2D` / `[1, 1, 96, 192]` | Existing bias updated | 98304 / 0 | 2.28881836e-05 |
| 5 | `stack3_block3_post_group_norm` | `Reshape_4` (Reshape) -> `transpose_2` (Transpose) -> `Reshape_5` (Reshape) -> `transpose_3` (Transpose) -> `Reshape_6` (Reshape) | `stack3_block3_post_1_conv/Conv2D` / `[1, 1, 96, 192]` | New BiasAdd | 98304 / 0 | 4.57763672e-05 |
| 6 | `stack4_block2_attn_group_norm` | `reshape_4_1/Reshape` (Reshape) | `stack4_block2_attn_mhsa_qkv_conv/Conv2D` / `[1, 1, 144, 289]` | Existing bias updated | 36864 / 0 | 8.58306885e-06 |
| 7 | `stack4_block2_mlp_group_norm` | `reshape_5_1/Reshape` (Reshape) | `stack4_block2_mlp_Conv_0/Conv2D` / `[1, 1, 144, 288]` | Existing bias updated | 36864 / 0 | 1.90734863e-05 |
| 8 | `stack4_block3_attn_group_norm` | `reshape_6_1/Reshape` (Reshape) | `stack4_block3_attn_mhsa_qkv_conv/Conv2D` / `[1, 1, 144, 289]` | Existing bias updated | 36864 / 0 | 8.58306885e-06 |
| 9 | `stack4_block3_mlp_group_norm` | `reshape_7_1/Reshape` (Reshape) | `stack4_block3_mlp_Conv_0/Conv2D` / `[1, 1, 144, 288]` | Existing bias updated | 36864 / 0 | 1.90734863e-05 |
| 10 | `stack4_block4_attn_group_norm` | `reshape_8_1/Reshape` (Reshape) | `stack4_block4_attn_mhsa_qkv_conv/Conv2D` / `[1, 1, 144, 289]` | Existing bias updated | 36864 / 0 | 7.62939453e-06 |
| 11 | `stack4_block4_mlp_group_norm` | `reshape_9_1/Reshape` (Reshape) | `stack4_block4_mlp_Conv_0/Conv2D` / `[1, 1, 144, 288]` | Existing bias updated | 36864 / 0 | 3.05175781e-05 |
| 12 | `stack4_block5_attn_group_norm` | `reshape_10/Reshape` (Reshape) | `stack4_block5_attn_mhsa_qkv_conv/Conv2D` / `[1, 1, 144, 289]` | Existing bias updated | 36864 / 0 | 8.58306885e-06 |
| 13 | `stack4_block5_mlp_group_norm` | `reshape_11/Reshape` (Reshape) | `stack4_block5_mlp_Conv_0/Conv2D` / `[1, 1, 144, 288]` | Existing bias updated | 36864 / 0 | 2.0980835e-05 |
| 14 | `stack4_block5_post_group_norm` | `Reshape_12` (Reshape) -> `transpose_6` (Transpose) -> `Reshape_13` (Reshape) -> `transpose_7` (Transpose) -> `Reshape_14` (Reshape) | `stack4_block5_post_1_conv/Conv2D` / `[1, 1, 144, 288]` | New BiasAdd | 36864 / 0 | 4.19616699e-05 |
| 15 | `stack5_block2_attn_group_norm` | `reshape_12_1/Reshape` (Reshape) | `stack5_block2_attn_mhsa_qkv_conv/Conv2D` / `[1, 1, 192, 385]` | Existing bias updated | 12288 / 0 | 6.19888306e-06 |
| 16 | `stack5_block2_mlp_group_norm` | `reshape_13_1/Reshape` (Reshape) | `stack5_block2_mlp_Conv_0/Conv2D` / `[1, 1, 192, 384]` | Existing bias updated | 12288 / 0 | 1.90734863e-05 |
| 17 | `stack5_block3_attn_group_norm` | `reshape_14_1/Reshape` (Reshape) | `stack5_block3_attn_mhsa_qkv_conv/Conv2D` / `[1, 1, 192, 385]` | Existing bias updated | 12288 / 0 | 1.14440918e-05 |
| 18 | `stack5_block3_mlp_group_norm` | `reshape_15_1/Reshape` (Reshape) | `stack5_block3_mlp_Conv_0/Conv2D` / `[1, 1, 192, 384]` | Existing bias updated | 12288 / 0 | 8.39233398e-05 |
| 19 | `stack5_block4_attn_group_norm` | `reshape_16_1/Reshape` (Reshape) | `stack5_block4_attn_mhsa_qkv_conv/Conv2D` / `[1, 1, 192, 385]` | Existing bias updated | 12288 / 0 | 2.28881836e-05 |
| 20 | `stack5_block4_mlp_group_norm` | `reshape_17_1/Reshape` (Reshape) | `stack5_block4_mlp_Conv_0/Conv2D` / `[1, 1, 192, 384]` | Existing bias updated | 12288 / 0 | 8.39233398e-05 |
| 21 | `stack5_block4_post_group_norm` | `Reshape_18` (Reshape) -> `transpose_10` (Transpose) -> `Reshape_19` (Reshape) -> `transpose_11` (Transpose) -> `Reshape_20` (Reshape) | `stack5_block4_post_1_conv/Conv2D` / `[1, 1, 192, 384]` | New BiasAdd | 12288 / 0 | 0.000183105469 |

## Preserved computations

- All 21 dynamic GroupNorm cores have exactly equal NodeDefs, including dynamic Mean, SquaredDifference, StopGradient, Sub, Rsqrt, normalized output Reshape, reduction axes and epsilon.
- Grouped reductions stay on axes [1,3], groups=1; these correspond to original input axes [1,2,3]. float32 epsilon stays 9.999999747378752e-6.
- All 9 Softmax NodeDefs and their existing layout/axis constants are unchanged.
- All 25 inference BatchNorm attributes and parameters are unchanged. Only the data edge of the 3 post-convolution BNs is redirected through the new BiasAdd.
- All 188 integer shape/permutation Const NodeDefs are exactly equal; no nonconstant integer shape operation is introduced.
- The only removed nodes are 42 affine arithmetic nodes, 42 private gamma/beta Const nodes and 42 private parameter Identity reads.
- All 1066 other existing NodeDefs are exactly equal; 63 allowed existing-node changes are enumerated in the JSON report.

## Serialized PB logits validation

Candidate bytes were written to disk, reloaded and compared with the original PB in independent TF1.15 CPU sessions. Threshold was fixed in advance at rtol=1e-4, atol=1e-4. All logits are finite float32. Random seed: 20261028.

Real images are the first sorted JPG/PNG file in each of 10 class directories. Original 256x256 RGB uint8 is decoded without resize, divided by 255 and normalized with mean [0.485,0.456,0.406] / std [0.229,0.224,0.225]. Per-image paths, image hashes and normalized-input hashes are in the JSON.

| Input | Max absolute error | Mean absolute error | Original / folded class | allclose |
| --- | ---: | ---: | --- | --- |
| zero | 7.15255737e-07 | 3.87430191e-07 | 5 / 5 | True |
| random_normal_0 | 7.15255737e-07 | 2.81631947e-07 | 5 / 5 | True |
| random_normal_1 | 5.66244125e-07 | 2.47359276e-07 | 5 / 5 | True |
| random_normal_2 | 9.53674316e-07 | 3.75509262e-07 | 5 / 5 | True |
| real_c0 | 9.53674316e-07 | 4.2244792e-07 | 0 / 0 | True |
| real_c1 | 4.76837158e-07 | 2.11596489e-07 | 1 / 1 | True |
| real_c2 | 8.64267349e-07 | 3.7997961e-07 | 8 / 8 | True |
| real_c3 | 7.15255737e-07 | 2.33948231e-07 | 3 / 3 | True |
| real_c4 | 2.20537186e-06 | 7.77840614e-07 | 4 / 4 | True |
| real_c5 | 4.76837158e-07 | 2.33948231e-07 | 5 / 5 | True |
| real_c6 | 9.53674316e-07 | 2.83122063e-07 | 6 / 6 | True |
| real_c7 | 5.96046448e-07 | 3.15904617e-07 | 7 / 7 | True |
| real_c8 | 2.2649765e-06 | 6.40749931e-07 | 8 / 8 | True |
| real_c9 | 2.50339508e-06 | 6.04987144e-07 | 9 / 9 | True |

- All 14 logits comparisons and all 294 local-fold comparisons pass; classes agree on 14/14 inputs.
- Overall maximum logits absolute error: 2.50339508057e-06; overall mean across all 140 logits: 3.85461109025e-07.
- Local tensor maximum absolute error: 0.00018310546875; all local elements meet the same rtol/atol criterion.
- Class agreement compares old and new models; it is not a dataset accuracy measurement. No full-dataset evaluation or Vitis-AI quantization/DPU compilation was performed.

## Quantization path update after validation

- Applied: `True`
- Quantizer script: `C:\WorkSpace\Z19-Onborad\MobileViT\compile_tools\1_tf_quantize.sh`
- Only change: `TF_NETWORK_PATH` now references `mobilevitv2_075_10cls_relu_deploy_b1_gnfold.pb`.
- Script SHA256 before: `C30D5187D333FAD02B860CAE7335B3DEADDBF031C26674884AD99EBFC0BC69FA`; after: `5FACCA88E0F6983473F733100B48C9D3CDC453102782631BFED94EC05F4562A1`.
- Copied PB: `C:\WorkSpace\Z19-Onborad\MobileViT\compile_tools\modle\driver\pb_file\mobilevitv2_075_10cls_relu_deploy_b1_gnfold.pb`; its hash equals the validated deployment hash above.
- Script backup: `C:\WorkSpace\Z19-Onborad\Model\keras_cv_attention_models\2board\artifacts\mobilevitv2_tf115\1_tf_quantize.before_gnfold.sh`. Original PB remains in place.
- Input/output nodes, input shape, calibration function and all other script settings are unchanged.

## Artifacts and reproduction

- Export/audit script: `2board/fold_mobilevitv2_gn_affine_pb.py` (run in existing tf115 interpreter; use --target with a fresh path to avoid overwriting).
- Full machine-readable audit: `mobilevitv2_075_10cls_relu_deploy_b1_gnfold_report.json`.
- Both logits arrays: `mobilevitv2_075_10cls_relu_deploy_b1_gnfold_validation_logits.npz`.
- Independent pure-inference inspection: `deploy_b1_gnfold_structure_audit.json`.
- Quantizer update script: `2board/update_mobilevitv2_gnfold_quantization.py` (prepare by default, --apply writes external compile_tools files).
- Quantizer update record: `quantization_script_update_gnfold.json`.
