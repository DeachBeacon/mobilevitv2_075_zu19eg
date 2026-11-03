#!/usr/bin/env python3
"""
板端推理 + 结果叠加输出脚本
=============================
遍历指定目录中的图片，使用同级 deploy_best_fused 中的混合模型进行推理，
将预测类别、置信度、各阶段耗时等信息绘制在图片右上角，
然后将结果图片保存到 results_visual/ 目录，同时在外接显示器上逐张全屏展示。

用法（在板子上运行）：
    cd /path/to/Z19-Onborad/board
    python3 run_visual_board.py --image-dir ../data/calibration/jpegimages --output-dir results_visual \
        --labels-from-folders --display

依赖：
    - OpenCV (cv2)  — 板上 Python 已有
    - NumPy         — 板上 Python 已有
    - dnndk / n2cube — 板上 DPU 运行时
"""
from __future__ import print_function
import argparse
import json
import math
import os
import time
from pathlib import Path

# ── 自动设置显示环境变量（解决 SSH 终端下 Gtk: cannot open display 报错）──
if 'DISPLAY' not in os.environ or not os.environ['DISPLAY']:
    os.environ['DISPLAY'] = ':0'

if 'XAUTHORITY' not in os.environ:
    for auth_path in ['/root/.Xauthority', '/home/root/.Xauthority', '/home/alinx/.Xauthority']:
        if os.path.isfile(auth_path):
            os.environ['XAUTHORITY'] = auth_path
            break

import cv2
import numpy as np

# ── 自动选择部署模型目录 ──
# 优先使用新修改的 deploy_best_fused，支持命令行 --deploy-dir / --config 显式指定
import sys

SCRIPT_DIR = Path(__file__).resolve().parent

def _resolve_deploy_dir():
    # 1. 优先检查命令行是否指定了 --deploy-dir 或 --config
    for i, arg in enumerate(sys.argv):
        if arg == '--deploy-dir' and i + 1 < len(sys.argv):
            return Path(sys.argv[i + 1]).resolve()
        if arg.startswith('--deploy-dir='):
            return Path(arg.split('=', 1)[1]).resolve()
        if arg == '--config' and i + 1 < len(sys.argv):
            return Path(sys.argv[i + 1]).resolve().parent
        if arg.startswith('--config='):
            return Path(arg.split('=', 1)[1]).resolve().parent

    # 2. 优先使用同目录下的 deploy_best_fused
    best_fused = SCRIPT_DIR / 'deploy_best_fused'
    if best_fused.is_dir() and (best_fused / 'runtime_config.json').is_file():
        return best_fused

    # 3. 检查子目录 Visual_final/deploy_best_fused（兼容在父级 Visual 目录下调用的情况）
    visual_final = SCRIPT_DIR / 'Visual_final' / 'deploy_best_fused'
    if visual_final.is_dir() and (visual_final / 'runtime_config.json').is_file():
        return visual_final

    # 4. 回退检查当前工作目录或父目录是否有 runtime_config.json
    cwd = Path.cwd().resolve()
    if (cwd / 'runtime_config.json').is_file():
        return cwd
    if (cwd.parent / 'runtime_config.json').is_file():
        return cwd.parent

    # 5. 回退到旧的 deploy_in_board
    in_board = SCRIPT_DIR / 'deploy_in_board'
    if in_board.is_dir():
        return in_board

    return best_fused

DEPLOY_DIR = _resolve_deploy_dir()
RUNTIME_DIR = DEPLOY_DIR / 'runtime'
if str(RUNTIME_DIR) not in sys.path:
    sys.path.insert(0, str(RUNTIME_DIR))

from run_hybrid_board import HybridRunner, add_options
from dpu_runtime import load_image
from model_libraries import ensure_library_search_path

# ── 类别名称（c0–c9 对应 10 个类别） ──
CLASS_NAMES = {i: 'c{}'.format(i) for i in range(10)}

EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp'}


# ============================================================
#  第一部分：在图片上叠加推理结果
# ============================================================

def softmax(logits):
    """将 raw logits 转换为概率。"""
    logits = np.asarray(logits, dtype=np.float64)
    e = np.exp(logits - logits.max())
    return (e / e.sum()).astype(np.float64)


def draw_results_on_image(bgr_image, result, true_label=None, image_name='',
                          canvas_w=1920, canvas_h=1080, model_info=''):
    """
    创建一张 canvas_w x canvas_h 的画布：
      - 左侧：原始图片（等比缩放居中，保持比例）
      - 右侧：独立的深色信息面板，显示所有推理数据

    返回：合成后的 BGR 画布图片。
    """
    pred_class = result['class_index']
    logits = result['logits']
    probs = softmax(logits)
    seconds = result['seconds']

    # ── 画布布局：左 55% 放图片，右 45% 放信息面板 ──
    panel_ratio = 0.42
    img_area_w = int(canvas_w * (1.0 - panel_ratio))
    panel_area_w = canvas_w - img_area_w

    # 创建深灰画布
    canvas = np.full((canvas_h, canvas_w, 3), (30, 30, 30), dtype=np.uint8)

    # ── 左侧：放置图片 ──
    src_h, src_w = bgr_image.shape[:2]
    # 在左侧区域内等比缩放图片，上下左右留 margin
    margin = 30
    max_img_w = img_area_w - 2 * margin
    max_img_h = canvas_h - 2 * margin
    scale = min(max_img_w / src_w, max_img_h / src_h)
    new_w = int(src_w * scale)
    new_h = int(src_h * scale)
    resized = cv2.resize(bgr_image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

    # 居中放置
    y_offset = (canvas_h - new_h) // 2
    x_offset = (img_area_w - new_w) // 2
    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized

    # 图片边框
    cv2.rectangle(canvas, (x_offset - 1, y_offset - 1),
                  (x_offset + new_w, y_offset + new_h), (80, 80, 80), 1)

    # ── 右侧分割线 ──
    cv2.line(canvas, (img_area_w, 0), (img_area_w, canvas_h), (80, 80, 80), 2)

    # ── 右侧：信息面板 ──
    panel_x = img_area_w + 25  # 文字起始 x
    font = cv2.FONT_HERSHEY_SIMPLEX

    # 自适应字体大小：基于面板高度，确保所有内容都能放下
    # 预估总行数（含空行）约 24 行
    total_lines_estimate = 24
    line_height = max(22, min(40, (canvas_h - 60) // total_lines_estimate))
    font_scale = line_height / 35.0  # 35px 对应 scale=1.0
    font_scale = max(0.55, min(1.2, font_scale))
    thickness = max(1, int(round(font_scale)))
    title_scale = font_scale * 1.15
    title_thickness = max(1, int(round(title_scale)))

    y_cursor = 35  # 当前绘制的 y 坐标

    def put_text(text, color, scale=None, thick=None, dy=None):
        nonlocal y_cursor
        if dy is not None:
            y_cursor += dy
        else:
            y_cursor += line_height
        s = scale if scale else font_scale
        t = thick if thick else thickness
        cv2.putText(canvas, text, (panel_x, y_cursor), font, s, color, t, cv2.LINE_AA)

    def blank(pixels=None):
        nonlocal y_cursor
        y_cursor += (pixels if pixels else int(line_height * 0.5))

    # ── 标题区域 ──
    put_text('DPU Inference Result', (255, 255, 255), scale=title_scale, thick=title_thickness)
    # 标题下划线
    y_cursor += 8
    cv2.line(canvas, (panel_x, y_cursor), (canvas_w - 25, y_cursor), (100, 100, 100), 1)
    blank(5)

    # ── 模型信息 ──
    if model_info:
        put_text('Model: {}'.format(model_info), (100, 220, 255), scale=font_scale * 0.95)
        blank(2)

    # ── 文件名 ──
    put_text('File:', (160, 160, 160))
    put_text('  {}'.format(image_name), (255, 255, 255))
    blank()

    # ── 预测结果 ──
    confidence = probs[pred_class] * 100
    put_text('Predicted: {} ({:.1f}%)'.format(
        CLASS_NAMES.get(pred_class, str(pred_class)), confidence), (0, 255, 0),
        scale=font_scale * 1.1, thick=max(1, thickness + 1))

    # 真实标签
    if true_label is not None:
        is_correct = (pred_class == true_label)
        color = (0, 255, 0) if is_correct else (0, 0, 255)
        label_text = 'True:      {} | {}'.format(
            CLASS_NAMES.get(true_label, str(true_label)),
            'CORRECT' if is_correct else 'WRONG')
        put_text(label_text, color, scale=font_scale * 1.1, thick=max(1, thickness + 1))
    blank()

    # ── 分割线 ──
    y_cursor += 3
    cv2.line(canvas, (panel_x, y_cursor), (canvas_w - 25, y_cursor), (60, 60, 60), 1)
    blank(3)

    # ── 各阶段耗时 ──
    put_text('Timing (ms)', (200, 200, 200), scale=font_scale * 1.05)
    blank(3)

    stage_order = ['front_dpu', 'ps1', 'mv2_4', 'ps2', 'mv2_5', 'ps3']
    for stage in stage_order:
        if stage in seconds:
            ms_val = seconds[stage] * 1000
            put_text('  {:<12s} {:>8.2f}'.format(stage, ms_val), (220, 220, 220))
    blank(3)

    # DPU 汇总 / NumPy 汇总 / Pipeline 总耗时
    if 'dpu_with_transfer' in seconds:
        put_text('  DPU total    {:>8.2f}'.format(seconds['dpu_with_transfer'] * 1000), (100, 200, 255))
    if 'numpy_tail' in seconds:
        put_text('  NumPy tail   {:>8.2f}'.format(seconds['numpy_tail'] * 1000), (100, 200, 255))
    total_ms = sum(seconds.get(s, 0) for s in stage_order)
    put_text('  Pipeline     {:>8.2f}'.format(total_ms * 1000), (0, 255, 255))

    if 'total' in seconds:
        put_text('  Total(+pre)  {:>8.2f}'.format(seconds['total'] * 1000), (0, 220, 220))
    blank()

    # ── 分割线 ──
    y_cursor += 3
    cv2.line(canvas, (panel_x, y_cursor), (canvas_w - 25, y_cursor), (60, 60, 60), 1)
    blank(3)

    # ── Top-3 概率 ──
    put_text('Top-3 Probabilities', (200, 200, 200), scale=font_scale * 1.05)
    blank(3)
    top3 = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)[:3]
    rank_colors = [(0, 255, 255), (180, 255, 180), (200, 200, 200)]
    for rank, idx in enumerate(top3):
        bar_text = '  #{} {} : {:.2f}%'.format(rank + 1, CLASS_NAMES.get(idx, str(idx)), probs[idx] * 100)
        put_text(bar_text, rank_colors[rank])

        # 画一个小概率条
        bar_y = y_cursor - int(line_height * 0.3)
        bar_x_start = panel_x + int(panel_area_w * 0.55)
        bar_max_w = canvas_w - 25 - bar_x_start
        bar_w = max(1, int(bar_max_w * probs[idx]))
        bar_h_px = max(4, int(line_height * 0.4))
        cv2.rectangle(canvas, (bar_x_start, bar_y),
                      (bar_x_start + bar_w, bar_y + bar_h_px),
                      rank_colors[rank], cv2.FILLED)

    return canvas


def draw_summary_screen(stats, canvas_w=1920, canvas_h=1080):
    """
    在所有图片推理完成后，绘制高品质总结看板（1920x1080 FHD）：
      - 顶部：Banner 标题 + 平台信息
      - 左侧：模型配置与总体准确率卡片、各阶段耗时与 FPS 吞吐量卡片
      - 右侧：c0-c9 各类别准确率柱状图与统计、平均置信度卡片
      - 底部：操作提示文字
    返回 BGR 画布图片。
    """
    canvas = np.full((canvas_h, canvas_w, 3), (22, 22, 26), dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX

    # ── 顶部 Banner ──
    banner_h = int(canvas_h * 0.075)
    cv2.rectangle(canvas, (0, 0), (canvas_w, banner_h), (32, 32, 38), cv2.FILLED)
    cv2.line(canvas, (0, banner_h), (canvas_w, banner_h), (60, 60, 75), 2)
    cv2.putText(canvas, 'MobileViTv2 DPU Hybrid Pipeline - Inference Summary', (60, int(banner_h * 0.65)),
                font, 1.25, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, 'ALINX Z19 (Zynq UltraScale+)', (canvas_w - 480, int(banner_h * 0.65)),
                font, 0.85, (180, 180, 200), 2, cv2.LINE_AA)

    left_w = int(canvas_w * 0.52)
    cv2.line(canvas, (left_w, banner_h + 30), (left_w, canvas_h - 70), (50, 50, 65), 2)

    # ══════════════════════════════════════════
    #  左侧卡片 1：模型配置与总体准确率
    # ══════════════════════════════════════════
    card1_y1 = banner_h + 30
    card1_y2 = card1_y1 + 250
    cv2.rectangle(canvas, (60, card1_y1), (left_w - 40, card1_y2), (30, 30, 38), cv2.FILLED)
    cv2.rectangle(canvas, (60, card1_y1), (left_w - 40, card1_y2), (60, 60, 75), 1)

    cv2.putText(canvas, 'MODEL CONFIGURATION & OVERALL ACCURACY', (85, card1_y1 + 35),
                font, 0.75, (140, 200, 255), 2, cv2.LINE_AA)
    cv2.line(canvas, (85, card1_y1 + 48), (left_w - 65, card1_y1 + 48), (55, 55, 70), 1)

    model_tag = stats.get('model_tag', 'deploy_best_fused [fused]')
    cv2.putText(canvas, 'Model: {}'.format(model_tag), (85, card1_y1 + 85),
                font, 0.85, (240, 240, 240), 2, cv2.LINE_AA)
    tail_desc = stats.get('tail_impl', 'ps_tail_fused')
    cv2.putText(canvas, 'Tail: {} (fused GroupNorm + Attention)'.format(tail_desc), (85, card1_y1 + 120),
                font, 0.75, (170, 170, 170), 1, cv2.LINE_AA)

    total = stats.get('total_images', 0)
    success = stats.get('successful', 0)
    correct_count = stats.get('correct', 0)
    accuracy = stats.get('accuracy', 0.0)

    if stats.get('has_labels'):
        cv2.putText(canvas, 'Accuracy:', (85, card1_y1 + 175), font, 1.0, (200, 200, 200), 2, cv2.LINE_AA)
        acc_col = (0, 255, 0) if accuracy >= 85 else ((0, 255, 255) if accuracy >= 70 else (0, 100, 255))
        cv2.putText(canvas, '{:.2f}%'.format(accuracy), (250, card1_y1 + 175), font, 1.3, acc_col, 3, cv2.LINE_AA)
        cv2.putText(canvas, '({} / {} correct)'.format(correct_count, success), (440, card1_y1 + 175),
                    font, 0.85, (180, 180, 180), 1, cv2.LINE_AA)

        bar_x = 85
        bar_y = card1_y1 + 200
        bar_w = left_w - 40 - 85 - 25
        bar_h = 24
        cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (50, 50, 60), cv2.FILLED)
        fill_w = max(1, int(bar_w * accuracy / 100.0))
        cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), acc_col, cv2.FILLED)
    else:
        cv2.putText(canvas, 'Processed: {} / {} images'.format(success, total), (85, card1_y1 + 175),
                    font, 0.9, (0, 255, 255), 2, cv2.LINE_AA)

    # ══════════════════════════════════════════
    #  左侧卡片 2：各阶段耗时与 FPS 吞吐量
    # ══════════════════════════════════════════
    card2_y1 = card1_y2 + 30
    card2_y2 = canvas_h - 120
    cv2.rectangle(canvas, (60, card2_y1), (left_w - 40, card2_y2), (30, 30, 38), cv2.FILLED)
    cv2.rectangle(canvas, (60, card2_y1), (left_w - 40, card2_y2), (60, 60, 75), 1)

    cv2.putText(canvas, 'PERFORMANCE & LATENCY BREAKDOWN', (85, card2_y1 + 40),
                font, 0.75, (140, 200, 255), 2, cv2.LINE_AA)
    cv2.line(canvas, (85, card2_y1 + 53), (left_w - 65, card2_y1 + 53), (55, 55, 70), 1)

    avg_times = stats.get('avg_times', {})
    stage_defs = [
        ('front_dpu (DPU)', 'front_dpu', (220, 220, 220)),
        ('ps1       (NumPy fused)', 'ps1', (220, 220, 220)),
        ('mv2_4     (DPU)', 'mv2_4', (220, 220, 220)),
        ('ps2       (NumPy fused)', 'ps2', (220, 220, 220)),
        ('mv2_5     (DPU)', 'mv2_5', (220, 220, 220)),
        ('ps3       (NumPy fused)', 'ps3', (220, 220, 220)),
    ]

    ty = card2_y1 + 95
    for label, key, col in stage_defs:
        val_str = '{:>8.2f} ms'.format(avg_times.get(key, 0.0))
        cv2.putText(canvas, label, (95, ty), font, 0.75, col, 1, cv2.LINE_AA)
        cv2.putText(canvas, val_str, (left_w - 220, ty), font, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        ty += 42

    cv2.line(canvas, (85, ty - 8), (left_w - 65, ty - 8), (60, 60, 75), 1)
    ty += 28

    total_lat = avg_times.get('total', 0.0)
    fps_val = (1000.0 / total_lat) if total_lat > 0 else 0.0

    summaries = [
        ('DPU Total Latency', '{:>8.2f} ms'.format(avg_times.get('dpu_total', 0.0)), (100, 200, 255)),
        ('NumPy Tail Latency', '{:>8.2f} ms'.format(avg_times.get('numpy_tail', 0.0)), (100, 200, 255)),
        ('Pipeline Latency', '{:>8.2f} ms'.format(avg_times.get('pipeline', 0.0)), (0, 255, 255)),
        ('Total Latency (+preprocess)', '{:>8.2f} ms'.format(total_lat), (0, 220, 220)),
        ('Throughput (FPS)', '{:>8.2f} FPS'.format(fps_val), (0, 255, 120)),
    ]
    for label, val_str, col in summaries:
        cv2.putText(canvas, label, (95, ty), font, 0.8, col, 2, cv2.LINE_AA)
        cv2.putText(canvas, val_str, (left_w - 220, ty), font, 0.85, col, 2, cv2.LINE_AA)
        ty += 42

    # ══════════════════════════════════════════
    #  右侧卡片：各类别准确率与综合指标
    # ══════════════════════════════════════════
    rx = left_w + 40
    rw = canvas_w - rx - 60

    cv2.rectangle(canvas, (rx, card1_y1), (rx + rw, card2_y2), (30, 30, 38), cv2.FILLED)
    cv2.rectangle(canvas, (rx, card1_y1), (rx + rw, card2_y2), (60, 60, 75), 1)

    if stats.get('has_labels'):
        cv2.putText(canvas, 'PER-CLASS ACCURACY BREAKDOWN (c0 - c9)', (rx + 25, card1_y1 + 40),
                    font, 0.75, (140, 200, 255), 2, cv2.LINE_AA)
        cv2.line(canvas, (rx + 25, card1_y1 + 53), (rx + rw - 25, card1_y1 + 53), (55, 55, 70), 1)

        per_class = stats.get('per_class', {})
        cy = card1_y1 + 100
        bar_start_x = rx + 360
        bar_total_w = rw - 400

        best_classes = []
        for cls_idx in range(10):
            cls_name = 'c{}'.format(cls_idx)
            info = per_class.get(cls_idx, {'correct': 0, 'total': 0})
            cls_total = info['total']
            cls_correct = info['correct']
            cls_acc = (cls_correct / cls_total * 100) if cls_total > 0 else 0.0
            best_classes.append((cls_name, cls_acc))

            color = (0, 255, 0) if cls_acc >= 90 else ((0, 255, 255) if cls_acc >= 80 else (0, 160, 255))
            txt = '{}:  {:>2d} / {:<2d}  ({:>6.2f}%)'.format(cls_name, cls_correct, cls_total, cls_acc)
            cv2.putText(canvas, txt, (rx + 30, cy), font, 0.8, color, 2, cv2.LINE_AA)

            by = cy - 18
            bh = 20
            cv2.rectangle(canvas, (bar_start_x, by), (bar_start_x + bar_total_w, by + bh),
                          (50, 50, 60), cv2.FILLED)
            bw = max(1, int(bar_total_w * cls_acc / 100.0))
            cv2.rectangle(canvas, (bar_start_x, by), (bar_start_x + bw, by + bh),
                          color, cv2.FILLED)
            cy += 58

        cv2.line(canvas, (rx + 25, cy + 5), (rx + rw - 25, cy + 5), (55, 55, 70), 1)
        cy += 45
        avg_conf = stats.get('avg_confidence', 0.0)
        cv2.putText(canvas, 'Average Prediction Confidence: {:.2f}%'.format(avg_conf), (rx + 30, cy),
                    font, 0.85, (180, 255, 180), 2, cv2.LINE_AA)
        cy += 40
        best_sorted = sorted(best_classes, key=lambda x: x[1], reverse=True)[:3]
        best_str = ', '.join(['{} ({:.1f}%)'.format(n, a) for n, a in best_sorted])
        cv2.putText(canvas, 'Best Classes: {}'.format(best_str), (rx + 30, cy),
                    font, 0.75, (200, 200, 200), 1, cv2.LINE_AA)
    else:
        cv2.putText(canvas, 'PREDICTION SUMMARY', (rx + 25, card1_y1 + 40),
                    font, 0.75, (140, 200, 255), 2, cv2.LINE_AA)
        cv2.line(canvas, (rx + 25, card1_y1 + 53), (rx + rw - 25, card1_y1 + 53), (55, 55, 70), 1)
        cv2.putText(canvas, 'Unlabeled Dataset Mode', (rx + 30, card1_y1 + 120),
                    font, 0.9, (100, 220, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, 'To calculate ground truth accuracy:', (rx + 30, card1_y1 + 180),
                    font, 0.8, (180, 180, 180), 1, cv2.LINE_AA)
        cv2.putText(canvas, '  1. Organize images into c0-c9 folders', (rx + 30, card1_y1 + 225),
                    font, 0.8, (220, 220, 220), 1, cv2.LINE_AA)
        cv2.putText(canvas, '  2. Pass flag: --labels-from-folders', (rx + 30, card1_y1 + 270),
                    font, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
        avg_conf = stats.get('avg_confidence', 0.0)
        cv2.putText(canvas, 'Average Confidence: {:.2f}%'.format(avg_conf), (rx + 30, card1_y1 + 350),
                    font, 0.85, (180, 255, 180), 2, cv2.LINE_AA)

    # ── 底部提示文字 ──
    hint = 'Press any key or wait 30 seconds to auto-exit'
    hint_sz = cv2.getTextSize(hint, font, 0.75, 1)[0]
    hx = (canvas_w - hint_sz[0]) // 2
    cv2.putText(canvas, hint, (hx, canvas_h - 25), font, 0.75, (140, 140, 150), 1, cv2.LINE_AA)

    return canvas

# ============================================================
#  第二部分：全屏显示（外接显示器 via miniDP → DP 1.4）
# ============================================================

def find_display_for_fullscreen():
    """
    在 Zynq UltraScale+ 板上，miniDP 输出通常被 DRM/KMS 注册为显示设备。
    使用 modetest 或 xrandr 来确认外接显示器的状态和分辨率。
    此函数返回推荐的显示配置信息。
    """
    info = {
        'note': 'Z19 的 mini DisplayPort 支持 DP V1.2a，最高 4Kx2K@30fps',
        'cable': 'miniDP → DP 1.4 转接线（向下兼容 DP 1.2a）',
        'setup_commands': [
            '# 1. 检查 DRM 显示设备',
            'ls /sys/class/drm/',
            'cat /sys/class/drm/card*/status',
            '',
            '# 2. 如果使用 X11：',
            'export DISPLAY=:0',
            'xrandr --output DP-1 --auto  # 或 --mode 1920x1080',
            '',
            '# 3. 如果无桌面环境（framebuffer 方式）：',
            'cat /sys/class/graphics/fb0/virtual_size',
            'modetest -M xlnx -s <connector_id>:<mode>',
        ]
    }
    return info


def display_images_fullscreen(image_dir, delay_ms=3000, window_name='DPU Inference Results',
                              screen_w=1920, screen_h=1080):
    """
    使用 OpenCV 的 highgui 全屏窗口，在外接显示器上逐张展示结果图片。
    结果图片已经是完整的画布（图片+信息面板），直接全屏显示即可。

    参数：
        image_dir  : 包含叠加结果的图片目录
        delay_ms   : 每张图片停留时间（毫秒），按任意键翻页，按 'q'/ESC 退出
        window_name: 窗口名称
    """
    files = sorted(
        p for p in Path(image_dir).rglob('*')
        if p.is_file() and p.suffix.lower() in EXTENSIONS
    )
    if not files:
        print('[display] 没有找到结果图片: {}'.format(image_dir))
        return

    print('[display] 共 {} 张结果图片，按任意键翻页，按 q 或 ESC 退出'.format(len(files)))
    print('[display] 每张默认停留 {} 毫秒'.format(delay_ms))

    # ── 创建全屏窗口 ──
    try:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    except Exception as e:
        print('[display-error] 无法打开图形显示窗口: {}'.format(e))
        print('[display-error] 请在 SSH 终端运行: export DISPLAY=:0')
        return

    for idx, filepath in enumerate(files, 1):
        img = cv2.imread(str(filepath))
        if img is None:
            print('[display] 跳过无法读取: {}'.format(filepath))
            continue

        is_summary = (filepath.stem == 'summary_result')
        if not is_summary:
            # 在画布左下角加序号
            h, w = img.shape[:2]
            counter_text = '[{}/{}]  {}'.format(idx, len(files), filepath.stem)
            cv2.putText(img, counter_text, (15, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow(window_name, img)
        if is_summary:
            # 总结画面使用真实系统时钟控制 30 秒停留
            for _ in range(5):
                cv2.waitKey(1)
            t_start = time.time()
            t_total = 30.0
            last_sec = -1
            while True:
                elapsed = time.time() - t_start
                if elapsed >= t_total:
                    break
                rem = int(math.ceil(t_total - elapsed))
                if rem != last_sec and rem % 5 == 0:
                    print('[display] 总结画面展示中... 剩余 {} 秒 (按任意键可退出)'.format(rem))
                    last_sec = rem
                k = cv2.waitKey(100) & 0xFF
                if 0 < k < 255:
                    print('[display] 用户按键退出')
                    break
            break
        else:
            key = cv2.waitKey(delay_ms) & 0xFF
            if key == ord('q') or key == 27:  # q 或 ESC 退出
                print('[display] 用户退出')
                break
            elif key == ord('p'):  # p 暂停，再按任意键继续
                print('[display] 暂停，按任意键继续...')
                cv2.waitKey(0)

    cv2.destroyAllWindows()


# ============================================================
#  主流程
# ============================================================

def _auto_sync_runtime_config(config_path):
    """
    自动检测并修复 runtime_config.json 与 model/ 目录下 .so 文件的同步。
    当用户更换了新版内核（如从 front_cle 更换为 fronta_ctlb100）时，自动更新配置，避免报 Missing SO。
    """
    try:
        cfg = json.loads(config_path.read_text(encoding='utf-8'))
        model_dir = config_path.parent / 'model'
        changed = False

        # 检查并同步 front stage
        front_so = model_dir / ('libdpumodel' + cfg['front']['kernel'] + '.so')
        if not front_so.is_file():
            for so_file in sorted(model_dir.glob('*.so')):
                if 'front' in so_file.name:
                    from model_libraries import elf_kernel_names
                    kernels = elf_kernel_names(so_file.read_bytes())
                    if kernels:
                        new_kernel = kernels[0]
                        new_sha = hashlib.sha256(so_file.read_bytes()).hexdigest()
                        print('[auto-config] 检测到前段内核更新: {} -> {} ({})'.format(
                            cfg['front']['kernel'], new_kernel, so_file.name))
                        cfg['front']['kernel'] = new_kernel
                        cfg['front']['so_sha256'] = new_sha
                        if 'ctlb100' in new_kernel:
                            cfg['front']['output_node'] = 'stack3_block2_pre_2_conv_Conv2D_linear'
                        changed = True
                        break

        # 检查各 DPU stage
        for stage in cfg.get('stages', []):
            if stage.get('backend') == 'dpu':
                st_so = model_dir / ('libdpumodel' + stage['kernel'] + '.so')
                if not st_so.is_file():
                    for so_file in sorted(model_dir.glob('*.so')):
                        if stage['name'] in so_file.name:
                            from model_libraries import elf_kernel_names
                            kernels = elf_kernel_names(so_file.read_bytes())
                            if kernels:
                                new_kernel = kernels[0]
                                new_sha = hashlib.sha256(so_file.read_bytes()).hexdigest()
                                print('[auto-config] 检测到 {} 内核更新: {} -> {} ({})'.format(
                                    stage['name'], stage['kernel'], new_kernel, so_file.name))
                                stage['kernel'] = new_kernel
                                stage['so_sha256'] = new_sha
                                changed = True
                                break

        if changed:
            config_path.write_text(json.dumps(cfg, indent=2), encoding='utf-8')
            print('[auto-config] 已成功同步更新配置文件: {}'.format(config_path))
    except Exception as e:
        pass


def run_visual_batch(args):
    """
    主逻辑：遍历图片 → 推理 → 叠加结果 → 保存 → 可选实时显示。
    """
    image_dir = args.image_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # 收集图片
    files = sorted(
        p for p in image_dir.rglob('*')
        if p.is_file() and p.suffix.lower() in EXTENSIONS
    )
    if args.limit:
        files = files[:args.limit]
    if not files:
        raise ValueError('没有找到图片: {}'.format(image_dir))

    # 解析标签（从文件夹名 c0–c9）
    names_map = {'c{}'.format(i): i for i in range(10)}
    labels = []
    for path in files:
        parts = path.relative_to(image_dir).parts
        if args.labels_from_folders:
            if len(parts) < 2 or parts[0] not in names_map:
                raise ValueError('期望 c0–c9 子文件夹: {}'.format(path))
            labels.append(names_map[parts[0]])
        else:
            labels.append(None)

    print('=' * 60)
    print('  Visual Board Inference')
    print('  图片目录: {}'.format(image_dir))
    print('  图片数量: {}'.format(len(files)))
    print('  输出目录: {}'.format(output_dir))
    print('  显示模式: {}'.format('全屏外接显示器' if args.display else '仅保存'))
    print('=' * 60)

    # ── 如果启用实时显示，创建全屏窗口 ──
    window_name = 'DPU Inference Results'
    if args.display:
        try:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        except Exception as e:
            print('[display-error] 无法打开图形显示窗口: {}'.format(e))
            print('[display-error] 常见原因及排查方法：')
            print('[display-error]   1. SSH 终端未指定显示端口，请运行: export DISPLAY=:0')
            print('[display-error]   2. X 服务未运行，请检查: ps aux | grep -i xorg')
            print('[display-error]   3. 权限受限，请运行: xhost + local:')
            print('[display-error] 本次将自动切换为【仅保存结果图片模式】，不中断推理')
            args.display = False

    # ── 初始化推理引擎 ──
    config_path = args.config
    if not config_path.is_file():
        candidate = DEPLOY_DIR / 'runtime_config.json'
        if candidate.is_file():
            config_path = candidate
    _auto_sync_runtime_config(config_path)
    ensure_library_search_path(config_path)

    correct, total_success = 0, 0
    results_summary = []
    # 累计各阶段耗时用于计算平均值
    timing_accum = {}
    per_class_stats = {i: {'correct': 0, 'total': 0} for i in range(10)}
    confidence_sum = 0.0

    # 动态适配 HybridRunner（支持 deploy_best_fused 的 fused_tail 参数）
    runner_kwargs = dict(
        mv2_backend=args.mv2_backend,
        fast_tail=args.fast_tail,
        input_range_policy=getattr(args, 'input_range_policy', 'error'),
        trace_dpu_input=getattr(args, 'trace_dpu_input', False),
    )
    import inspect
    sig = inspect.signature(HybridRunner.__init__)
    if 'fused_tail' in sig.parameters:
        runner_kwargs['fused_tail'] = getattr(args, 'fuse', True)
        is_fused = getattr(args, 'fuse', True)
    else:
        is_fused = False

    tail_desc = 'fused' if is_fused else ('fast' if args.fast_tail else 'numpy')
    model_info = '{} [{}]'.format(DEPLOY_DIR.name, tail_desc)

    with HybridRunner(config_path, **runner_kwargs) as runner:

        # 保存配置信息（用于汇总报告，在 with 退出前获取）
        runner_config_snapshot = {
            'mv2_transform': runner.config.get('mv2_transform'),
            'front_kernel': runner.config['front']['kernel'],
            'dpu_kernels': [runner.config['front']['kernel']] + [
                s['kernel'] for s in runner.config['stages']
                if s.get('backend') == 'dpu' and args.mv2_backend == 'dpu'
            ],
        }

        # warmup
        if files:
            warmup_img = load_image(files[0])
            for _ in range(args.warmup):
                runner.predict(warmup_img)
            print('[warmup] 完成 {} 次热身推理 (模型: {})'.format(args.warmup, model_info))

        for idx, (filepath, label) in enumerate(zip(files, labels), 1):
            rel_path = filepath.relative_to(image_dir)
            image_name = rel_path.as_posix()

            try:
                # 读取原始图片（用于显示）
                bgr_original = cv2.imread(str(filepath))
                if bgr_original is None:
                    print('[{}/{}] 无法读取: {}'.format(idx, len(files), image_name))
                    continue

                # 预处理 + 推理
                t_start = time.perf_counter()
                preprocessed = load_image(filepath)
                t_preprocess = time.perf_counter() - t_start

                t_start = time.perf_counter()
                result = runner.predict(preprocessed)
                t_predict = time.perf_counter() - t_start

                # 将预处理时间也记录
                result['seconds']['preprocess'] = t_preprocess
                result['seconds']['total'] = t_preprocess + t_predict

                pred = result['class_index']
                probs = softmax(result['logits'])
                confidence = probs[pred] * 100
                confidence_sum += confidence

                # 累计各阶段耗时
                for k, v in result['seconds'].items():
                    if k not in timing_accum:
                        timing_accum[k] = 0.0
                    timing_accum[k] += v * 1000  # 转为 ms
                # 累计 pipeline 总耗时
                stage_order_keys = ['front_dpu', 'ps1', 'mv2_4', 'ps2', 'mv2_5', 'ps3']
                pipeline_ms = sum(result['seconds'].get(s, 0) for s in stage_order_keys) * 1000
                if 'pipeline' not in timing_accum:
                    timing_accum['pipeline'] = 0.0
                timing_accum['pipeline'] += pipeline_ms

                # 判断正确性
                is_correct = (pred == label) if label is not None else None
                if label is not None and is_correct:
                    correct += 1
                total_success += 1

                # 累计每类别统计
                if label is not None:
                    per_class_stats[label]['total'] += 1
                    if is_correct:
                        per_class_stats[label]['correct'] += 1

                # ── 将推理结果绘制到画布上（左图片 + 右信息面板） ──
                result_img = draw_results_on_image(bgr_original, result,
                                                    true_label=label,
                                                    image_name=image_name,
                                                    canvas_w=args.canvas_w,
                                                    canvas_h=args.canvas_h,
                                                    model_info=model_info)

                # ── 保存结果图片 ──
                out_subdir = output_dir / rel_path.parent
                out_subdir.mkdir(parents=True, exist_ok=True)
                out_path = out_subdir / (filepath.stem + '_result.jpg')
                cv2.imwrite(str(out_path), result_img, [cv2.IMWRITE_JPEG_QUALITY, 95])

                # ── 实时显示 ──
                if args.display:
                    # 左下角加序号
                    show_img = result_img.copy()
                    h_r, w_r = show_img.shape[:2]
                    counter = '[{}/{}]'.format(idx, len(files))
                    cv2.putText(show_img, counter, (15, h_r - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2, cv2.LINE_AA)
                    cv2.imshow(window_name, show_img)
                    # 默认 1ms：刷新 GUI 缓冲后立即进入下一张，一口气全自动跑完不等待
                    # 最后一张图片停留 1.5 秒，方便看清最终结果再平滑过渡到总结看板
                    if idx == len(files):
                        step_delay = max(1500, args.display_delay)
                    else:
                        step_delay = max(1, args.display_delay)
                    key = cv2.waitKey(step_delay) & 0xFF
                    if key == ord('q') or key == 27:
                        print('[display] 用户退出')
                        break
                    elif key == ord('p'):
                        print('[display] 暂停，按任意键继续...')
                        cv2.waitKey(0)

                # 控制台输出
                status = ''
                if label is not None:
                    status = ' ✓' if is_correct else ' ✗'
                print('[{}/{}] {} → c{} ({:.1f}%){} | {:.1f}ms'.format(
                    idx, len(files), image_name, pred, confidence, status,
                    result['seconds']['total'] * 1000))

                results_summary.append({
                    'image': image_name,
                    'predicted': pred,
                    'true_label': label,
                    'correct': is_correct,
                    'confidence': round(confidence, 2),
                    'total_ms': round(result['seconds']['total'] * 1000, 2),
                })

            except Exception as e:
                print('[{}/{}] 错误 {}: {}'.format(idx, len(files), image_name, e))
                continue

    # ── 绘制并显示/保存总结画面 ──
    if total_success > 0:
        # 计算平均耗时
        avg_times = {}
        for k, v in timing_accum.items():
            avg_times[k] = v / total_success
        # 额外计算 dpu_total 和 numpy_tail 的平均值
        if 'dpu_with_transfer' in avg_times:
            avg_times['dpu_total'] = avg_times.pop('dpu_with_transfer')

        accuracy_val = correct / total_success * 100 if args.labels_from_folders else 0.0

        summary_stats = {
            'model_tag': model_info,
            'tail_impl': tail_desc,
            'total_images': len(files),
            'successful': total_success,
            'correct': correct,
            'accuracy': accuracy_val,
            'has_labels': args.labels_from_folders,
            'avg_times': avg_times,
            'per_class': per_class_stats,
            'avg_confidence': confidence_sum / total_success if total_success else 0.0,
        }

        summary_canvas = draw_summary_screen(summary_stats,
                                             canvas_w=args.canvas_w,
                                             canvas_h=args.canvas_h)

        # 保存总结画面
        summary_img_path = output_dir / 'summary_result.jpg'
        cv2.imwrite(str(summary_img_path), summary_canvas, [cv2.IMWRITE_JPEG_QUALITY, 95])
        print('[summary] 总结画面已保存: {}'.format(summary_img_path))

        # 在显示器上展示总结画面，基于真实时钟严格停留 30 秒（按任意键可提前退出）
        if args.display:
            cv2.imshow(window_name, summary_canvas)
            print('[summary] 总结画面已全屏显示，保持 30 秒中（按任意键可提前退出）...')

            # 1. 彻底清空此前推理循环中残留的任何按键/事件缓冲
            for _ in range(10):
                cv2.waitKey(1)

            # 2. 真实系统时钟控制 30 秒倒计时，保证绝不因偶发事件闪退
            t_start = time.time()
            t_total = 30.0
            last_sec = -1
            while True:
                elapsed = time.time() - t_start
                if elapsed >= t_total:
                    break

                rem = int(math.ceil(t_total - elapsed))
                if rem != last_sec and rem % 5 == 0:
                    print('[summary] 总结画面展示中... 剩余 {} 秒 (按任意键可退出)'.format(rem))
                    last_sec = rem

                # 每次等待 100ms 处理 GUI 重绘
                k = cv2.waitKey(100) & 0xFF
                if 0 < k < 255:
                    print('[summary] 收到按键退出指令 (key={})，提前关闭'.format(k))
                    break

    if args.display:
        cv2.destroyAllWindows()

    # ── 输出汇总 ──
    print('\n' + '=' * 60)
    print('  推理完成')
    print('  模型: {}'.format(model_info))
    print('  成功: {}/{}'.format(total_success, len(files)))
    if args.labels_from_folders and total_success > 0:
        accuracy = correct / total_success * 100
        print('  准确率: {}/{} = {:.2f}%'.format(correct, total_success, accuracy))
    print('  结果图片已保存到: {}'.format(output_dir))
    print('=' * 60)

    # 保存 JSON 汇总
    summary = {
        'total_images': len(files),
        'successful': total_success,
        'correct': correct if args.labels_from_folders else None,
        'accuracy': round(correct / total_success * 100, 2) if args.labels_from_folders and total_success else None,
        'deploy_dir': str(DEPLOY_DIR),
        'model_tag': model_info,
        'mv2_backend': args.mv2_backend,
        'tail_impl': 'ps_tail_fused' if is_fused else ('ps_tail_fast' if args.fast_tail else 'ps_tail_numpy'),
        'mv2_transform': runner_config_snapshot['mv2_transform'],
        'dpu_kernels': runner_config_snapshot['dpu_kernels'],
        'results': results_summary,
    }
    summary_path = output_dir / 'visual_summary.json'
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    print('汇总已保存: {}'.format(summary_path))


def main():
    parser = argparse.ArgumentParser(
        description='板端推理 + 可视化输出 + 外接显示器展示',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  # 仅推理并保存结果图片
  python3 run_visual_board.py --image-dir ../data/calibration/jpegimages --output-dir results_visual --labels-from-folders

  # 推理 + 在外接显示器上实时全屏展示
  python3 run_visual_board.py --image-dir ../data/calibration/jpegimages --output-dir results_visual --labels-from-folders --display

  # 推理 + 显示，每张停留 5 秒
  python3 run_visual_board.py --image-dir ../data/calibration/jpegimages --output-dir results_visual --labels-from-folders --display --display-delay 5000

  # 仅查看已保存的结果图片（不重新推理）
  python3 run_visual_board.py --replay results_visual
        """)

    # 推理引擎选项（复用 run_hybrid_board 的参数）
    add_options(parser)

    # 针对 deploy_best_fused 优化：默认启用融合加速 (--fuse)
    parser.set_defaults(fuse=True)
    parser.add_argument('--no-fuse', dest='fuse', action='store_false',
                        help='禁用 GroupNorm/注意力融合（回退到 fast tail）')
    parser.add_argument('--deploy-dir', type=Path, default=DEPLOY_DIR,
                        help='模型部署根目录（默认自动定位 deploy_best_fused）')

    # 图片目录
    parser.add_argument('--image-dir', type=Path,
                        help='输入图片目录（如 ../data/calibration/jpegimages）')
    parser.add_argument('--output-dir', type=Path, default=Path('results_visual'),
                        help='结果图片输出目录（默认 results_visual）')
    parser.add_argument('--labels-from-folders', action='store_true',
                        help='从 c0–c9 子文件夹名推断真实标签')
    parser.add_argument('--warmup', type=int, default=1,
                        help='热身推理次数（默认 1）')
    parser.add_argument('--limit', type=int,
                        help='最多处理前 N 张图片')

    # 显示选项
    parser.add_argument('--display', action='store_true',
                        help='在外接显示器上实时全屏显示每张结果')
    parser.add_argument('--display-delay', '-display-delay', type=int, default=1,
                        help='每张图片显示停留时间（毫秒，默认 1 毫秒，即一口气快速流式跑完不等待；如需仔细查看可设为 5000）。按 q/ESC 退出，p 暂停')
    parser.add_argument('--canvas-w', type=int, default=1920,
                        help='输出画布宽度（默认 1920，匹配 FHD 显示器）')
    parser.add_argument('--canvas-h', type=int, default=1080,
                        help='输出画布高度（默认 1080，匹配 FHD 显示器）')

    # 回放模式
    parser.add_argument('--replay', type=Path,
                        help='回放模式：直接显示已有的结果图片目录（不重新推理）')

    args = parser.parse_args()

    if args.replay:
        # 纯回放模式：如果未显式提供延时，默认使用适合肉眼观看的 1500 毫秒
        replay_delay = args.display_delay if args.display_delay > 1 else 1500
        display_images_fullscreen(args.replay, delay_ms=replay_delay,
                                  screen_w=args.canvas_w, screen_h=args.canvas_h)
    else:
        if not args.image_dir:
            parser.error('需要指定 --image-dir')
        run_visual_batch(args)


if __name__ == '__main__':
    main()
