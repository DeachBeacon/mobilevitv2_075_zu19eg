#!/bin/bash
# ============================================================
# setup_dp_display.sh
# ============================================================
# Z19 开发板外接显示器配置脚本
#
# 硬件连接：
#   Z19 mini DisplayPort → miniDP 转 DP 1.4 线 → 显示器 DP 输入
#
# 说明：
#   Z19 的 mini DP 支持 DP V1.2a（向下兼容），miniDP 转 DP 1.4
#   的线完全兼容，会自动协商到 DP 1.2a 工作。
#
# 用法：
#   chmod +x setup_dp_display.sh
#   sudo ./setup_dp_display.sh
# ============================================================

set -e

echo "============================================"
echo "  Z19 DP 外接显示器配置"
echo "============================================"
echo ""

# ── 1. 检查 DRM 设备 ──
echo "[1/6] 检查 DRM 显示设备..."
if [ -d /sys/class/drm ]; then
    ls /sys/class/drm/
    echo ""
    for card in /sys/class/drm/card*-*/status; do
        if [ -f "$card" ]; then
            echo "  $card : $(cat $card)"
        fi
    done
else
    echo "  警告：未找到 /sys/class/drm/，可能需要加载 DRM 驱动"
fi
echo ""

# ── 2. 检查 framebuffer 设备 ──
echo "[2/6] 检查 Framebuffer 设备..."
if [ -d /sys/class/graphics ]; then
    for fb in /sys/class/graphics/fb*; do
        if [ -d "$fb" ]; then
            echo "  $(basename $fb):"
            [ -f "$fb/virtual_size" ] && echo "    分辨率: $(cat $fb/virtual_size)"
            [ -f "$fb/bits_per_pixel" ] && echo "    色深:   $(cat $fb/bits_per_pixel) bpp"
            [ -f "$fb/name" ] && echo "    名称:   $(cat $fb/name)"
        fi
    done
else
    echo "  未找到 framebuffer 设备"
fi
echo ""

# ── 3. 检查 Xilinx DRM 驱动 ──
echo "[3/6] 检查 Xilinx DisplayPort 驱动..."
if lsmod 2>/dev/null | grep -qi "xlnx\|zynqmp\|dp"; then
    lsmod | grep -i "xlnx\|zynqmp\|dp" || true
else
    echo "  未检测到 Xilinx DP 相关内核模块"
    echo "  尝试加载..."
    modprobe zynqmp_dp 2>/dev/null || modprobe xlnx_dp 2>/dev/null || \
        echo "  无法自动加载，请确认内核已编译 DP 支持"
fi
echo ""

# ── 4. 使用 modetest 检查连接器和模式 ──
echo "[4/6] 检查可用显示模式..."
if command -v modetest &>/dev/null; then
    echo "  使用 modetest 探测..."
    # 尝试 Xilinx DRM 驱动
    modetest -M xlnx 2>/dev/null | grep -A5 "Connectors:" || \
    modetest 2>/dev/null | grep -A5 "Connectors:" || \
    echo "  modetest 执行失败"
else
    echo "  modetest 未安装。"
    echo "  如需安装：apt install libdrm-tests 或从源码编译 libdrm"
fi
echo ""

# ── 5. 如果有 X11，配置 xrandr ──
echo "[5/6] 配置 X11 显示（如果可用）..."
if command -v xrandr &>/dev/null; then
    export DISPLAY=${DISPLAY:-:0}
    echo "  DISPLAY=$DISPLAY"
    echo "  当前 xrandr 输出："
    xrandr 2>/dev/null || echo "  无法连接 X 服务器"
    echo ""
    echo "  尝试自动配置 DP 输出..."
    # 常见的 DP 输出名称
    for output in DP-1 DP-0 DisplayPort-0 DisplayPort-1; do
        if xrandr 2>/dev/null | grep -q "^$output connected"; then
            echo "  检测到已连接的输出: $output"
            xrandr --output "$output" --auto 2>/dev/null && \
                echo "  ✓ 已启用 $output" || \
                echo "  启用失败，请手动配置"
            break
        fi
    done
else
    echo "  xrandr 不可用（无 X11 桌面环境）"
    echo "  将使用 framebuffer / DRM 直接输出"
fi
echo ""

# ── 6. 环境变量配置 ──
echo "[6/6] 设置 OpenCV 显示环境..."
echo ""
echo "  请在运行推理脚本前设置以下环境变量："
echo ""
echo "  # 如果使用 X11 桌面环境："
echo "  export DISPLAY=:0"
echo "  export QT_QPA_PLATFORM=xcb"
echo ""
echo "  # 如果使用纯 framebuffer（无桌面环境）："
echo "  export QT_QPA_PLATFORM=linuxfb:fb=/dev/fb0"
echo "  # 或者使用 eglfs（推荐，性能更好）："
echo "  export QT_QPA_PLATFORM=eglfs"
echo "  export QT_QPA_EGLFS_KMS_CONFIG=/tmp/kms_config.json"
echo ""

# 生成 KMS 配置文件（用于无桌面环境的直接显示）
cat > /tmp/kms_config.json << 'EOF'
{
  "device": "/dev/dri/card0",
  "outputs": [
    {
      "name": "DP-1",
      "mode": "preferred"
    }
  ]
}
EOF
echo "  已生成 KMS 配置: /tmp/kms_config.json"

echo ""
echo "============================================"
echo "  配置完成！"
echo ""
echo "  接下来运行推理+显示脚本（使用 deploy_best_fused 模型）："
echo "    cd /path/to/Z19-Onborad/board"
echo "    python3 run_visual_board.py \\"
echo "      --image-dir ../data/calibration/jpegimages \\"
echo "      --output-dir results_visual \\"
echo "      --labels-from-folders --display"
echo "============================================"
