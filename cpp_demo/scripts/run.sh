#!/bin/bash
set -e

# 运行脚本
echo "===================================="
echo "RTSP Server with CUDA Processing"
echo "===================================="

# 进入项目目录
cd "$(dirname "$0")/.."

# 检查是否已编译
if [ ! -f "build/rtsp_demo" ]; then
    echo "Error: Executable not found. Please run build.sh first."
    exit 1
fi

# 获取本机IP地址
IP_ADDR=$(hostname -I | awk '{print $1}')

echo ""
echo "Server will start on: rtsp://$IP_ADDR:8554/live"
echo ""
echo "Test with:"
echo "  ffplay rtsp://$IP_ADDR:8554/live"
echo "  vlc rtsp://$IP_ADDR:8554/live"
echo ""
echo "Press Ctrl+C to stop"
echo "===================================="
echo ""

# 运行程序
./build/rtsp_demo

