#!/bin/bash
set -e

# 构建脚本
echo "===================================="
echo "Building Jetson RTSP C++ Demo"
echo "===================================="

# 进入构建目录
cd "$(dirname "$0")/.."

# 创建并进入build目录
mkdir -p build
cd build

# 运行CMake
echo "Running CMake..."
cmake ..

# 编译
echo "Compiling..."
make -j$(nproc)

echo ""
echo "===================================="
echo "Build completed successfully!"
echo "Executable: build/rtsp_demo"
echo "===================================="

