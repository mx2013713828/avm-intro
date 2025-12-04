#!/bin/bash
set -e

# 启动 RTSP server（后台）
python3 rtsp_server/rtsp_server.py &
RTSP_PID=$!
sleep 1

# 启动 producer
python3 producer/producer.py

# 当 producer 退出，杀掉 server
kill $RTSP_PID || true
