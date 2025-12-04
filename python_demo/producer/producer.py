# producer.py
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib

from camera_pipeline import build_camera_pipeline
from cuda_processor import cuda_process_dmabuf

Gst.init(None)

# 摄像头参数（可替换为 config 读取）
DEVICE = '/dev/video0'
WIDTH = 640
HEIGHT = 480
FPS = 30

# 创建 camera pipeline
cam_pipeline, appsink = build_camera_pipeline(DEVICE, WIDTH, HEIGHT, FPS)

# appsrc 将由 rtsp_server 创建并注入；这里我们创建一个占位 appsrc，
# 然后 producer 会把处理后的 NVMM buffer push 到该 appsrc。
appsrc = None  # rtsp_server 会设置实际的 appsrc 引用


def set_appsrc(ref):
    global appsrc
    appsrc = ref


def on_new_sample(sink, user_data):
    """appsink 的回调，接收 NVMM buffer（dmabuf fd），在 GPU 上处理后 push 给 appsrc"""
    global appsrc
    sample = sink.emit('pull-sample')
    if not sample:
        return Gst.FlowReturn.ERROR

    buf = sample.get_buffer()
    # 获取 caps 以读尺寸
    caps = sample.get_caps()
    s = caps.get_structure(0)
    width = s.get_value('width')
    height = s.get_value('height')

    # 取得 dmabuf fd
    # buf.peek_memory(0) 返回 Gst.Memory，使用 get_fd() 来获取 fd（某些系统 API 名不同）
    mem = buf.peek_memory(0)
    try:
        ok, fd = mem.get_fd()
    except AttributeError:
        # 兼容：某些 GStreamer 绑定为不同方法
        fd = None
        print('Cannot get fd from memory; ensure gst supports dmabuf')

    if fd is None:
        return Gst.FlowReturn.ERROR

    # GPU 处理（你的 TensorRT 替换点）
    success = cuda_process_dmabuf(fd, width, height, brighten_value=40)
    if not success:
        print('cuda process failed')

    # 推送原始 NVMM memory 到 appsrc（零拷贝）
    if appsrc is not None:
        # 注意：这里直接把 sample 的 buffer（引用 NVMM memory）送入 appsrc，保证零拷贝
        outbuf = buf.copy()
        appsrc.emit('push-buffer', outbuf)

    return Gst.FlowReturn.OK


if __name__ == '__main__':
    # 由 rtsp_server 启动前，producer 可先启动 camera
    cam_pipeline.set_state(Gst.State.PLAYING)
    appsink.connect('new-sample', on_new_sample, None)

    print('Producer running, waiting for RTSP appsrc to be attached...')
    loop = GLib.MainLoop()
    loop.run()
