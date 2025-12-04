# 一个简单的RTSP服务器，用于推流测试
#!/usr/bin/env python3
import gi
gi.require_version('Gst', '1.0')
gi.require_version('GstRtspServer', '1.0')
from gi.repository import Gst, GstRtspServer, GObject

Gst.init(None)

class RTSPServer:
    def __init__(self):
        self.server = GstRtspServer.RTSPServer()
        self.server.set_service("8554")  # RTSP 端口
        self.mounts = self.server.get_mount_points()

        # RTSP 推流管线
        # v4l2src -> nvvidconv -> nvv4l2h264enc -> rtph264pay
        self.factory = GstRtspServer.RTSPMediaFactory()
        self.factory.set_launch(
            "( v4l2src device=/dev/video1 ! "
            "video/x-raw,format=YUY2,width=1920,height=1080,framerate=30/1 ! "
            "nvvidconv ! video/x-raw(memory:NVMM),format=NV12 ! "
            "nvv4l2h264enc bitrate=10000000 preset=low-latency insert-sps-pps=true iframeinterval=15 ! "
            "rtph264pay name=pay0 pt=96 )"
        )
        self.factory.set_shared(True)
        self.factory.set_latency(50)
        self.mounts.add_factory("/camera", self.factory)
        self.server.attach(None)
        print("RTSP Server running at rtsp://<IP>:8554/camera")

if __name__ == "__main__":
    server = RTSPServer()
    loop = GObject.MainLoop()
    loop.run()
