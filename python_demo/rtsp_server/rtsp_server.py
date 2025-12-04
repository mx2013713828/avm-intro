# rtsp_server.py
import gi
gi.require_version('Gst', '1.0')
gi.require_version('GstRtspServer', '1.0')
from gi.repository import Gst, GstRtspServer, GObject

# producer 里的 set_appsrc 用于把 appsrc 引用注入 producer
import sys
sys.path.append('../producer')
from producer import producer as prod_mod  # 以包形式导入

Gst.init(None)

class NVMMFactory(GstRtspServer.RTSPMediaFactory):
    def __init__(self, width, height, fps, producer_module):
        super().__init__()
        self.width = width
        self.height = height
        self.fps = fps
        self.producer = producer_module
        self.set_shared(True)

    def do_create_element(self, url):
        # 建立 appsrc -> nvv4l2h264enc -> rtph264pay pipeline
        launch = (
            f"appsrc name=src is-live=true block=true format=time caps=video/x-raw(memory:NVMM),format=NV12,width={self.width},height={self.height},framerate={self.fps}/1 ! "
            "nvv4l2h264enc bitrate=4000000 insert-sps-pps=true iframeinterval=30 ! "
            "rtph264pay name=pay0 pt=96"
        )
        return Gst.parse_launch(launch)

    def do_configure(self, rtsp_media):
        element = rtsp_media.get_element()
        appsrc = element.get_child_by_name('src')
        # 将 appsrc 引用传回 producer
        self.producer.set_appsrc(appsrc)
        # 现在 appsrc 已注入，producer 的 appsink 回调会 push-buffer 到此 appsrc


if __name__ == '__main__':
    server = GstRtspServer.RTSPServer()
    factory = NVMMFactory(640, 480, 30, prod_mod)
    mounts = server.get_mount_points()
    mounts.add_factory('/live', factory)
    server.attach(None)
    print('RTSP server running at rtsp://192.168.0.103:8554/live')
    loop = GObject.MainLoop()
    loop.run()
