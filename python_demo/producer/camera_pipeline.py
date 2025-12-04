# camera_pipeline.py
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst

Gst.init(None)

def build_camera_pipeline(device='/dev/video0', width=640, height=480, fps=30):
    """返回已经创建但未启动的 camera pipeline 与 appsink 元素"""
    pipeline_str = (
        f"v4l2src device={device} ! "
        f"video/x-raw,format=YUY2,width={width},height={height},framerate={fps}/1 ! "
        "nvvidconv ! "
        f"video/x-raw(memory:NVMM),format=NV12,width={width},height={height},framerate={fps}/1 ! "
        "appsink name=mysink emit-signals=true sync=false max-buffers=1 drop=true"
    )
    pipeline = Gst.parse_launch(pipeline_str)
    appsink = pipeline.get_by_name('mysink')
    return pipeline, appsink
