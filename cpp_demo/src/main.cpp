#include <gst/gst.h>
#include <gst/rtsp-server/rtsp-server.h>
#include <gst/app/gstappsrc.h>
#include <gst/app/gstappsink.h>
#include <glib.h>
#include <stdio.h>
#include <signal.h>
#include <chrono>
#include <vector>
#include <string>
#include <mutex>
#include <condition_variable>
#include <opencv2/opencv.hpp>
#include "nvbuffer_cuda_processor.h"

#ifdef HAVE_JETSON_NVMM
#include <nvbufsurface.h>
#endif

// --- 配置参数 ---
static const char *DEFAULT_DEVICES[4] = {"/dev/video0", "/dev/video1", "/dev/video2", "/dev/video3"};
static const int IN_WIDTH = 1280;   // ROECS input width
static const int IN_HEIGHT = 1080;  // ROECS input height
static const int OUT_WIDTH = 1000;  // Stitching output width
static const int OUT_HEIGHT = 1000; // Stitching output height
static const int FPS = 30;

// --- 全局状态 ---
static GMainLoop *main_loop = nullptr;
static gboolean is_simulation = FALSE;
static std::string dataset_path = "../ROECS_dataset/full_texture";
static int current_frame_idx = 204;
static const int START_FRAME = 204;
static const int END_FRAME = 303;

// 显存资源
static uchar4* d_ins[4] = {nullptr}; // for Sim
static uchar4* d_out = nullptr;      // Common output buffer

// 线程同步 (用于 Real Mode: Capture Thread -> RTSP Thread)
static std::mutex g_frame_mutex;
static std::condition_variable g_frame_cv;
static bool g_has_new_frame = false;
static std::chrono::high_resolution_clock::time_point g_capture_time;

// 采集管线 (Orin Only)
static GstElement *g_capture_pipeline = nullptr;

// --- 信号处理 ---
static void signal_handler(int signum) {
    g_print("\nInterrupt signal (%d) received.\n", signum);
    if (main_loop) g_main_loop_quit(main_loop);
}

// =========================================================================
// 模式 A: 仿真模式 (读取文件)
// =========================================================================
static void process_simulation_frame() {
    char buf[256];
    const char* cam_names[] = {"F", "L", "B", "R"};
    std::vector<uchar4*> input_ptrs;

    for (int i = 0; i < 4; i++) {
        snprintf(buf, sizeof(buf), "%s/%06d %s.jpg", dataset_path.c_str(), current_frame_idx, cam_names[i]);
        cv::Mat img = cv::imread(buf);
        if (img.empty()) {
            current_frame_idx = START_FRAME; 
            snprintf(buf, sizeof(buf), "%s/%06d %s.jpg", dataset_path.c_str(), current_frame_idx, cam_names[i]);
            img = cv::imread(buf);
        }
        
        cv::Mat rgba;
        cv::cvtColor(img, rgba, cv::COLOR_BGR2RGBA);
        if (rgba.cols != IN_WIDTH || rgba.rows != IN_HEIGHT) cv::resize(rgba, rgba, cv::Size(IN_WIDTH, IN_HEIGHT));
        
        cudaMemcpy(d_ins[i], rgba.data, IN_WIDTH * IN_HEIGHT * sizeof(uchar4), cudaMemcpyHostToDevice);
        input_ptrs.push_back(d_ins[i]);
    }

    current_frame_idx++;
    if (current_frame_idx > END_FRAME) current_frame_idx = START_FRAME;

    auto start = std::chrono::high_resolution_clock::now();
    stitching_process(d_out, OUT_WIDTH * sizeof(uchar4), input_ptrs);
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    
    std::chrono::duration<double, std::milli> diff = end - start;
    static int frame_count = 0;
    if (++frame_count % 30 == 0) {
        printf("CUDA Stitching time: %.2f ms\n", diff.count());
    }
}

// =========================================================================
// 模式 B: 实车模式 (Jetson NVStreamMux 采集)
// =========================================================================
// =========================================================================
// 模式 B: 实车模式 (Jetson NVStreamMux 采集)
// =========================================================================
#ifdef HAVE_JETSON_NVMM
static GstFlowReturn on_capture_sample(GstAppSink *appsink, gpointer user_data) {
    GstSample *sample = gst_app_sink_pull_sample(appsink);
    if (!sample) return GST_FLOW_OK;

    // 从 nvstreammux 下游获取的是 NVMM Batch Buffer
    GstBuffer *batch_buffer = gst_sample_get_buffer(sample);
    
    // 我们必须构造一个临时的输出 Buffer 供 adapter 使用
    // 由于 nvbuffer_cuda_process_multi 需要一个 GstBuffer* out 来 map NvBufSurface
    // 在这里，我们可以复用 batch_buffer 的结构，或者更简单点：
    // 因为 nvbuffer_cuda_process_multi 实际上并没用到 out_buffer 的数据，只要它的 metadata 是对的。
    // 为了稳妥，我们在 Real Mode 初始化时应该创建一个专门的 NVMM Output Buffer。
    // 但为简化，我们暂时复用 d_out 的显存，通过一个 Fake GstBuffer 或者直接让 adapter 支持 d_out 指针。
    
    // 修正：我们刚才修改 nvbuffer_cuda_process_multi 实现时，它是从 out_buffer 提取 d_out。
    // 这里我们处于 Real Mode，理想流程是：
    // Capture -> NvStreamMux -> AppSink -> (Callback) -> CUDA Stitch -> d_out (GPU) -> Signal
    
    // 为了让代码通过，我们需要一个 "NVMM Output Buffer"。
    // 我们可以动态创建一个。
    // 但更好的做法是：修改 nvbuffer_cuda_process_multi 接口，让它接受 uchar4* d_out。
    // 这样我们就可以传我们在 main() 里 malloc 的 d_out。
    
    // 然而 adapter 已经在 .h 里定死了。我们得用复杂的 Buffer Wrap 吗？
    // 不，通过 NvBufSurfaceFromFd 可以... 这太复杂了。
    
    // 💡 最佳方案：既然 d_out 已经在 GPU 上，而且是 cudaMalloc 的 (Pitch 可能不等于 NVMM Pitch)。
    // 我们在 Capture Thread 里直接从 batch_buffer 获取 4 个输入指针，
    // 然后调用 stitching_process 输出到全局 d_out。
    // adapter 的 nvbuffer_cuda_process_multi 其实是给全 GStreamer 管道用的。
    // 在这里我们是 AppSink，我们已经拿到了 raw buffer。
    
    GstMapInfo map;
    gst_buffer_map(batch_buffer, &map, GST_MAP_READ);
    NvBufSurface *surf = (NvBufSurface *)map.data;
    
    // 此时 surf->numFilled 应该是 4
    if (NvBufSurfaceMap(surf, -1, -1, NVBUF_MAP_READ) == 0) {
        NvBufSurfaceSyncForDevice(surf, -1, -1);
        
        std::vector<uchar4*> input_ptrs;
        // 假设 nvstreammux 的 batch 顺序就是 camera 顺序 (0,1,2,3)
        // nvstreammux sink_0 -> cam 0, sink_1 -> cam 1 ...
        for (int i = 0; i < 4 && i < surf->numFilled; i++) {
            input_ptrs.push_back((uchar4*)surf->surfaceList[i].dataPtr);
        }
        
        if (input_ptrs.size() == 4) {
             // 这里的 Pitch 是问题。cudaMalloc 的 d_out 是紧凑的吗？
             // stitching_process 接受 out_pitch。
             // 对于 d_out (cudaMalloc), pitch = width * 4。
             auto start = std::chrono::high_resolution_clock::now();
             stitching_process(d_out, OUT_WIDTH * sizeof(uchar4), input_ptrs);
             cudaDeviceSynchronize();
             auto end = std::chrono::high_resolution_clock::now();
             
             std::chrono::duration<double, std::milli> diff = end - start;
             static int frame_count_real = 0;
             if (++frame_count_real % 30 == 0) {
                 printf("Real-mode CUDA Stitching time: %.2f ms\n", diff.count());
             }
             
             // 唤醒 RTSP 推流线程，并记录采集开始时间
             {
                 std::lock_guard<std::mutex> lock(g_frame_mutex);
                 g_has_new_frame = true;
                 g_capture_time = std::chrono::high_resolution_clock::now();
             }
             g_frame_cv.notify_one();
        }
        
        NvBufSurfaceUnMap(surf, -1, -1);
    }
    
    gst_buffer_unmap(batch_buffer, &map);
    gst_sample_unref(sample);
    return GST_FLOW_OK;
}
#endif

// =========================================================================
// RTSP Server 回调 (负责把 d_out 推出去)
// =========================================================================
static void on_rtsp_need_data(GstElement *appsrc, guint unused, gpointer user_data) {
    static GstClockTime timestamp = 0;

    auto e2e_start = std::chrono::high_resolution_clock::now();

    // 1. 生成或获取最新帧
    if (is_simulation) {
        process_simulation_frame();
    } else {
        // Real Mode: Wait for `on_capture_sample` to update d_out
        std::unique_lock<std::mutex> lock(g_frame_mutex);
        if (g_frame_cv.wait_for(lock, std::chrono::milliseconds(100), []{ return g_has_new_frame; })) {
            g_has_new_frame = false;
            e2e_start = g_capture_time; // Start latency from camera capture
        } else {
            // Timeout or no frame, push a black frame or just skip? 
            // For now, let's just use what's in d_out.
        }
    }

    // 2. 将 d_out (GPU) 拷贝回 CPU 发送给 RTSP (x264enc)
    GstBuffer *buffer = gst_buffer_new_allocate(nullptr, OUT_WIDTH * OUT_HEIGHT * 4, nullptr);
    GstMapInfo map;
    gst_buffer_map(buffer, &map, GST_MAP_WRITE);
    cudaMemcpy(map.data, d_out, OUT_WIDTH * OUT_HEIGHT * sizeof(uchar4), cudaMemcpyDeviceToHost);
    gst_buffer_unmap(buffer, &map);

    GST_BUFFER_PTS(buffer) = timestamp;
    GST_BUFFER_DURATION(buffer) = gst_util_uint64_scale_int(1, GST_SECOND, FPS);
    timestamp += GST_BUFFER_DURATION(buffer);

    GstFlowReturn ret;
    g_signal_emit_by_name(appsrc, "push-buffer", buffer, &ret);
    gst_buffer_unref(buffer);

    auto e2e_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> e2e_diff = e2e_end - e2e_start;
    
    static int e2e_frame_count = 0;
    if (++e2e_frame_count % 30 == 0) {
        printf("End-to-End Latency: %.2f ms\n", e2e_diff.count());
    }
}

static void media_configure_cb(GstRTSPMediaFactory *factory, GstRTSPMedia *media, gpointer user_data) {
    GstElement *element = gst_rtsp_media_get_element(media);
    GstElement *appsrc = gst_bin_get_by_name(GST_BIN(element), "mysrc");
    if (appsrc) {
        g_signal_connect(appsrc, "need-data", G_CALLBACK(on_rtsp_need_data), nullptr);
        gst_object_unref(appsrc);
    }
    gst_object_unref(element);
}

static GstRTSPMediaFactory* create_rtsp_factory() {
    GstRTSPMediaFactory *factory = gst_rtsp_media_factory_new();
    // 统一使用 appsrc，无论是 Sim 还是 Real。
    // 这解耦了 RTSP 传输层和图像生成层。
    char* launch_str = g_strdup_printf(
        "( appsrc name=mysrc is-live=true format=GST_FORMAT_TIME "
        "caps=\"video/x-raw,format=RGBA,width=%d,height=%d,framerate=%d/1\" ! "
        "videoconvert ! video/x-raw,format=I420 ! "
        "x264enc speed-preset=ultrafast tune=zerolatency bitrate=4000 ! "
        "rtph264pay name=pay0 pt=96 )",
        OUT_WIDTH, OUT_HEIGHT, FPS
    );

    gst_rtsp_media_factory_set_launch(factory, launch_str);
    g_free(launch_str);
    gst_rtsp_media_factory_set_shared(factory, TRUE);
    g_signal_connect(factory, "media-configure", G_CALLBACK(media_configure_cb), nullptr);
    return factory;
}

// =========================================================================
// 主程序
// =========================================================================
int main(int argc, char *argv[]) {
    for (int i = 1; i < argc; i++) {
        if (std::string(argv[i]) == "--sim") is_simulation = TRUE;
    }

    signal(SIGINT, signal_handler);
    gst_init(&argc, &argv);
    cuda_init();

    printf("Searching for LUT...\n");
    const char* lut_path = "surround_view.binary";
    if (fopen(lut_path, "r") == nullptr) lut_path = "stitching/surround_view.binary";
    if (fopen(lut_path, "r") == nullptr) lut_path = "../stitching/surround_view.binary";
    
    if (!stitching_init(lut_path, OUT_WIDTH, OUT_HEIGHT, IN_WIDTH, IN_HEIGHT)) {
        printf("Error: Failed to load LUT from %s\n", lut_path);
        return -1;
    }

    cudaMalloc(&d_out, OUT_WIDTH * OUT_HEIGHT * sizeof(uchar4));
    if (is_simulation) {
        for(int i=0; i<4; i++) cudaMalloc(&d_ins[i], IN_WIDTH * IN_HEIGHT * sizeof(uchar4));
    } else {
#ifdef HAVE_JETSON_NVMM
        // Real Mode: Build Capture Pipeline
        // v4l2src x 4 -> nvvideoconvert -> nvstreammux -> appsink
        // Note: For simplicity, we hardcode camera devices and nvstreammux settings.
        // nvstreammux batch-size=4 width=1280 height=1080 batched-push-timeout=40000
        
        GString *launch = g_string_new("");
        g_string_append_printf(launch, "nvstreammux name=mux batch-size=4 width=%d height=%d batched-push-timeout=40000 live-source=1 ! "
                                       "video/x-raw(memory:NVMM),format=RGBA ! "
                                       "appsink name=sink emit-signals=true max-buffers=1 drop=true ", IN_WIDTH, IN_HEIGHT);
        
        // Append sources
        for(int i=0; i<4; i++) {
             // Assuming devices are video0, 1, 2, 3
             // Need nvvidconv to ensure format is compatible with mux sink
             // v4l2src -> nvvidconv -> mux.sink_i
            g_string_append_printf(launch, 
                "v4l2src device=%s ! video/x-raw,width=%d,height=%d,framerate=%d/1 ! "
                "nvvidconv ! video/x-raw(memory:NVMM),format=RGBA ! "
                "mux.sink_%d ", 
                DEFAULT_DEVICES[i], IN_WIDTH, IN_HEIGHT, FPS, i);
        }
        
        printf("Launching Capture Pipeline: %s\n", launch->str);
        GError *err = nullptr;
        g_capture_pipeline = gst_parse_launch(launch->str, &err);
        g_string_free(launch, TRUE);
        
        if (!g_capture_pipeline || err) {
            printf("Error creating capture pipeline: %s\n", err ? err->message : "Unknown");
            return -1;
        }
        
        GstElement *sink = gst_bin_get_by_name(GST_BIN(g_capture_pipeline), "sink");
        g_signal_connect(sink, "new-sample", G_CALLBACK(on_capture_sample), nullptr);
        gst_object_unref(sink);
        
        gst_element_set_state(g_capture_pipeline, GST_STATE_PLAYING);
#else
        printf("Error: Real Mode requires Jetson NVMM support.\n");
        return -1;
#endif
    }

    main_loop = g_main_loop_new(nullptr, FALSE);
    GstRTSPServer *server = gst_rtsp_server_new();
    gst_rtsp_server_set_service(server, "8554");

    GstRTSPMountPoints *mounts = gst_rtsp_server_get_mount_points(server);
    gst_rtsp_mount_points_add_factory(mounts, "/live", create_rtsp_factory());
    g_object_unref(mounts);

    if (gst_rtsp_server_attach(server, nullptr) == 0) return -1;

    g_print("\n====================================\n");
    g_print("RTSP Server: %s Mode\n", is_simulation ? "SIMULATION" : "REAL CAMERA");
    g_print("Stream URL: rtsp://<IP>:8554/live\n");
    g_print("====================================\n\n");

    g_main_loop_run(main_loop);

    cuda_cleanup();
    if (d_out) cudaFree(d_out);
    for(int i=0; i<4; i++) if (d_ins[i]) cudaFree(d_ins[i]);
    
    return 0;
}
