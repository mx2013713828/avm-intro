#include <gst/gst.h>
#include <gst/rtsp-server/rtsp-server.h>
#include <gst/allocators/gstdmabuf.h>
#include <glib.h>
#include <stdio.h>
#include <signal.h>
#include <chrono>
#include "nvbuffer_cuda_processor.h"

// 全局变量
static GMainLoop *main_loop = nullptr;
static gboolean signal_received = FALSE;
static int frame_count = 0;

// 摄像头参数
static const char *DEVICE = "/dev/video0";
static const int WIDTH = 1920;
static const int HEIGHT = 1080;
static const int FPS = 30;

// 信号处理函数
static void signal_handler(int signum) {
    g_print("\nInterrupt signal (%d) received.\n", signum);
    signal_received = TRUE;
    if (main_loop) {
        g_main_loop_quit(main_loop);
    }
}

// GStreamer消息总线回调
static gboolean bus_message_cb(GstBus *bus, GstMessage *message, gpointer user_data) {
    switch (GST_MESSAGE_TYPE(message)) {
        case GST_MESSAGE_ERROR: {
            GError *err;
            gchar *debug;
            gst_message_parse_error(message, &err, &debug);
            g_printerr("\n===========================================\n");
            g_printerr("ERROR from %s: %s\n", GST_OBJECT_NAME(message->src), err->message);
            g_printerr("Debug info: %s\n", debug ? debug : "none");
            g_printerr("===========================================\n");
            g_error_free(err);
            g_free(debug);
            g_main_loop_quit(main_loop);
            break;
        }
        case GST_MESSAGE_EOS:
            g_print("End-Of-Stream reached.\n");
            g_main_loop_quit(main_loop);
            break;
        case GST_MESSAGE_WARNING: {
            GError *warn;
            gchar *debug;
            gst_message_parse_warning(message, &warn, &debug);
            g_printerr("Warning from %s: %s\n", GST_OBJECT_NAME(message->src), warn->message);
            if (debug) {
                g_printerr("Debug info: %s\n", debug);
            }
            g_error_free(warn);
            g_free(debug);
            break;
        }
        case GST_MESSAGE_STATE_CHANGED: {
            GstState old_state, new_state;
            gst_message_parse_state_changed(message, &old_state, &new_state, nullptr);
            if (new_state == GST_STATE_PLAYING && old_state == GST_STATE_PAUSED) {
                g_print("Pipeline is now PLAYING\n");
            }
            break;
        }
        default:
            break;
    }
    return TRUE;
}

// Identity element的handoff回调 - 在这里进行CUDA处理
static void on_identity_handoff(GstElement *identity, GstBuffer *buffer, gpointer user_data) {
    static int cuda_success_count = 0;
    static int cuda_skip_count = 0;
    
    frame_count++;
    
    // 获取buffer的caps来获取尺寸信息
    GstPad *pad = gst_element_get_static_pad(identity, "sink");
    GstCaps *caps = gst_pad_get_current_caps(pad);
    gst_object_unref(pad);
    
    if (!caps) {
        return;
    }
    
    GstStructure *structure = gst_caps_get_structure(caps, 0);
    gint width, height;
    gst_structure_get_int(structure, "width", &width);
    gst_structure_get_int(structure, "height", &height);
    gst_caps_unref(caps);
    
    // 使用NvBuffer API进行CUDA处理
    bool cuda_processed = false;
    
    // 计时开始
    auto start_time = std::chrono::high_resolution_clock::now();

    // 直接使用NvBuffer API处理NVMM内存
    if (nvbuffer_cuda_process(buffer, width, height, 80)) {
        cuda_processed = true;
        cuda_success_count++;
    } else {
        cuda_skip_count++;
        if (frame_count == 1) {
            g_printerr("⚠️  NvBuffer CUDA processing failed (will continue without CUDA)\n");
        }
    }
    
    // 计时结束
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();

    // 每30帧打印一次统计
    if (frame_count % 30 == 0) {
        if (cuda_success_count > 0) {
            g_print("✓ Processed %d frames | CUDA: %d success, %d skipped | Last Frame Time: %.2f ms\n", 
                    frame_count, cuda_success_count, cuda_skip_count, duration / 1000.0f);
        } else {
            g_print("⚠ Processed %d frames | CUDA: NOT ACTIVE (all skipped)\n", frame_count);
        }
    }
}

// RTSP媒体配置回调
static void media_configure_cb(GstRTSPMediaFactory *factory, GstRTSPMedia *media, gpointer user_data) {
    g_print("RTSP client connected - Media configured\n");
    
    // 获取pipeline中的identity element并连接信号
    GstElement *element = gst_rtsp_media_get_element(media);
    GstElement *identity = gst_bin_get_by_name(GST_BIN(element), "cuda_hook");
    
    if (identity) {
        g_signal_connect(identity, "handoff", G_CALLBACK(on_identity_handoff), nullptr);
        g_print("CUDA processing hook installed\n");
        gst_object_unref(identity);
    }
    
    gst_object_unref(element);
}

// 客户端连接回调
static void client_connected_cb(GstRTSPServer *server, GstRTSPClient *client, gpointer user_data) {
    g_print("New RTSP client connected\n");
}

// 创建RTSP factory
static GstRTSPMediaFactory* create_rtsp_factory() {
    GstRTSPMediaFactory *factory;
    gchar *launch_str;

    factory = gst_rtsp_media_factory_new();

    // 完整的pipeline，在nvvidconv之后插入identity element用于CUDA处理
    // identity element是pass-through的，不会改变数据，只是给我们一个hook点
    launch_str = g_strdup_printf(
        "( v4l2src device=%s ! "
        "video/x-raw,format=YUY2,width=%d,height=%d,framerate=%d/1 ! "
        "nvvidconv ! "
        "video/x-raw(memory:NVMM),format=RGBA,width=%d,height=%d,framerate=%d/1 ! "
        "identity name=cuda_hook signal-handoffs=true ! "  // CUDA处理hook点
        "nvvidconv ! "
        "video/x-raw(memory:NVMM),format=NV12 ! "
        "nvv4l2h264enc bitrate=8000000 insert-sps-pps=true iframeinterval=30 preset-level=1 ! "
        "h264parse ! "
        "rtph264pay name=pay0 pt=96 config-interval=1 )",
        DEVICE, WIDTH, HEIGHT, FPS, WIDTH, HEIGHT, FPS
    );

    g_print("RTSP pipeline:\n%s\n\n", launch_str);

    gst_rtsp_media_factory_set_launch(factory, launch_str);
    g_free(launch_str);

    // 设置为shared，所有客户端共享同一个pipeline实例
    gst_rtsp_media_factory_set_shared(factory, TRUE);
    
    // 设置为可重用
    gst_rtsp_media_factory_set_eos_shutdown(factory, FALSE);
    
    // 连接media-configure信号
    g_signal_connect(factory, "media-configure", G_CALLBACK(media_configure_cb), nullptr);

    return factory;
}

int main(int argc, char *argv[]) {
    GstRTSPServer *server;
    GstRTSPMountPoints *mounts;
    GstRTSPMediaFactory *factory;

    // 注册信号处理
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);

    // 初始化GStreamer
    gst_init(&argc, &argv);

    // 初始化CUDA
    g_print("Initializing CUDA...\n");
    if (!cuda_init()) {
        g_printerr("Warning: CUDA initialization failed (will continue without CUDA processing)\n");
    } else {
        g_print("CUDA initialized successfully\n");
    }

    // 创建主循环
    main_loop = g_main_loop_new(nullptr, FALSE);

    // 创建RTSP服务器
    server = gst_rtsp_server_new();
    gst_rtsp_server_set_service(server, "8554");

    // 监听客户端连接
    g_signal_connect(server, "client-connected", G_CALLBACK(client_connected_cb), nullptr);

    // 创建并挂载RTSP factory
    factory = create_rtsp_factory();
    mounts = gst_rtsp_server_get_mount_points(server);
    gst_rtsp_mount_points_add_factory(mounts, "/live", factory);
    g_object_unref(mounts);

    // 附加服务器到默认上下文
    if (gst_rtsp_server_attach(server, nullptr) == 0) {
        g_printerr("Failed to attach RTSP server\n");
        cuda_cleanup();
        return -1;
    }

    g_print("\n====================================\n");
    g_print("RTSP Server with CUDA Processing\n");
    g_print("====================================\n");
    g_print("Stream URL: rtsp://<IP>:8554/live\n");
    g_print("Camera: %s (%dx%d @ %dfps)\n", DEVICE, WIDTH, HEIGHT, FPS);
    g_print("CUDA: Brightness enhancement (+80) [Extreme for comparison]\n");
    g_print("Platform: Jetson Orin (CUDA 11.4)\n");
    g_print("\nWaiting for RTSP clients...\n");
    g_print("Press Ctrl+C to stop\n");
    g_print("====================================\n\n");

    // 运行主循环
    g_main_loop_run(main_loop);

    // 清理资源
    g_print("\nShutting down...\n");
    
    g_object_unref(server);
    g_main_loop_unref(main_loop);
    
    cuda_cleanup();
    
    g_print("Cleanup complete\n");
    
    return 0;
}

