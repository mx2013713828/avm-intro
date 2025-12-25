#include <gst/gst.h>
#include <gst/rtsp-server/rtsp-server.h>
#include <gst/app/gstappsrc.h>
#include <glib.h>
#include <stdio.h>
#include <signal.h>
#include <chrono>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include "nvbuffer_cuda_processor.h"

// --- Constants ---
static const char *DEFAULT_DEVICE = "/dev/video0";
static const int IN_WIDTH = 1280;   // ROECS input width
static const int IN_HEIGHT = 1080;  // ROECS input height
static const int OUT_WIDTH = 1000;  // Stitching output width
static const int OUT_HEIGHT = 1000; // Stitching output height
static const int FPS = 30;

// --- Global State ---
static GMainLoop *main_loop = nullptr;
static gboolean is_simulation = FALSE;
static std::string dataset_path = "ROECS_dataset/full_texture";
static int current_frame_idx = 204;
static const int START_FRAME = 204;
static const int END_FRAME = 303;

// GPU buffers for simulation
static uchar4* d_ins[4] = {nullptr};
static uchar4* d_out = nullptr;

// --- Signal Handling ---
static void signal_handler(int signum) {
    g_print("\nInterrupt signal (%d) received.\n", signum);
    if (main_loop) g_main_loop_quit(main_loop);
}

// --- AppSrc Callbacks (Simulation Mode) ---
static void on_need_data(GstElement *appsrc, guint unused, gpointer user_data) {
    static GstClockTime timestamp = 0;
    char buf[256];
    const char* cam_names[] = {"F", "L", "B", "R"};
    std::vector<uchar4*> input_ptrs;

    // Load and upload frames
    for (int i = 0; i < 4; i++) {
        snprintf(buf, sizeof(buf), "%s/%06d %s.jpg", dataset_path.c_str(), current_frame_idx, cam_names[i]);
        cv::Mat img = cv::imread(buf);
        if (img.empty()) {
            current_frame_idx = START_FRAME; // Re-attempt or loop
            snprintf(buf, sizeof(buf), "%s/%06d %s.jpg", dataset_path.c_str(), current_frame_idx, cam_names[i]);
            img = cv::imread(buf);
        }
        
        cv::Mat rgba;
        cv::cvtColor(img, rgba, cv::COLOR_BGR2RGBA);
        
        // Upload to GPU
        // Ensure image size matches what Surrounder expects
        if (rgba.cols != IN_WIDTH || rgba.rows != IN_HEIGHT) {
            cv::resize(rgba, rgba, cv::Size(IN_WIDTH, IN_HEIGHT));
        }
        
        cudaMemcpy(d_ins[i], rgba.data, IN_WIDTH * IN_HEIGHT * sizeof(uchar4), cudaMemcpyHostToDevice);
        input_ptrs.push_back(d_ins[i]);
    }

    current_frame_idx++;
    if (current_frame_idx > END_FRAME) current_frame_idx = START_FRAME;

    // Process stitching
    stitching_process(d_out, OUT_WIDTH * sizeof(uchar4), input_ptrs);

    // Create GStreamer buffer
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
}

static void media_configure_cb(GstRTSPMediaFactory *factory, GstRTSPMedia *media, gpointer user_data) {
    if (is_simulation) {
        GstElement *element = gst_rtsp_media_get_element(media);
        GstElement *appsrc = gst_bin_get_by_name(GST_BIN(element), "mysrc");
        if (appsrc) {
            g_signal_connect(appsrc, "need-data", G_CALLBACK(on_need_data), nullptr);
            gst_object_unref(appsrc);
        }
        gst_object_unref(element);
    } else {
        // Real camera: Connect to existing identity hook
        GstElement *element = gst_rtsp_media_get_element(media);
        GstElement *identity = gst_bin_get_by_name(GST_BIN(element), "cuda_hook");
        if (identity) {
            // Re-use current on_identity_handoff logic
            // Add callback if needed, but for now we focus on SIM
            gst_object_unref(identity);
        }
        gst_object_unref(element);
    }
}

static GstRTSPMediaFactory* create_factory() {
    GstRTSPMediaFactory *factory = gst_rtsp_media_factory_new();
    gchar *launch_str;

    if (is_simulation) {
        launch_str = g_strdup_printf(
            "( appsrc name=mysrc is-live=true format=GST_FORMAT_TIME "
            "caps=\"video/x-raw,format=RGBA,width=%d,height=%d,framerate=%d/1\" ! "
            "videoconvert ! video/x-raw,format=I420 ! "
            "x264enc speed-preset=ultrafast tune=zerolatency bitrate=4000 ! "
            "rtph264pay name=pay0 pt=96 )",
            OUT_WIDTH, OUT_HEIGHT, FPS
        );
    } else {
        launch_str = g_strdup_printf(
            "( v4l2src device=%s ! "
            "video/x-raw,format=YUY2,width=%d,height=%d,framerate=%d/1 ! "
            "nvvidconv ! video/x-raw(memory:NVMM),format=RGBA ! "
            "identity name=cuda_hook signal-handoffs=true ! "
            "nvvidconv ! video/x-raw(memory:NVMM),format=NV12 ! "
            "nvv4l2h264enc bitrate=4000000 ! h264parse ! "
            "rtph264pay name=pay0 pt=96 )",
            DEFAULT_DEVICE, IN_WIDTH, IN_HEIGHT, FPS
        );
    }

    gst_rtsp_media_factory_set_launch(factory, launch_str);
    g_free(launch_str);
    gst_rtsp_media_factory_set_shared(factory, TRUE);
    g_signal_connect(factory, "media-configure", G_CALLBACK(media_configure_cb), nullptr);
    return factory;
}

int main(int argc, char *argv[]) {
    for (int i = 1; i < argc; i++) {
        if (std::string(argv[i]) == "--sim") is_simulation = TRUE;
    }

    signal(SIGINT, signal_handler);
    gst_init(&argc, &argv);
    cuda_init();

    // Prepare simulation resources
    if (is_simulation) {
        printf("Searching for LUT...\n");
        const char* lut_path = "surround_view.binary";
        if (fopen(lut_path, "r") == nullptr) lut_path = "stitching/surround_view.binary";
        
        if (!stitching_init(lut_path, OUT_WIDTH, OUT_HEIGHT, IN_WIDTH, IN_HEIGHT)) {
            printf("Error: Failed to load LUT from %s\n", lut_path);
            return -1;
        }

        cudaMalloc(&d_out, OUT_WIDTH * OUT_HEIGHT * sizeof(uchar4));
        for(int i=0; i<4; i++) cudaMalloc(&d_ins[i], IN_WIDTH * IN_HEIGHT * sizeof(uchar4));
    }

    main_loop = g_main_loop_new(nullptr, FALSE);
    GstRTSPServer *server = gst_rtsp_server_new();
    gst_rtsp_server_set_service(server, "8554");

    GstRTSPMediaFactory *factory = create_factory();
    GstRTSPMountPoints *mounts = gst_rtsp_server_get_mount_points(server);
    gst_rtsp_mount_points_add_factory(mounts, "/live", factory);
    g_object_unref(mounts);

    if (gst_rtsp_server_attach(server, nullptr) == 0) return -1;

    g_print("\n====================================\n");
    g_print("RTSP Server: %s Mode\n", is_simulation ? "SIMULATION" : "REAL CAMERA");
    g_print("Stream URL: rtsp://<IP>:8554/live\n");
    if (is_simulation) g_print("Dataset: %s\n", dataset_path.c_str());
    g_print("====================================\n\n");

    g_main_loop_run(main_loop);

    cuda_cleanup();
    if (d_out) cudaFree(d_out);
    for(int i=0; i<4; i++) if (d_ins[i]) cudaFree(d_ins[i]);
    
    return 0;
}
