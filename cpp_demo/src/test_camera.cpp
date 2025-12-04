#include <gst/gst.h>
#include <gst/app/gstappsink.h>
#include <glib.h>
#include <stdio.h>
#include <signal.h>

static GMainLoop *main_loop = nullptr;
static gboolean signal_received = FALSE;
static int frame_count = 0;

static void signal_handler(int signum) {
    g_print("Interrupt signal received.\n");
    signal_received = TRUE;
    if (main_loop) {
        g_main_loop_quit(main_loop);
    }
}

static GstFlowReturn on_new_sample(GstAppSink *sink, gpointer user_data) {
    GstSample *sample;
    
    frame_count++;
    if (frame_count % 30 == 0) {
        g_print("Received %d frames\n", frame_count);
    }
    
    sample = gst_app_sink_pull_sample(sink);
    if (!sample) {
        g_printerr("Failed to pull sample\n");
        return GST_FLOW_ERROR;
    }
    
    gst_sample_unref(sample);
    return GST_FLOW_OK;
}

static gboolean bus_message_cb(GstBus *bus, GstMessage *message, gpointer user_data) {
    switch (GST_MESSAGE_TYPE(message)) {
        case GST_MESSAGE_ERROR: {
            GError *err;
            gchar *debug;
            gst_message_parse_error(message, &err, &debug);
            g_printerr("ERROR from %s: %s\n", GST_OBJECT_NAME(message->src), err->message);
            g_printerr("Debug: %s\n", debug ? debug : "none");
            g_error_free(err);
            g_free(debug);
            g_main_loop_quit(main_loop);
            break;
        }
        case GST_MESSAGE_EOS:
            g_print("End-Of-Stream\n");
            g_main_loop_quit(main_loop);
            break;
        case GST_MESSAGE_STATE_CHANGED: {
            GstState old_state, new_state;
            gst_message_parse_state_changed(message, &old_state, &new_state, nullptr);
            g_print("State: %s -> %s\n",
                    gst_element_state_get_name(old_state),
                    gst_element_state_get_name(new_state));
            break;
        }
        default:
            break;
    }
    return TRUE;
}

int main(int argc, char *argv[]) {
    GstElement *pipeline, *appsink;
    GstBus *bus;
    gchar *pipeline_str;
    
    signal(SIGINT, signal_handler);
    gst_init(&argc, &argv);
    
    // 创建pipeline
    pipeline_str = g_strdup_printf(
        "v4l2src device=/dev/video1 ! "
        "video/x-raw,format=YUY2,width=1920,height=1080,framerate=30/1 ! "
        "nvvidconv ! "
        "video/x-raw(memory:NVMM),format=NV12,width=1920,height=1080,framerate=30/1 ! "
        "appsink name=sink emit-signals=false sync=false max-buffers=1 drop=true"
    );
    
    g_print("Pipeline: %s\n\n", pipeline_str);
    
    pipeline = gst_parse_launch(pipeline_str, nullptr);
    g_free(pipeline_str);
    
    if (!pipeline) {
        g_printerr("Failed to create pipeline\n");
        return -1;
    }
    
    // 获取appsink
    appsink = gst_bin_get_by_name(GST_BIN(pipeline), "sink");
    if (!appsink) {
        g_printerr("Failed to get appsink\n");
        gst_object_unref(pipeline);
        return -1;
    }
    
    // 设置回调
    GstAppSinkCallbacks callbacks = {
        nullptr,
        nullptr,
        on_new_sample,
        nullptr
    };
    gst_app_sink_set_callbacks(GST_APP_SINK(appsink), &callbacks, nullptr, nullptr);
    
    // 添加消息总线
    bus = gst_element_get_bus(pipeline);
    gst_bus_add_watch(bus, bus_message_cb, nullptr);
    gst_object_unref(bus);
    
    // 启动pipeline
    g_print("Starting camera test...\n");
    if (gst_element_set_state(pipeline, GST_STATE_PLAYING) == GST_STATE_CHANGE_FAILURE) {
        g_printerr("Failed to start pipeline\n");
        gst_object_unref(pipeline);
        return -1;
    }
    
    // 运行主循环
    main_loop = g_main_loop_new(nullptr, FALSE);
    g_print("Running... Press Ctrl+C to stop\n\n");
    g_main_loop_run(main_loop);
    
    // 清理
    g_print("\nCleaning up...\n");
    gst_element_set_state(pipeline, GST_STATE_NULL);
    gst_object_unref(appsink);
    gst_object_unref(pipeline);
    g_main_loop_unref(main_loop);
    
    g_print("Done!\n");
    return 0;
}

