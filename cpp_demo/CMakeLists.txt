cmake_minimum_required(VERSION 3.10)
project(jetson_rtsp_cpp)

set(CMAKE_CXX_STANDARD 14)

# Find CUDA
find_package(CUDA REQUIRED)

# Find GStreamer
find_package(PkgConfig REQUIRED)
pkg_check_modules(GST REQUIRED gstreamer-1.0 gstreamer-rtsp-server-1.0 gstreamer-app-1.0 gstreamer-video-1.0 gstreamer-allocators-1.0)

# Find OpenCV
find_package(OpenCV REQUIRED)

# Find EGL and GLES (optional on x86, required on Jetson)
find_path(EGL_INCLUDE_DIR EGL/egl.h)
find_library(EGL_LIBRARY EGL)
find_library(GLES_LIBRARY GLESv2)

# Include directories
include_directories(
    ${CMAKE_SOURCE_DIR}/src
    ${CUDA_INCLUDE_DIRS}
    ${GST_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    /usr/src/jetson_multimedia_api/include
)

if(EGL_INCLUDE_DIR)
    include_directories(${EGL_INCLUDE_DIR})
endif()

# Add CUDA include directories for all targets
include_directories(SYSTEM ${CUDA_INCLUDE_DIRS})

# Source files
set(SOURCES src/main.cpp)
set(CUDA_SOURCES src/nvbuffer_cuda_processor.cu)

# Compile CUDA files
cuda_compile(CUDA_OBJS ${CUDA_SOURCES})

# Executables
add_executable(rtsp_demo src/main.cpp ${CUDA_OBJS})
add_executable(test_camera src/test_camera.cpp)

# Optional Jetson Multimedia Libraries
find_library(NVBUF_UTILS nvbuf_utils PATHS /usr/lib/aarch64-linux-gnu/tegra)
find_library(NVBUFSURFACE nvbufsurface PATHS /usr/lib/aarch64-linux-gnu/tegra)
find_library(NVBUFSURFTRANSFORM nvbufsurftransform PATHS /usr/lib/aarch64-linux-gnu/tegra)

if(NVBUF_UTILS AND NVBUFSURFACE)
    add_definitions(-DHAVE_JETSON_NVMM)
    set(JETSON_LIBS ${NVBUF_UTILS} ${NVBUFSURFACE} ${NVBUFSURFTRANSFORM})
endif()

# Link libraries
target_link_libraries(rtsp_demo
    ${CUDA_LIBRARIES}
    ${GST_LIBRARIES}
    ${OpenCV_LIBS}
    ${JETSON_LIBS}
    ${EGL_LIBRARY}
    ${GLES_LIBRARY}
    cuda
    cudart
    -L/usr/local/cuda/lib64
)

target_link_libraries(test_camera
    ${GST_LIBRARIES}
)
