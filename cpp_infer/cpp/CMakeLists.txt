cmake_minimum_required(VERSION 3.16)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(cpp_infer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(mnist_onnx_infer main.cc)

find_package(OpenCV REQUIRED)
find_package(onnxruntime CONFIG QUIET)

if(TARGET onnxruntime::onnxruntime)
    target_link_libraries(mnist_onnx_infer PRIVATE onnxruntime::onnxruntime)
else()
    find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h
        PATH_SUFFIXES onnxruntime
    )
    find_library(ONNXRUNTIME_LIBRARY onnxruntime)

    if(NOT ONNXRUNTIME_INCLUDE_DIR OR NOT ONNXRUNTIME_LIBRARY)
        message(FATAL_ERROR "ONNX Runtime C++ library not found. Install onnxruntime headers/library first.")
    endif()

    target_include_directories(mnist_onnx_infer PRIVATE ${ONNXRUNTIME_INCLUDE_DIR})
    target_link_libraries(mnist_onnx_infer PRIVATE ${ONNXRUNTIME_LIBRARY})
endif()

target_compile_definitions(mnist_onnx_infer
    PRIVATE CPP_INFER_PROJECT_ROOT="${CMAKE_CURRENT_LIST_DIR}/.."
)

target_include_directories(mnist_onnx_infer PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(mnist_onnx_infer PRIVATE ${OpenCV_LIBS})
