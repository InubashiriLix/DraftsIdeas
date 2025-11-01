cmake_minimum_required(VERSION 3.10)
project(trace CXX)

# C++11
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 生成 compile_commands.json
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ---- 简单开关：根目录若存在 .release 文件，则走发布配置，否则走调试配置 ----
set(_TRACE_MARKER_RELEASE "${CMAKE_SOURCE_DIR}/.release")
if(EXISTS "${_TRACE_MARKER_RELEASE}")
    set(TRACE_DEV_MODE OFF CACHE BOOL "Enable debug-friendly flags and sanitizers" FORCE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
else()
    set(TRACE_DEV_MODE ON CACHE BOOL "Enable debug-friendly flags and sanitizers" FORCE)
    if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
        set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type" FORCE)
    endif()
endif()

# Sanitizer 开关（默认开；只在 TRACE_DEV_MODE=ON 时生效）
option(TRACE_USE_SANITIZERS "Enable Address/Undefined sanitizers" ON)

# 头文件目录（沿用你的写法）
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# 可执行文件
add_executable(trace
    src/main.cc
)

# 仅针对 GCC 配置编译/链接选项
if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_compile_options(trace PRIVATE -Wall -Wextra -Wpedantic)

    if (TRACE_DEV_MODE)
        # 调试友好：行号/无优化/保留帧指针/开启 libstdc++ 断言
        target_compile_options(trace PRIVATE -O0 -g3 -fno-omit-frame-pointer -D_GLIBCXX_ASSERTIONS)
        target_link_options(trace PRIVATE -g)
        if (TRACE_USE_SANITIZERS)
            target_compile_options(trace PRIVATE -fsanitize=address,undefined)
            target_link_options(trace PRIVATE -fsanitize=address,undefined)
        endif()
    else()
        # 发布：优化+关闭断言
        target_compile_options(trace PRIVATE -O2 -DNDEBUG)
    endif()
else()
    message(WARNING "当前编译器是 ${CMAKE_CXX_COMPILER_ID}，这份脚本只针对 GCC 做了优化配置。")
endif()

