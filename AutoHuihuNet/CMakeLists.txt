CMAKE_MINIMUM_REQUIRED(VERSION 3.12)
project(HuihuFucker VERSION 0.10 LANGUAGE CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type" FORCE)
endif()

include_directories(
    include/
)

find_package(CURL REQUIRED)

add_executable(HuihuFucker
    src/HuihuFucker.cc
    src/main.cc
)

target_compile_features(HuihuFucker PUBLIC cxx_std_20)
target_link_libraries(HuihuFucker PRIVATE ${CURL_LIBRARIES})
