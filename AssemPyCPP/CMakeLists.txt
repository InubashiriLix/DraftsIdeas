cmake_minimum_required(VERSION 3.14)
project(CPPPythonDemo LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(Python3 COMPONENTS Development REQUIRED)

add_executable(cpp_python_demo main.cpp)
target_link_libraries(cpp_python_demo PRIVATE Python3::Python)
