CMAKE_MINIMUM_REQUIRED(VERSION 3.10)
project(cd_proj)

if(NOT CMAKE_CXX_COMPILER)
    find_program(CDPROJ_GXX NAMES g++-15.2 g++-15 g++-14 g++-13 g++-12 g++)
    if(CDPROJ_GXX)
        set(CMAKE_CXX_COMPILER "${CDPROJ_GXX}" CACHE FILEPATH "C++ compiler" FORCE)
    else()
        set(CMAKE_CXX_COMPILER g++)
    endif()
endif()

if(NOT CMAKE_C_COMPILER)
    find_program(CDPROJ_GCC NAMES gcc-15.2 gcc-15 gcc-14 gcc-13 gcc-12 gcc)
    if(CDPROJ_GCC)
        set(CMAKE_C_COMPILER "${CDPROJ_GCC}" CACHE FILEPATH "C compiler" FORCE)
    else()
        set(CMAKE_C_COMPILER gcc)
    endif()
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type" FORCE)
endif()

find_package(nlohmann_json 3.12 REQUIRED)

add_executable(cd_proj
    src/main.cc
)

target_include_directories(cd_proj PRIVATE
    include
)

target_compile_features(cd_proj PUBLIC cxx_std_20)

target_link_libraries(cd_proj
    nlohmann_json::nlohmann_json
)

# Older GCC versions (pre-9) require linking stdc++fs for std::filesystem.
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9)
    target_link_libraries(cd_proj stdc++fs)
endif()
