cmake_minimum_required(VERSION 3.10)
project(base_api_llm CXX)

set(CMAKE_CXX_STANDARD 11)

# Resolve SDK root relative to this CMakeLists.txt
get_filename_component(LLMSDK_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../llmsdk" ABSOLUTE)

# Select the pre-built library directory via -DTARGET_PLATFORM=<value>.
# Supported values:
#   android_lib64       android_lib32
#   linux_yocto_lib64   linux_yocto_lib32
if(NOT DEFINED TARGET_PLATFORM)
    set(TARGET_PLATFORM "linux_yocto_lib64")
    message(STATUS "TARGET_PLATFORM not set, defaulting to ${TARGET_PLATFORM}")
endif()

if(TARGET_PLATFORM STREQUAL "android_lib64")
    set(LLMSDK_LIB_DIR "${LLMSDK_ROOT}/android/lib64")
elseif(TARGET_PLATFORM STREQUAL "android_lib32")
    set(LLMSDK_LIB_DIR "${LLMSDK_ROOT}/android/lib32")
elseif(TARGET_PLATFORM STREQUAL "linux_yocto_lib64")
    set(LLMSDK_LIB_DIR "${LLMSDK_ROOT}/linux/yocto/lib64")
elseif(TARGET_PLATFORM STREQUAL "linux_yocto_lib32")
    set(LLMSDK_LIB_DIR "${LLMSDK_ROOT}/linux/yocto/lib32")
else()
    message(FATAL_ERROR
        "Unknown TARGET_PLATFORM '${TARGET_PLATFORM}'.\n"
        "Supported values: android_lib64, android_lib32, "
        "linux_yocto_lib64, linux_yocto_lib32")
endif()

message(STATUS "LLMSDK_ROOT    : ${LLMSDK_ROOT}")
message(STATUS "LLMSDK_LIB_DIR : ${LLMSDK_LIB_DIR}")

include_directories("${LLMSDK_ROOT}/include")
link_directories("${LLMSDK_LIB_DIR}")

add_executable(base_api_llm base_api_llm.cpp)
target_link_libraries(base_api_llm llmsdk)
