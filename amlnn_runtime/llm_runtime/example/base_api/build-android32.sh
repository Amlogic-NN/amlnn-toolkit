#!/bin/bash
set -e

#
# Copyright (C) 2024–2025 Amlogic, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

ANDROID_NDK_PATH=/opt/android-ndk-r25c   # <-- modify this line

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build/android32"
INSTALL_DIR="${SCRIPT_DIR}/install/android32"

mkdir -p "${BUILD_DIR}" "${INSTALL_DIR}"

cmake -S "${SCRIPT_DIR}" -B "${BUILD_DIR}" \
    -DCMAKE_TOOLCHAIN_FILE="${ANDROID_NDK_PATH}/build/cmake/android.toolchain.cmake" \
    -DANDROID_ABI=armeabi-v7a \
    -DANDROID_PLATFORM=android-21 \
    -DTARGET_PLATFORM=android_lib32

cmake --build "${BUILD_DIR}" -j$(nproc)

cp "${BUILD_DIR}/base_api_llm" "${INSTALL_DIR}/"
echo "Output: ${INSTALL_DIR}/base_api_llm"
