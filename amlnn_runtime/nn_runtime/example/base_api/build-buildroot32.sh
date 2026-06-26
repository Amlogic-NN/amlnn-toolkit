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

BUILDROOT_TOOLCHAIN_PATH=/opt/gcc-arm-10.3-2021.07-x86_64-arm-none-linux-gnueabihf   # <-- modify this line

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build/buildroot32"
INSTALL_DIR="${SCRIPT_DIR}/install/buildroot32"

mkdir -p "${BUILD_DIR}" "${INSTALL_DIR}"

cmake -S "${SCRIPT_DIR}" -B "${BUILD_DIR}" \
    -DCMAKE_C_COMPILER="${BUILDROOT_TOOLCHAIN_PATH}/bin/arm-none-linux-gnueabihf-gcc" \
    -DCMAKE_CXX_COMPILER="${BUILDROOT_TOOLCHAIN_PATH}/bin/arm-none-linux-gnueabihf-g++" \
    -DCMAKE_SYSTEM_NAME=Linux \
    -DCMAKE_SYSTEM_PROCESSOR=arm \
    -DTARGET_PLATFORM=linux_buildroot_lib32

cmake --build "${BUILD_DIR}" -j$(nproc)

cp "${BUILD_DIR}/base_api_nn" "${INSTALL_DIR}/"
echo "Output: ${INSTALL_DIR}/base_api_nn"
