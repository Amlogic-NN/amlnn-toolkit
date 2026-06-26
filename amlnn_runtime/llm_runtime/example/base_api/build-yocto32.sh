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

YOCTO_TOOLCHAIN_PATH=/opt/poky/4.0.20   # <-- modify this line

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build/yocto32"
INSTALL_DIR="${SCRIPT_DIR}/install/yocto32"

ENV_SCRIPT=$(find "${YOCTO_TOOLCHAIN_PATH}" -maxdepth 1 -name "environment-setup-*gnueabi*" | head -1)
if [ -z "${ENV_SCRIPT}" ]; then
    echo "Error: Yocto 32-bit environment setup script not found in ${YOCTO_TOOLCHAIN_PATH}"
    exit 1
fi
source "${ENV_SCRIPT}"

mkdir -p "${BUILD_DIR}" "${INSTALL_DIR}"

cmake -S "${SCRIPT_DIR}" -B "${BUILD_DIR}" \
    -DCMAKE_TOOLCHAIN_FILE="${OECORE_NATIVE_SYSROOT}/usr/share/cmake/OEToolchainConfig.cmake" \
    -DTARGET_PLATFORM=linux_yocto_lib32

cmake --build "${BUILD_DIR}" -j$(nproc)

cp "${BUILD_DIR}/base_api_llm" "${INSTALL_DIR}/"
echo "Output: ${INSTALL_DIR}/base_api_llm"
