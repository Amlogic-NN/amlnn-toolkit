#
# Copyright (C) 2026 Amlogic, Inc. All rights reserved.
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

import numpy as np
import os
import argparse
import urllib.request
from PIL import Image
from amlnn.api import AMLNN


TFLITE_URL = "https://github.com/tensorflow/tflite-support/raw/master/tensorflow_lite_support/metadata/python/tests/testdata/image_classifier/mobilenet_v2_1.0_224_quant.tflite"
LABELS_URL  = "https://raw.githubusercontent.com/tensorflow/tflite-support/refs/heads/master/tensorflow_lite_support/metadata/python/tests/testdata/image_classifier/labels.txt"

SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
TFLITE_PATH   = os.path.join(SCRIPT_DIR, "mobilenet_v2_1.0_224_quant.tflite")
LABELS_PATH   = os.path.join(SCRIPT_DIR, "labels.txt")
IMAGE_PATH    = os.path.join(SCRIPT_DIR, "fish_224x224.jpeg")
DATASET_PATH  = os.path.join(SCRIPT_DIR, "datasets.txt")


def download_if_missing(url, path):
    if not os.path.exists(path):
        print(f"Downloading {os.path.basename(path)} ...")
        urllib.request.urlretrieve(url, path)
        print(f"Saved to {path}")


def preprocess(path, tensor_attr):
    dims = tuple(int(dim) for dim in tensor_attr["dims"][:4])
    input_format = tensor_attr["format_name"]
    if input_format == "NHWC":
        _, height, width, _ = dims
    elif input_format == "NCHW":
        _, _, height, width = dims
    else:
        raise ValueError(f"Unsupported input format: {input_format}")

    img = Image.open(path).convert("RGB").resize((width, height))
    img = np.array(img, dtype=np.float32)
    img = img / 127.5 - 1.0

    data = np.expand_dims(img, axis=0)

    scale = float(tensor_attr["scale"])
    zp = int(tensor_attr["zp"])
    tensor_type = int(tensor_attr["type"])
    if tensor_type == 2:
        data = np.round(data / scale + zp).astype(np.int8)
    elif tensor_type == 3:
        data = np.round(data / scale + zp).astype(np.uint8)

    return data


def postprocess_topk(logits, labels, k=5):
    logits = logits.squeeze()
    idx = np.argsort(logits)[::-1][:k]

    print(f"\n    Top-{k} Results:")
    for i, c in enumerate(idx):
        name = labels[c] if c < len(labels) else f"Unknown({c})"
        score = logits[c]
        print(f"      {i+1}. {name:20s}  score={score:.6f}")


def main():
    parser = argparse.ArgumentParser(description="MobileNetV2 demo for Amlogic NPU")
    parser.add_argument('--mode', default='native', choices=['native', 'nnserver'],
                        help='Runtime mode: native (on-device) or nnserver (PC to board via ADB)')
    parser.add_argument('--target-platform', required=True,
                        help='Target platform ID, e.g. 001, 002, 003')
    args = parser.parse_args()

    download_if_missing(TFLITE_URL, TFLITE_PATH)
    download_if_missing(LABELS_URL, LABELS_PATH)

    amlnn = AMLNN()

    amlnn.load_tflite(model=TFLITE_PATH, quantized_model=True)

    amlnn.config(quantized_dtype='w8a8', target_platform=f"PRODUCT_PID0XA{args.target_platform.zfill(3)}")

    amlnn.compile(dataset=DATASET_PATH)

    amlnn.export_adla()

    amlnn.init_runtime(mode=args.mode, enable_perf=True)

    tensor_info = amlnn.get_tensor_info()

    print(amlnn.get_sdk_version())

    with open(LABELS_PATH, "r") as f:
        labels = [line.strip() for line in f.readlines()]

    input_data = preprocess(IMAGE_PATH, tensor_attr=tensor_info["inputs"][0])

    bin_path = os.path.splitext(IMAGE_PATH)[0] + ".bin"
    input_data.tofile(bin_path)
    print(f"Saved preprocessed input to {bin_path}")

    outputs = amlnn.inference(inputs=[input_data])

    postprocess_topk(outputs[0], labels, k=5)

    print(amlnn.get_perf_info())

    amlnn.perf_visualize()

    amlnn.uninit()


if __name__ == "__main__":
    main()
