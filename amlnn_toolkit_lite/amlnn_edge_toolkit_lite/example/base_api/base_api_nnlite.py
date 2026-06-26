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
from PIL import Image
from amlnnlite.api import AMLNNLite


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
    parser = argparse.ArgumentParser(description="MobileNetV2 demo for Amlogic NPU (AMLNNLite)")
    parser.add_argument('--model-path', required=True, help='Path to .adla model file')
    parser.add_argument('--image-path', required=True, help='Path to input image')
    parser.add_argument('--labels', required=True, help='Path to labels.txt')
    args = parser.parse_args()

    amlnn = AMLNNLite()

    amlnn.init_runtime(mode="native", enable_perf=True)

    amlnn.load_model(path=args.model_path)

    tensor_info = amlnn.get_tensor_info()

    print(amlnn.get_sdk_version())

    if not os.path.exists(args.labels):
        print(f"Error: Label file not found: {args.labels}")
        amlnn.uninit()
        return

    with open(args.labels, "r") as f:
        labels = [line.strip() for line in f.readlines()]

    input_data = preprocess(args.image_path, tensor_attr=tensor_info["inputs"][0])

    outputs = amlnn.inference(inputs=[input_data])

    postprocess_topk(outputs[0], labels, k=5)

    print(amlnn.get_perf_info())

    amlnn.perf_visualize()

    amlnn.uninit()


if __name__ == "__main__":
    main()
