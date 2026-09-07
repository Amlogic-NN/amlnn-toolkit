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

import argparse
import sys

from amlllmlite.api import AMLLLMLite
from amlllmlite.api.inference import RunStatus


def stream_callback(token):
    status = token.get("status")
    if status == RunStatus.FINISH:
        print()
    elif status == RunStatus.ERROR:
        print("\n[Generation error]")
    elif token.get("text"):
        print(token["text"], end="", flush=True)


def load_rgb_image(path):
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("VLM demo requires Pillow: pip install Pillow") from exc

    with Image.open(path) as image:
        rgb = image.convert("RGB")
        width, height = rgb.size
        return {"data": rgb.tobytes(), "width": width, "height": height}


def parse_args():
    parser = argparse.ArgumentParser(description="Amlogic VLM interactive demo")
    parser.add_argument("--model", required=True, help="Path to LLM model file")
    parser.add_argument("--mmproj", required=True, help="Path to multimodal projector model file")
    parser.add_argument("--sampling-mode", default="argmax", choices=["argmax", "top_p", "top_k"])
    parser.add_argument("--top-k", type=int, default=3, dest="top_k", help="Top-K parameter")
    parser.add_argument("--top-p", type=float, default=0.9, dest="top_p", help="Top-P parameter")
    parser.add_argument("--temperature", type=float, default=1.0, help="Softmax temperature")
    parser.add_argument("--repeat-penalty", type=float, default=1.1, dest="repeat_penalty", help="Repeat penalty factor")
    parser.add_argument("--log-level", default="ERROR", dest="log_level", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--img-start", default="<|vision_start|>")
    parser.add_argument("--img-end", default="<|vision_end|>")
    parser.add_argument("--img-content", default="<image>")
    return parser.parse_args()


def main():
    args = parse_args()
    amlllm = AMLLLMLite(log_level=args.log_level)
    amlllm.config(
        model_path=args.model,
        mmproj_path=args.mmproj,
        sampling_mode=args.sampling_mode,
        top_k=args.top_k,
        top_p=args.top_p,
        temperature=args.temperature,
        repeat_penalty=args.repeat_penalty,
        on_token=stream_callback,
    )
    amlllm.init()
    images = []

    print("VLM model loaded. Use /image <path> to append images, then enter a prompt.")
    print("Commands: exit | new_talk | break")
    try:
        while True:
            try:
                line = input("\nVLM>>> ").strip()
            except EOFError:
                break
            if not line:
                print("Please enter a non-empty prompt.")
                continue
            if line == "exit":
                break
            if line == "new_talk":
                amlllm.reset_session()
                images.clear()
                print("Conversation state cleared.")
                continue
            if line == "break":
                amlllm.break_generation()
                print("Stop signal sent.")
                continue
            if line.startswith("/image "):
                path = line[7:].strip()
                try:
                    image = load_rgb_image(path)
                except Exception as exc:
                    print(f"Image load failed: {exc}")
                    continue
                images.append(image)
                print(
                    f"Image appended: {path}, {image['width']}x{image['height']}, "
                    f"total={len(images)}"
                )
                continue

            prompt = line
            if images and args.img_content not in prompt:
                prompt = args.img_content * len(images) + prompt
                print(f"Prepended {len(images)} image marker(s).")

            print("robot: ", end="", flush=True)
            result = amlllm.run(
                prompt=prompt,
                input_type="multimodal" if images else "prompt",
                retain_history=False,
                enable_think=False,
                role="user",
                images=images or None,
                img_start=args.img_start,
                img_end=args.img_end,
                img_content=args.img_content,
            )
            if not result["text"].endswith("\n"):
                print()
            images.clear()
    finally:
        amlllm.uninit()
    return 0


if __name__ == "__main__":
    sys.exit(main())
