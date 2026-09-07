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
import queue
import sys
import threading
from amlllmlite.api import AMLLLMLite
from amlllmlite.api.inference import RunStatus


def stream_callback(token, userdata=None):
    """Print tokens as they arrive (mimic C demo callback behavior)."""
    text = token.get("text", "")
    status = token.get("status")
    if userdata and not userdata.get("printed"):
        print(f"[Request #{userdata.get('request_id', 0)}]")
        userdata["printed"] = True
    if status == RunStatus.FINISH:
        print()
    elif status == RunStatus.ERROR:
        print("\n[Generation error]")
    elif text:
        print(text, end="", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Amlogic LLM interactive demo (Python)")
    parser.add_argument("--model", required=True, help="Path to LLM model file")
    parser.add_argument("--sampling-mode", default="argmax", choices=["argmax", "top_p", "top_k"], help="Sampling mode")
    parser.add_argument("--top-k", type=int, default=3, dest="top_k", help="Top-K parameter")
    parser.add_argument("--top-p", type=float, default=0.9, dest="top_p", help="Top-P parameter")
    parser.add_argument("--temperature", type=float, default=1.0, help="Softmax temperature")
    parser.add_argument("--repeat-penalty", type=float, default=1.1, dest="repeat_penalty", help="Repeat penalty factor")
    parser.add_argument("--log-level", default="ERROR", dest="log_level", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def main():
    args = parse_args()
    amlllm = AMLLLMLite(log_level=args.log_level)
    amlllm.config(
           model_path=args.model,
           sampling_mode=args.sampling_mode,
           top_k=args.top_k,
           top_p=args.top_p,
           temperature=args.temperature,
           repeat_penalty=args.repeat_penalty,
           on_token=stream_callback,
    )
    amlllm.init()

    print("Welcome to Amlogic LLM interactive demo (Python).")
    print("Commands: exit | new_talk | break")

    user_state = {"request_id": 0, "printed": False}

    input_queue = queue.Queue()

    def read_input():
        while True:
            try:
                user_input = input().strip()
            except EOFError:
                input_queue.put("exit")
                break
            if user_input == "break":
                try:
                    amlllm.break_generation()
                    print("\nStop signal sent.")
                except Exception as exc:
                    print(f"\nBreak failed: {exc}")
                continue
            if user_input == "new_talk":
                try:
                    amlllm.reset_session()
                    print("\nConversation state cleared.")
                except Exception as exc:
                    print(f"\nReset failed: {exc}")
                continue
            input_queue.put(user_input)
            if user_input == "exit":
                break

    input_thread = threading.Thread(target=read_input, daemon=True)
    input_thread.start()

    try:
        while True:
            print("\nLLM@Amlogic>>> ", end="", flush=True)
            user_input = input_queue.get()
            if not user_input:
                print("Please enter a non-empty prompt.")
                continue
            if user_input == "exit":
                break

            try:
                user_state["request_id"] += 1
                user_state["printed"] = False
                result = amlllm.run(
                    prompt=user_input,
                    input_type="prompt",
                    run_mode="generate",
                    retain_history=False,
                    enable_think=False,
                    role="user",
                    user_data=user_state,
                )
                if not result["text"].endswith("\n"):
                    print()
                print(f"Tokens generated: {result['token_count']}")
            except KeyboardInterrupt:
                print("\nKeyboardInterrupt received. Sending break...")
                amlllm.break_generation()
            except Exception as exc:
                print(f"\nGeneration failed: {exc}")
    finally:
        amlllm.uninit()


if __name__ == "__main__":
    sys.exit(main())
