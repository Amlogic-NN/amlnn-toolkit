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
from datetime import datetime

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


def apply_model_template(amlllm: AMLLLMLite, model_type: str):
    """Set chat templates using the same defaults as the C demo."""
    system_prompt = ""
    prompt_prefix = ""
    prompt_postfix = ""

    if model_type == "qwen":
        system_prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        prompt_prefix = "<|im_start|>user\n"
        prompt_postfix = "<|im_end|>\n<|im_start|>assistant\n"
    elif model_type == "deepseek":
        system_prompt = "<｜begin▁of▁sentence｜>"
        prompt_prefix = "<｜User｜>"
        prompt_postfix = "<｜Assistant｜>please don't include <think> tags in your answers\n"
    elif model_type in ("gemma", "gemma3"):
        system_prompt = "<bos>"
        prompt_prefix = "<start_of_turn>user\n"
        prompt_postfix = "<end_of_turn>\n<start_of_turn>model\n"
    elif model_type == "llama":
        date_str = datetime.now().strftime("%d %b %Y")
        system_prompt = (
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            "Cutting Knowledge Date: December 2023\n"
            f"Today Date: {date_str}\n\n"
            "<|eot_id|>"
        )
        prompt_prefix = "<|start_header_id|>user<|end_header_id|>\n\n"
        prompt_postfix = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    elif model_type == "tiny_llama":
        system_prompt = "<|im_start|>system\nYou are a friendly chatbot.<|im_end|>\n"
        prompt_prefix = "<|im_start|>user\n"
        prompt_postfix = "<|im_end|>\n<|im_start|>assistant\n"
    elif model_type == "tiny_llama_v0_4":
        system_prompt = "<s>"
        prompt_prefix = "<|im_start|>user\n"
        prompt_postfix = "<|im_end|>\n<|im_start|>assistant\n"
    elif model_type == "phi_1_5":
        prompt_postfix = "\nAnswer:"
    elif model_type == "phi_2":
        prompt_prefix = "Instruct: "
        prompt_postfix = "\nOutput:"
    elif model_type == "minicpm4":
        system_prompt = ""
        prompt_prefix = "<im_start>user\n"
        prompt_postfix = "<im_end>\n"

    if system_prompt or prompt_prefix or prompt_postfix:
        amlllm.set_chat_template(system_prompt, prompt_prefix, prompt_postfix)


def parse_args():
    parser = argparse.ArgumentParser(description="Amlogic LLM interactive demo (Python)")
    parser.add_argument("--model", required=True, help="Path to LLM model file")
    parser.add_argument("--sampling-mode", default="argmax", choices=["argmax", "top_p", "top_k"], help="Sampling mode")
    parser.add_argument("--top-k", type=int, default=3, dest="top_k", help="Top-K parameter")
    parser.add_argument("--top-p", type=float, default=0.9, dest="top_p", help="Top-P parameter")
    parser.add_argument("--temperature", type=float, default=1.0, help="Softmax temperature")
    parser.add_argument("--repeat-penalty", type=float, default=1.1, dest="repeat_penalty", help="Repeat penalty factor")
    parser.add_argument("--log-level", default="ERROR", dest="log_level", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--model-type", default="none", dest="model_type",
                        choices=["none", "qwen", "deepseek", "gemma", "gemma3", "llama", "tiny_llama", "tiny_llama_v0_4", "phi_1_5", "phi_2", "minicpm4"],
                        help="Optional builtin model template")
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

    if args.model_type != "none":
        apply_model_template(amlllm, args.model_type)

    print("Welcome to Amlogic LLM interactive demo (Python).")
    print("Commands: exit | new_talk | break")

    user_state = {"request_id": 0, "printed": False}

    try:
        while True:
            try:
                user_input = input("\nLLM@Amlogic>>> ").strip()
            except EOFError:
                print("\nExit")
                break

            if not user_input:
                print("Please enter a non-empty prompt.")
                continue

            if user_input == "exit":
                break

            if user_input == "new_talk":
                amlllm.reset_session()
                print("Conversation state cleared.")
                continue

            if user_input == "break":
                amlllm.break_generation()
                print("Stop signal sent.")
                continue

            try:
                user_state["request_id"] += 1
                user_state["printed"] = False
                result = amlllm.run(
                    prompt=user_input,
                    input_type="prompt",
                    run_mode="generate",
                    retain_history=False,
                    enable_think=False,
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
