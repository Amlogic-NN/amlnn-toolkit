/****************************************************************************
 *
 *    Copyright (C) 2026 Amlogic, Inc. All rights reserved.
 *
 *    Licensed under the Apache License, Version 2.0 (the "License");
 *    you may not use this file except in compliance with the License.
 *    You may obtain a copy of the License at
 *
 *        http://www.apache.org/licenses/LICENSE-2.0
 *
 *    Unless required by applicable law or agreed to in writing, software
 *    distributed under the License is distributed on an "AS IS" BASIS,
 *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *    See the License for the specific language governing permissions and
 *    limitations under the License.
 *
 ***************************************************************************/

#include <string.h>
#include <unistd.h>
#include <string>
#include <iostream>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>

#include "llmsdk.h"

typedef struct
{
    int request_id;
    bool printed;
} MyUserData;

void callback(AML_LLMResult *result, void *userdata, AML_LLMRunStatus run_status)
{
    if (!userdata) return;

    MyUserData* my_data = (MyUserData*)userdata;

    if (run_status == AML_LLM_RUN_NORMAL)
    {
        if (!my_data->printed)
        {
            printf("[Request #%d]\n", my_data->request_id);
            my_data->printed = true;
        }
        printf("%s", result->text);
        // printf("%d,", result->token_id);
        fflush(stdout);
    }
    else if (run_status == AML_LLM_RUN_FINISH)
    {
        printf("\n");
    }
    else if (run_status == AML_LLM_RUN_ERROR)
    {
        printf("run error\n");
    }
}


std::queue<std::string> input_queue;
std::mutex mtx_talk;
std::condition_variable cv_talk;
static void get_input_string_thread(LLMContext context)
{
    while (true)
    {
        std::string input_line;
        std::getline(std::cin, input_line);

        if (input_line == "break")
        {
            printf("[Input] Call aml_llm_break\n");
            aml_llm_break(context);
            continue;
        }
        else if (input_line == "new_talk")
        {
            printf("[Input] Call aml_llm_reset\n");
            aml_llm_reset(context);
            continue;
        }
        else
        {
            std::unique_lock<std::mutex> lock(mtx_talk);
            input_queue.push(input_line);
            lock.unlock();
            cv_talk.notify_one();

            if (input_line == "exit")
            {
                break;
            }
        }
    }
}


enum ModelType
{
    QWEN,
    DEEPSEEK,
    GEMMA,
    GEMMA3,
    LLAMA,
    TINY_LLAMA,
    TINY_LLAMA_V0_4,
    PHI_1_5,
    PHI_2,
    MINICPM4,
    UNKNOWN
};

ModelType parse_model_type(const char* name)
{
    if (strcmp(name, "qwen") == 0) return QWEN;
    if (strcmp(name, "deepseek") == 0) return DEEPSEEK;
    if (strcmp(name, "gemma") == 0) return GEMMA;
    if (strcmp(name, "gemma3") == 0) return GEMMA3;
    if (strcmp(name, "llama") == 0) return LLAMA;
    if (strcmp(name, "tiny_llama") == 0) return TINY_LLAMA;
    if (strcmp(name, "tiny_llama_v0_4") == 0) return TINY_LLAMA_V0_4;
    if (strcmp(name, "phi_1_5") == 0) return PHI_1_5;
    if (strcmp(name, "phi_2") == 0) return PHI_2;
    if (strcmp(name, "minicpm4") == 0) return MINICPM4;
    return UNKNOWN;
}


int main(int argc, char **argv)
{
    if (argc < 2)
    {
        printf("Usage: %s <model_path> [--model_type <type>]\n", argv[0]);
        printf("Supported types: qwen(qwenx internvl), deepseek, gemma, gemma3, llama, tiny_llama, tiny_llama_v0_4, phi_1_5, phi_2, minicpm4\n");
        return -1;
    }

    const char* model_path = argv[1];
    ModelType model_type = UNKNOWN;

    for (int i = 2; i < argc - 1; i++)
    {
        if (strcmp(argv[i], "--model_type") == 0)
        {
            model_type = parse_model_type(argv[i + 1]);
        }
    }

    printf("\nWelcome to Amlogic LLM Demo!\n");

    LLMContext context;
    AML_LLMInitConfig init_config;
    memset(&init_config, 0, sizeof(AML_LLMInitConfig));
    init_config.model_path = model_path;
    init_config.sampling_mode = AML_LLM_ARG_Max;
    init_config.top_k = 3;
    init_config.top_p = 0.9f;
    init_config.temperature = 1.0f;
    init_config.repeat_penalty = 1.1f;

    aml_llm_init(&context, &init_config, callback);

    std::thread input_thread(get_input_string_thread, context);

    AML_LLMInput input;
    memset(&input, 0, sizeof(AML_LLMInput));
    input.input_type = AML_LLM_INPUT_PROMPT;

    AML_LLMRunConfig run_config;
    memset(&run_config, 0, sizeof(AML_LLMRunConfig));
    run_config.run_mode = AML_LLM_RUN_GENERATE;
    run_config.retain_history = 0;
    run_config.enable_think = 0;


    if (model_type != UNKNOWN)
    {
        /*
        Users can set the template prompt words according to their needs.
        After setting, call aml_llm_set_chat_template api to take effect.
        If aml_llm_set_chat_template api is not called, llmsdk will use the default template prompts words.
        The following are the default template prompt words for various models supported.
        */

        const char* system_prompt = "";
        const char* prompt_prefix = "";
        const char* prompt_postfix = "";

        switch (model_type)
        {
            case QWEN:
                system_prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n";
                prompt_prefix = "<|im_start|>user\n";
                prompt_postfix = "<|im_end|>\n<|im_start|>assistant\n";
                break;
            case DEEPSEEK:
                system_prompt = "<｜begin▁of▁sentence｜>";
                prompt_prefix = "<｜User｜>";
                prompt_postfix = "<｜Assistant｜>please don't include <think> tags in your answers\n";
                break;
            case GEMMA:
            case GEMMA3:
                system_prompt = "<bos>";
                prompt_prefix = "<start_of_turn>user\n";
                prompt_postfix = "<end_of_turn>\n<start_of_turn>model\n";
                break;
            case LLAMA:
            {
                static char system_prompt_buf[256];
                std::time_t now = std::time(nullptr);
                std::tm* local_time = std::localtime(&now);
                char date_str[30];
                std::strftime(date_str, sizeof(date_str), "%d %b %Y", local_time);
                snprintf(system_prompt_buf, sizeof(system_prompt_buf),
                    "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
                    "Cutting Knowledge Date: December 2023\n"
                    "Today Date: %s\n\n"
                    "<|eot_id|>", date_str);
                system_prompt = system_prompt_buf;
                prompt_prefix = "<|start_header_id|>user<|end_header_id|>\n\n";
                prompt_postfix = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n";
                break;
            }
            case TINY_LLAMA:
                system_prompt = "<|im_start|>system\nYou are a friendly chatbot.<|im_end|>\n";
                prompt_prefix = "<|im_start|>user\n";
                prompt_postfix = "<|im_end|>\n<|im_start|>assistant\n";
                break;
            case TINY_LLAMA_V0_4:
                system_prompt = "<s>";
                prompt_prefix = "<|im_start|>user\n";
                prompt_postfix = "<|im_end|>\n<|im_start|>assistant\n";
                break;
            case PHI_1_5:
                system_prompt = "";
                prompt_prefix = "";
                prompt_postfix = "\nAnswer:";
                break;
            case PHI_2:
                system_prompt = "";
                prompt_prefix = "Instruct: ";
                prompt_postfix = "\nOutput:";
                break;
            case MINICPM4:
                system_prompt = "";
                prompt_prefix = "<im_start>user\n";
                prompt_postfix = "<im_end>\n";
                break;
            default:
                break;
        }

        aml_llm_set_chat_template(context, system_prompt, prompt_prefix, prompt_postfix);
    }


    MyUserData my_data;
    memset(&my_data, 0, sizeof(MyUserData));

    printf("\nType your prompt and press Enter.\n");
    printf("Commands: [exit] to quit, [new_talk] to reset, [break] to stop.\n");

    while (true)
    {
        printf("\nLLM@Amlogic>>> ");
        fflush(stdout);

        std::unique_lock<std::mutex> lock(mtx_talk);
        cv_talk.wait(lock, [] { return !input_queue.empty(); });

        std::string input_str = input_queue.front();
        input_queue.pop();
        lock.unlock();

        if (input_str.empty())
        {
            printf("Please enter your question!\n");
            continue;
        }

        if (input_str == "exit")
        {
            break;
        }

        my_data.request_id++;
        my_data.printed = false;
        input.prompt_input = (const char *)input_str.c_str();

        aml_llm_run(context, &input, &run_config, &my_data);
    }

    printf("Bye~\n");

    input_thread.join();
    aml_llm_uninit(context);

    return 0;
}
