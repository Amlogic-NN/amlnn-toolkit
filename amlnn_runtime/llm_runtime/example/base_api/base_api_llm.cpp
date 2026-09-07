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
        printf("%s", result->generation.text);
        // printf("%d,", result->generation.token_id);
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



int main(int argc, char **argv)
{
    const char* model_path = NULL;
    AML_LLMSamplingMode sampling_mode = AML_LLM_ARG_Max;
    int retain_history = 0;

    for (int i = 1; i < argc; ++i)
    {
        if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0)
        {
            printf("Usage: %s --model_path <path> [--sampling_mode <0|1|2>] [--retain_history <0|1>]\n", argv[0]);
            printf("       %s <model_path>\n", argv[0]);
            return 0;
        }
        else if (strcmp(argv[i], "--model_path") == 0 && i + 1 < argc)
        {
            model_path = argv[++i];
        }
        else if (strcmp(argv[i], "--sampling_mode") == 0 && i + 1 < argc)
        {
            sampling_mode = (AML_LLMSamplingMode)atoi(argv[++i]);
        }
        else if (strcmp(argv[i], "--retain_history") == 0 && i + 1 < argc)
        {
            retain_history = atoi(argv[++i]);
        }
        else if (argv[i][0] != '-' && !model_path)
        {
            model_path = argv[i];
        }
    }

    if (!model_path)
    {
        printf("Usage: %s --model_path <path> [--sampling_mode <0|1|2>] [--retain_history <0|1>]\n", argv[0]);
        printf("       %s <model_path>\n", argv[0]);
        return -1;
    }

    printf("\nWelcome to Amlogic LLM Demo!\n");

    LLMContext context;
    AML_LLMInitConfig init_config;
    memset(&init_config, 0, sizeof(AML_LLMInitConfig));
    init_config.model_path = model_path;
    init_config.sampling_mode = sampling_mode;
    init_config.top_k = 3;
    init_config.top_p = 0.9f;
    init_config.temperature = 1.0f;
    init_config.repeat_penalty = 1.1f;

    if (aml_llm_init(&context, &init_config, callback) != AML_LLM_Status_Success)
    {
        printf("aml_llm_init failed\n");
        return -1;
    }

    std::thread input_thread(get_input_string_thread, context);

    AML_LLMInput input;
    memset(&input, 0, sizeof(AML_LLMInput));
    input.input_type = AML_LLM_INPUT_PROMPT;
    input.role = "user";

    AML_LLMRunConfig run_config;
    memset(&run_config, 0, sizeof(AML_LLMRunConfig));
    run_config.run_mode = AML_LLM_RUN_GENERATE;
    run_config.retain_history = retain_history;
    run_config.enable_think = 0;


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
