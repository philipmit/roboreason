
import gc

# things for data loader
import json
import math
import os
import random
import re
import textwrap
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Optional, Sized, Union

# from unittest.mock import patch
# import deepspeed
# import megfile
import numpy as np
import PIL
import torch
import torch.utils.data
import yaml
from accelerate.utils import broadcast_object_list, gather, gather_object, set_seed
from datasets import Image, load_dataset, load_from_disk
from PIL import Image
# from preprocess_data import get_frames
from qwen_vl_utils import process_vision_info, smart_resize
from tqdm import tqdm
from transformers import (
    AutoModelForVision2Seq,
    AutoProcessor,
    Qwen2VLForConditionalGeneration,
)
# from trl import GRPOConfig, ModelConfig, TrlParser, get_peft_config
from trl.data_utils import (
    apply_chat_template,
    is_conversational,
    maybe_apply_chat_template,
)
from vllm import LLM, SamplingParams

from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

# Example fine-tuned model.
# checkpoint_num = 'pnp_q2vl2b_new2_test2_3_v1_from_sft2_checkpoint4000_test1/checkpoint-2500'
# checkpoint_path = f"/mnt/proj-maple/pschroeder/PR1/new_output/{checkpoint_num}"

# Zero-shot Qwen V1.
# checkpoint_num = (
#     "pnp_q2vl2b_new2_test2_3_v1_from_sft2_checkpoint4000_test1/checkpoint-2500"
# )
# checkpoint_path = f"/mnt/proj-maple/pschroeder/PR1/new_output/{checkpoint_num}"

# SERL Pick cube GRPO.
# checkpoint_num = "franka_lift_cube_set1_q2vl2b_new2_test2_3_v1_from_b8sft4000grpo2500/checkpoint-4500"
# checkpoint_path = f"/mnt/proj-maple/pschroeder/PR1/new_output/{checkpoint_num}"

# Default Qwen.
# checkpoint_num = "q2vl2b"
# checkpoint_path = f"/root/qwen-reward-labeler/qwen_reward_labeler/Qwen2-VL-2B-Instruct"

temperature_ = 1.0
# temperature_ = 0.0

min_pixels = 3136
max_pixels = 12845056
max_prompt_length = 2048

system_prompt_template = (
    "You are an expert roboticist with the goal of predicting task progress percentages given frames from a video of a robot attempting to complete a task. "
    + "You first think, in the form of an internal monologue, before providing your final answer. "
    + "Your reasoning process MUST BE enclosed within <think> </think> tags and should include detailed reasoning. "
    + "Your final answer MUST BE enclosed within <answer> </answer> tags and should be a integer (positive or negative) representing current task progress percentage. "
    + "Example output format: <think>[detailed reasoning process]</think><answer>[current task progress]%</answer>"
)

# user_prompt_template = (
#     "Here is an image containing multiple camera views of a robot attempting to complete a task. "
#     + "The views on the left are from the robot's wrist camera. The views on the right are from an external camera. "
#     + "The views from the previous timestep are shown on the top row. The views from the current timestep are shown on the bottom row. "
#     + "The task description is: {task_description}. "
#     + "The task progress for the previous timestep is {prev_progress}%. Predict the task progress for the current timestep."
# )


question_template = "{question}"
problem_key = "question"
answer_key = "answer"
image_key = "image"


# def load_model(checkpoint_path: Path, vllm_device: str = "cuda:0", gpu_memory_utilization: float = 0.9):
#     if "q2vl2b" in checkpoint_path:
#         model_id = "Qwen/Qwen2-VL-2B-Instruct"
#     elif "q2vl7b" in checkpoint_path:
#         model_id = "Qwen/Qwen2-VL-7B-Instruct"
#     elif "q25vl3b" in checkpoint_path:
#         model_id = "Qwen/Qwen2.5-VL-3B-Instruct"
#     elif "q25vl7b" in checkpoint_path:
#         model_id = "Qwen/Qwen2.5-VL-7B-Instruct"
#     elif "q3vl2b" in checkpoint_path:
#         model_id = "Qwen/Qwen3-VL-2B-Instruct"
#     elif "q3vl4b" in checkpoint_path:
#         model_id = "Qwen/Qwen3-VL-4B-Instruct"
#     elif "q3vl8b" in checkpoint_path:
#         model_id = "Qwen/Qwen3-VL-8B-Instruct"
#     # 
#     if model_id in ["Qwen/Qwen2-VL-2B-Instruct"]:
#         os.system(f"cp /mnt/proj-maple/pschroeder/PR1/new_output/pnp_q2vl2b_new2_test2_3_v1/final/preprocessor_config.json {checkpoint_path}/")
#     elif model_id in ["Qwen/Qwen2-VL-7B-Instruct"]:
#         os.system(f"cp /mnt/proj-maple/pschroeder/PR1/new_output/pnp_q2vl7b_new3b_test2_3_v4/final/preprocessor_config.json {checkpoint_path}/")
#     elif model_id in ["Qwen/Qwen2.5-VL-3B-Instruct"]:
#         os.system(f"cp /mnt/proj-maple/pschroeder/PR1/new_output/pnp_q25vl3b_new2_test2_3_v2/final/preprocessor_config.json {checkpoint_path}/")
#     elif model_id in ["Qwen/Qwen2.5-VL-7B-Instruct"]:
#         os.system(f"cp /mnt/proj-maple/pschroeder/PR1/new_output/pnp_q25vl7b_new2_test2_3_v3/final/preprocessor_config.json {checkpoint_path}/")
#     elif model_id in ["Qwen/Qwen3-VL-8B-Instruct"]:
#         # os.system(f"cp /mnt/proj-maple/pschroeder/PR1/new_output/d_0717_1129_2_q3vl8b_sftv2_b48_ga1_lr1e5_eb48_test1/checkpoint-30000/preprocessor_config.json {checkpoint_path}/")
#         # os.system(f"cp /mnt/proj-maple/pschroeder/PR1/new_output/d_0717_1129b_q3vl8b_sftv2_b48_ga1_lr1e5_eb48_test1/checkpoint-85000/preprocessor_config.json {checkpoint_path}/")
#         processor = AutoProcessor.from_pretrained(checkpoint_path, fix_mistral_regex=True)
#         # os.system(f"rm {checkpoint_path}/preprocessor_config.json")
#         # processor.save_pretrained(f'{checkpoint_path}')
#         # processor = None
#     else:
#         assert False, "need to create preprocessor_config.json for this model"
#     # 
#     processing_class = AutoProcessor.from_pretrained(model_id)
#     pad_token_id = processing_class.tokenizer.pad_token_id
#     processing_class.pad_token_id = pad_token_id
#     processing_class.eos_token_id = processing_class.tokenizer.eos_token_id
#     if "Qwen" in model_id:
#         processing_class.image_processor.max_pixels = max_pixels
#         processing_class.image_processor.min_pixels = min_pixels
    
#     if 'Qwen3' in model_id:
#         # The tokenizer you are loading from '/mnt/proj-maple/pschroeder/PR1/new_output/d_0717_1129_2_q3vl8b_sftv2_b48_ga1_lr1e5_eb48_test1/checkpoint-30000' with an incorrect regex pattern: https://huggingface.co/mistralai/Mistral-Small-3.1-24B-Instruct-2503/discussions/84#69121093e8b480e709447d5e. This will lead to incorrect tokenization. You should set the `fix_mistral_regex=True` flag when loading this tokenizer to fix this issue.
#         processor = AutoProcessor.from_pretrained(checkpoint_path)
#         # processor
#         # os.system('rm /mnt/proj-maple/pschroeder/PR1/new_output/d_0717_1129_2_q3vl8b_sftv2_b48_ga1_lr1e5_eb48_test1/checkpoint-30000/preprocessor_config.json')
#         # os.system('ls /mnt/proj-maple/pschroeder/PR1/new_output/d_0717_1129_2_q3vl8b_sftv2_b48_ga1_lr1e5_eb48_test1/checkpoint-20000/')
#         # save processor to  
#         # processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")
#         # processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-8B-Instruct", fix_mistral_regex=True)
#         # processor = AutoProcessor.from_pretrained(checkpoint_path, fix_mistral_regex=True)
#         # processor.save_pretrained('/mnt/proj-maple/pschroeder/PR1/new_output/d_0717_1129_2_q3vl8b_sftv2_b48_ga1_lr1e5_eb48_test1/checkpoint-30000')
#     else:
#         processor = AutoProcessor.from_pretrained(checkpoint_path)
#     # 
#     if 'Qwen3' in model_id:
#         # del llm
#         llm = LLM(model=checkpoint_path, gpu_memory_utilization=gpu_memory_utilization, enable_prefix_caching=True)
#         # import vllm
#         # print(vllm.__version__)
#         # 0.12.0
#         # Traceback (most recent call last):
#         # File "<stdin>", line 1, in <module>
#         # File "/usr/local/lib/python3.10/dist-packages/vllm/entrypoints/llm.py", line 300, in __init__
#         #     engine_args = EngineArgs(
#         # TypeError: EngineArgs.__init__() got an unexpected keyword argument 'device'
#     else:
#         llm = LLM(model=checkpoint_path, device=vllm_device, gpu_memory_utilization=gpu_memory_utilization, enable_prefix_caching=True)
#     # 
#     sampling_params = SamplingParams(
#         # temperature=args.temperature,
#         temperature=temperature_,
#         top_p=0.9,
#         top_k=50,
#         # max_tokens=self.max_completion_length,
#         max_tokens=200,
#     )
#     return llm, processing_class, processor, sampling_params



def sole_batch_decode(
    processing_class,
    processor,
    llm,
    sampling_params,
    # task_description_text,
    dataset_dict,
    # batch_vlm_image_wrist_view_file_list_list,
    # batch_vlm_image_external_view_file_list_list,
    video_count,
    video_step_count,
    image_dir=None,
    video_idx_list=None,
    image_key="image",
    problem_key="question",
    repeated_sampling=False,
    dropout_100=False,
    dropout_random=0,
    global_random=0,
):
    #
    if video_count == 1:
        print(f'**************** SOLE-R1 batch decoding for 1 video for {video_step_count} steps ****************')
    else:
        print(f'**************** SOLE-R1 batch decoding across {video_count} videos (in parallel) for {video_step_count} steps ****************')
    # # assuming no progress prediction for first frame since this will always be 0
    # video_step_count = video_step_count - 1
    #
    # dataset_dict = get_dataset_dict(
    #     task_description_text,
    #     batch_vlm_image_wrist_view_file_list_list,
    #     batch_vlm_image_external_view_file_list_list,
    # )
    #
    if video_idx_list is None:
        video_idx_list = []
        lev_video_i_start_idx = 0
        for i in range(video_count):
            video_idx_list_i = []
            for j in range(video_step_count):
                video_idx_list_i.append(lev_video_i_start_idx + j)
            lev_video_i_start_idx = video_idx_list_i[-1] + 1
            video_idx_list.append(video_idx_list_i)
        #
    ############################## batch decoding
    text_output_list_batch = []
    text_input_list_batch = []
    example_list_batch = []
    answer_list_batch = []
    prev_answer_batch = [["0" for x in video_idx_list]]
    sol_list_batch = []
    # print(prev_answer_batch[-1])
    for video_step in range(video_step_count):
        gc.collect()
        # video_step = 0
        current_video_idx_list = [x[video_step] for x in video_idx_list]
        #
        current_video_idx_batch_input = []
        example_list = []
        for current_video_idx_list_idx in range(len(current_video_idx_list)):
            current_video_idx = current_video_idx_list[current_video_idx_list_idx]
            example = dataset_dict[current_video_idx]
            example_list.append(example)
            # 
            if image_key in example:
                image = load_image(example[image_key])
                # image = self.load_image(image_path)
                # image = load_image(image_path)
                # 
                width, height = image.size
                # min_pixels = self.script_args.min_pixels
                # max_pixels = self.script_args.max_pixels
                # min_pixels = 3136
                # max_pixels = 12845056
                resized_height, resized_width = smart_resize(
                    height,
                    width,
                    factor=28,
                    min_pixels=min_pixels,
                    max_pixels=max_pixels,
                )
                image = image.resize((resized_width, resized_height))
            #
            else:
                image = None 
            #
            prompt = (
                make_conversation(example)
                if "image" not in example
                else make_conversation_image(example)
            )
            #
            inputs = [
                {
                    "image": image,
                    "problem": example[problem_key],
                    # 'solution': example[answer_key],
                    "image_name": example[image_key],
                    "prompt": prompt,
                }
            ]
            if True:
                # # dynamic temperature in training
                # if self.accelerator.is_main_process and self.script_args.temperature_func is not None:
                #     self.sampling_params.temperature = self.temperature_func(self.state.global_step)
                # #
                # device = self.accelerator.device
                for x in inputs:
                    x["prompt"] = json.loads(x["prompt"])
                #
                # type(inputs[0]['prompt'])
                #
                prompts = [x["prompt"] for x in inputs]
                #
                # prompts_text = [maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in inputs]
                prompts_text = [
                    maybe_apply_chat_template(example, processing_class)["prompt"]
                    for example in inputs
                ]
                #
                images = [x["image"] for x in inputs]
                # prompt_inputs = self.processing_class(
                prompt_inputs = processing_class(
                    text=prompts_text,
                    images=images,
                    return_tensors="pt",
                    padding=True,
                    padding_side="left",
                    add_special_tokens=False,
                )
                # this just returns inputs
                # prompt_inputs = super()._prepare_inputs(prompt_inputs)
                batch_size = 1
                batched_inputs = {
                    k: v.repeat(batch_size, *[1] * (v.dim() - 1))
                    if isinstance(v, torch.Tensor)
                    else v
                    for k, v in prompt_inputs.items()
                }
                # if self.max_prompt_length is not None:
                if max_prompt_length is not None:
                    # batched_inputs["input_ids"].shape
                    # torch.Size([1, 1025])
                    # batched_inputs["attention_mask"].shape
                    # torch.Size([1, 1025])
                    batched_inputs["input_ids"] = batched_inputs["input_ids"][
                        :, -max_prompt_length:
                    ]
                    batched_inputs["attention_mask"] = batched_inputs["attention_mask"][
                        :, -max_prompt_length:
                    ]
                #
                inputs_vllm = []
                for image_data, messages in zip(images, prompts):
                    # prompt = self.processing_class.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    prompt = processing_class.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                    image_data, _ = (
                        process_vision_info(messages)
                        if not isinstance(image_data, PIL.Image.Image)
                        else (image_data, None)
                    )
                    for i in range(batch_size):
                        inputs_vllm.append(
                            {
                                "prompt": prompt,
                                "multi_modal_data": {"image": image_data},
                            }
                        )
                #
                all_inputs_vllm = gather_object(inputs_vllm)
            #
            if True:
                prev_answer = prev_answer_batch[video_step][current_video_idx_list_idx]
                if dropout_100:
                    try: 
                        if int(prev_answer) > 98:
                            print(f"dropout_100 applied at video_step {video_step}, current_video_idx_list_idx {current_video_idx_list_idx}, prev_answer {prev_answer} will be set to empty string")
                            prev_answer = " "
                    except:
                        print("could not convert prev_answer to int:", prev_answer)
                if not prev_answer in [" "]:
                    if dropout_random > 0:
                        # get random float between 0 and 1
                        random_float = random.uniform(0, 1)
                        if random_float < dropout_random:
                            print(f"dropout_random applied at video_step {video_step}, current_video_idx_list_idx {current_video_idx_list_idx}, prev_answer {prev_answer} will be set to empty string")
                            prev_answer = " "
                # 
                replace_start = "The task progress for the previous timestep is "
                if replace_start in all_inputs_vllm[0]["prompt"]:
                    replace_end = "%. "
                    all_inputs_vllm[0]["prompt"] = (
                        all_inputs_vllm[0]["prompt"].split(replace_start)[0]
                        + replace_start
                        + prev_answer
                        + replace_end
                        + all_inputs_vllm[0]["prompt"]
                        .split(replace_start)[1]
                        .split(replace_end)[1]
                    )
                #
            #
            current_video_idx_batch_input.append(all_inputs_vllm[0])
        #
        # if video_step>0:
        # print(example_list==example_list_old)
        # example_list_old = example_list.copy()
        # current_video_idx_batch_input[0]['prompt']
        #
        retry_gen_count = 0
        while retry_gen_count < 3:
            outputs = None
            del outputs
            outputs = llm.generate(
                current_video_idx_batch_input,
                sampling_params=sampling_params,
                use_tqdm=False,
            )
            # len(outputs)
            if not len(outputs) == len(current_video_idx_batch_input):
                print(retry_gen_count)
                print(
                    f"Outputs length {len(outputs)} is not equal to inputs length {len(current_video_idx_batch_input)}"
                )
                retry_gen_count = retry_gen_count + 1
                # assert False, f"Outputs length {len(outputs)} is not equal to inputs length {len(current_video_idx_batch_input)}"
            else:
                retry_gen_count = 999
            #
        completion_ids = [
            out.token_ids for completions in outputs for out in completions.outputs
        ]
        #
        text_output = processor.batch_decode(completion_ids, skip_special_tokens=True)
        #
        # print('')
        # print(checkpoint_path, lev_idx, lev_video_idx)
        # print(all_inputs_vllm[0]['prompt'])
        # print(text_output)
        # text_output_list.append(text_output)
        text_output_list_batch = text_output_list_batch + [text_output]
        example_list_batch = example_list_batch + [example_list]
        # text_input_list.append(all_inputs_vllm[0]['prompt'])
        text_input_list_batch = text_input_list_batch + [
            [
                current_video_idx_batch_input_i["prompt"]
                for current_video_idx_batch_input_i in current_video_idx_batch_input
            ]
        ]
        prev_answer_batch = prev_answer_batch + [
            [get_answer_from_completion(text_output_i) for text_output_i in text_output]
        ]
        answer_list_batch = answer_list_batch + [
            [get_answer_from_completion(text_output_i) for text_output_i in text_output]
        ]
        # sol_list_batch = sol_list_batch + [[example[answer_key] for example in example_list]]
        # print('')
        # print(current_video_idx_list)
        print('********** video_step', video_step, '**********')
        print('REASONING TRACES ACROSS VIDEOS:')
        print(text_output_list_batch[-1])
        print('PREDICTED PROGRESS ACROSS VIDEOS:')
        print(prev_answer_batch[-1])
        # print(sol_list_batch[-1])
        # len(prev_answer_batch)
        # prev_answer_batch[0]
        # prev_answer_batch[1]
    #
    ##############################
    #
    ############################## re-creating variable lists from batch input and output
    # answer_list = [get_answer_from_completion(text_output) for text_output in text_output_list]
    # sol_list = [example[answer_key] for example in example_list]
    #
    # answer_list_batch: first index is video_step (current step for all videos), second index is video_idx
    #
    return (
        text_input_list_batch,
        text_output_list_batch,
        example_list_batch,
        answer_list_batch,
    )



def load_image(image_input):
    # when image is on oss, please run `unset http_proxy https_proxy all_proxy no_proxy`
    from io import BytesIO
    import os
    os.environ['OSS_ENDPOINT'] = 'http://oss.i.basemind.com'
    if isinstance(image_input, np.ndarray):
        image = Image.fromarray(image_input.astype('uint8'))
    elif isinstance(image_input, bytes):
        image = Image.open(BytesIO(image_input), "r").convert('RGB')
    elif 's3://' in image_input:
        with megfile.smart_open(image_input, "rb") as f:
            bytes_data = f.read()
        image = Image.open(BytesIO(bytes_data), "r").convert('RGB')
    else:
        image = Image.open(image_input).convert('RGB')
    return image




def make_conversation(example):
    return json.dumps(
        [
            {
                "role": "system",
                "content": system_prompt_template.format(question=example[problem_key]),
            },
            {
                "role": "user",
                "content": question_template.format(question=example[problem_key]),
            },
        ],
    )


def make_conversation_image(example):
    return json.dumps(
        [
            {
                "role": "system",
                "content": system_prompt_template.format(question=example[problem_key]),
            },
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {
                        "type": "text",
                        "text": question_template.format(question=example[problem_key]),
                    },
                ],
            },
        ],
    )


def get_answer_from_completion(completion):
    answer = ""
    answer_pattern = r"<answer>(.*?)</answer>"
    answer_match = re.search(answer_pattern, completion, re.DOTALL)
    if not answer_match:
        answer_pattern = r"<answer>(.*?)</answer"
        answer_match = re.search(answer_pattern, completion, re.DOTALL)
    if answer_match:
        answer = answer_match.group(1).strip()
        answer = answer.replace("%", "")
        try:
            answer_int = int(answer)
            if answer_int < -100:
                answer = "-100"
            elif answer_int > 100:
                answer = "100"
        except Exception as e:
            answer_int = 0
        #
    return answer


def get_output_across_videos(
    video_count,
    text_input_list_batch,
    text_output_list_batch,
    example_list_batch,
    answer_list_batch,
):
    text_output_list_list = []
    text_input_list_list = []
    example_list_list = []
    answer_list_list = []
    for video_idx in range(video_count):
        text_input_list = [''] + [x[video_idx] for x in text_input_list_batch]
        text_output_list = [''] + [x[video_idx] for x in text_output_list_batch]
        example_list =  [{}] + [x[video_idx] for x in example_list_batch]
        answer_list = ['0'] + [x[video_idx] for x in answer_list_batch]
        #
        text_output_list_list.append(text_output_list)
        text_input_list_list.append(text_input_list)
        example_list_list.append(example_list)
        answer_list_list.append(answer_list)
    #
    return (
        text_output_list_list,
        text_input_list_list,
        example_list_list,
        answer_list_list,
    )



################################################################################################################################################################################################################################################







import json
import logging
import os
import uuid
from pathlib import Path
from typing import Any
# import hydra
import numpy as np
# from custom_inference import InferenceServer
# from omegaconf import DictConfig
from PIL import Image
# from sole_batch_decode2 import get_output_across_videos, load_model, sole_batch_decode
import os
import datetime
import imageio
import cv2
import torch
import random

# user_question_template = "Here is an image containing multiple camera views of a robot attempting to complete a task. " + \
# "The views on the left are from the robot's wrist camera. The views on the right are from an external camera. " + \
# "The views from the previous timestep are shown on the top row. The views from the current timestep are shown on the bottom row. " + \
# "The task description is: {task_description}. " + \
# "The task progress for the previous timestep is {prev_progress}%. Predict the task progress for the current timestep." 

user_question_template = "Here is an image containing multiple camera views of a robot attempting to complete a task. " + \
"The views on the top are from an external camera. The views on the bottom are from the robot's wrist camera. " + \
"The views from the very first timestep are shown to the left. The views from the previous timestep are shown in the middle. The views from the current timestep are shown to the right. " + \
"The task description is: {task_description}. " + \
"The task progress for the very first timestep is 0%. The task progress for the previous timestep is {prev_progress}%. Predict the task progress for the current timestep." 

user_question_template_external_view = "Here is an image containing multiple camera views of a robot attempting to complete a task. " + \
"The views from the very first timestep are shown to the left. The views from the previous timestep are shown in the middle. The views from the current timestep are shown to the right. " + \
"The task description is: {task_description}. " + \
"The task progress for the very first timestep is 0%. The task progress for the previous timestep is {prev_progress}%. Predict the task progress for the current timestep." 

user_question_template_wrist_view = "Here is an image containing multiple camera views of a robot attempting to complete a task. " + \
"The views are from the robot's wrist camera. " + \
"The views from the very first timestep are shown to the left. The views from the previous timestep are shown in the middle. The views from the current timestep are shown to the right. " + \
"The task description is: {task_description}. " + \
"The task progress for the very first timestep is 0%. The task progress for the previous timestep is {prev_progress}%. Predict the task progress for the current timestep." 


# question_no_prev_progress_template = "Here is an image containing multiple camera views of a robot attempting to complete a task. " + \
# "The views on the top are from an external camera. The views on the bottom are from the robot's wrist camera. " + \
# "The views from the very first timestep are shown to the left. The views from the previous timestep are shown in the middle. The views from the current timestep are shown to the right. " + \
# "The task description is: {task_description}. " + \
# "The task progress for the very first timestep is 0%. Predict the task progress for the current timestep." 

user_question_from_zero_template = "Here is an image containing multiple camera views of a robot attempting to complete a task. " + \
"The views on the top are from an external camera. The views on the bottom are from the robot's wrist camera. " + \
"The views from the very first timestep are shown to the left. The views from the current timestep are shown to the right. " + \
"The task description is: {task_description}. " + \
"The task progress for the very first timestep is 0%. Predict the task progress for the current timestep." 

user_question_from_zero_template_external_view = "Here is an image containing multiple camera views of a robot attempting to complete a task. " + \
"The views from the very first timestep are shown to the left. The views from the current timestep are shown to the right. " + \
"The task description is: {task_description}. " + \
"The task progress for the very first timestep is 0%. Predict the task progress for the current timestep." 


user_question_from_zero_template_wrist_view = "Here is an image containing multiple camera views of a robot attempting to complete a task. " + \
"The views are from the robot's wrist camera. " + \
"The views from the very first timestep are shown to the left. The views from the current timestep are shown to the right. " + \
"The task description is: {task_description}. " + \
"The task progress for the very first timestep is 0%. Predict the task progress for the current timestep." 



def resize_with_padding(img, size=384):
    h, w = img.shape[:2]
    # Determine scaling factor so max dimension == size
    scale = size / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)
    # Resize with preserved aspect ratio
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    # Create a black canvas 384x384
    output = np.zeros((size, size, 3), dtype=np.uint8)
    # Center the resized image on the canvas
    y_offset = (size - new_h) // 2
    x_offset = (size - new_w) // 2
    output[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    return output



def create_composite_frame( first_frame_wrist_view, 
                        first_frame_external_view,
                        frame0_wrist_view,
                        frame0_external_view,
                        frame1_wrist_view,
                        frame1_external_view,
                        use_two_timestep=True,
                           from_zero=False, 
                           view_type=['external', 'wrist']):
    size = 384
    padding = 5
    # 
    first_imgs = [first_frame_wrist_view, first_frame_external_view]
    imgs0 = [frame0_wrist_view, frame0_external_view]
    imgs2 = [frame1_wrist_view, frame1_external_view]
    # 
    for i in range(len(first_imgs)):
        first_imgs[i] = resize_with_padding(first_imgs[i], size)
        imgs0[i] = resize_with_padding(imgs0[i], size)
        imgs2[i] = resize_with_padding(imgs2[i], size)
    # 
    col_pad = np.zeros((size, padding, 3), dtype=np.uint8)
    external_view_idx = 1
    if not from_zero:
        bottom_row = np.hstack([first_imgs[0], col_pad, imgs0[0], col_pad, imgs2[0]])
        top_row = np.hstack([first_imgs[external_view_idx], col_pad, imgs0[external_view_idx], col_pad, imgs2[external_view_idx]])
    else:
        bottom_row = np.hstack([first_imgs[0], col_pad, imgs2[0]])
        top_row = np.hstack([first_imgs[external_view_idx], col_pad, imgs2[external_view_idx]])
    full_width = top_row.shape[1]
    row_pad = np.zeros((padding, full_width, 3), dtype=np.uint8)
    #
    if view_type == ['external']:
        composite = top_row
    elif view_type == ['wrist']:
        composite = bottom_row
    else:
        if use_two_timestep:
            composite = np.vstack([top_row, row_pad, bottom_row])
        else:
            assert False, "use_two_timestep==False not implemented"
    # 
    return composite


processing_class = None
processor = None
llm = None
sampling_params = None



def unload_model():
    import gc, torch

    for name in ["processing_class", "processor", "llm", "sampling_params"]:
        globals()[name] = None

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    
# def unload_model():
#     global processing_class, processor, llm, sampling_params
#     import gc
#     import torch
#     try: 
#         del processing_class
#     except:
#         pass
#     try:
#         del processor
#     except:
#         pass
#     try:
#         del llm
#     except:
#         pass
#     try:
#         del sampling_params
#     except:
#         pass
#     processing_class = None
#     processor = None
#     llm = None
#     sampling_params = None
#     gc.collect()
#     torch.cuda.empty_cache()
#     torch.cuda.ipc_collect()
    
def load_model():
    global processing_class, processor, llm, sampling_params
    if llm is None:
        # rbm.unload_model()
        print("Loading SOLE-R1 model and processor...")
        processing_class = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")
        # processor = AutoProcessor.from_pretrained("/data/sls/scratch/pschro/sole/checkpoints/checkpoint-85000")
        processor = AutoProcessor.from_pretrained("../model_checkpoints/SOLE-R1-8B")
        llm = LLM(
            # model="/data/sls/scratch/pschro/sole/checkpoints/checkpoint-85000",
            model="../model_checkpoints/SOLE-R1-8B",
            gpu_memory_utilization=0.95,
            max_model_len=16384,
            enable_prefix_caching=True,
        )
        sampling_params = SamplingParams(
            temperature=temperature_,
            top_p=0.9,
            top_k=50,
            max_tokens=200,
        )
        

def sole(videos, task_description, view_type_per_video=None, context_window = ['current', 'previous', 'first']):
    video_step_counts = [len(d) for d in videos]
    if len(set(video_step_counts)) != 1:
        logging.error(
            f"Episodes in batch have different lengths: {video_step_counts}. Using the minimum length."
        )
    # 
    video_step_count = int(np.min(video_step_counts)) - 1 # assuming 0 for first frame since this will always be 0 progress
    video_count = len(videos)
    # 
    dataset_dict_list = []
    for video_idx in range(len(videos)):
        # 
        if view_type_per_video[video_idx] == ['external and wrist']:
            user_question_template_final = user_question_template
            user_question_from_zero_template_final = user_question_from_zero_template
        elif view_type_per_video[video_idx] == ['wrist']:
            user_question_template_final = user_question_template_wrist_view
            user_question_from_zero_template_final = user_question_from_zero_template_wrist_view
        else:
            user_question_template_final = user_question_template_external_view
            user_question_from_zero_template_final = user_question_from_zero_template_external_view
        # 
        video = videos[video_idx]
        first_frame = video[0]
        frame_height, frame_width = first_frame.shape[:2]
        if view_type_per_video[video_idx] == 'external and wrist' or (view_type_per_video is None and frame_width == 2*frame_height):
            first_frame_external_view = first_frame[:, :first_frame.shape[1]//2, :]
            first_frame_wrist_view = first_frame[:, first_frame.shape[1]//2:, :]
        elif view_type_per_video[video_idx] == 'wrist':
            first_frame_wrist_view = first_frame
        else:
            first_frame_external_view = first_frame
        #
        for i in range(1, len(video)):
            prev_frame = video[i-1]
            current_frame = video[i]
            if view_type_per_video[video_idx] == 'external and wrist':
                prev_frame_external_view = prev_frame[:, :prev_frame.shape[1]//2, :]
                prev_frame_wrist_view = prev_frame[:, prev_frame.shape[1]//2:, :]
                current_frame_external_view = current_frame[:, :current_frame.shape[1]//2, :]
                current_frame_wrist_view = current_frame[:, current_frame.shape[1]//2:, :]
            elif view_type_per_video[video_idx] == 'wrist':
                prev_frame_external_view = None
                prev_frame_wrist_view = prev_frame
                current_frame_external_view = None
                current_frame_wrist_view = current_frame  
            else:
                prev_frame_external_view = prev_frame
                prev_frame_wrist_view = None
                current_frame_external_view = current_frame
                current_frame_wrist_view = None
            # 
            composite_frame = create_composite_frame( first_frame_wrist_view, first_frame_external_view, prev_frame_wrist_view, prev_frame_external_view, current_frame_wrist_view, current_frame_external_view, from_zero=False, view_type=view_type_per_video[video_idx])
            question_final = user_question_template_final.format(task_description=task_description, prev_progress=0)
            dataset_dict_list.append({
                "image": composite_frame,
                "question": question_final
            })
    # 
    # len(dataset_dict_list)
    # dataset_dict_list[0]['image'].shape
    # 
    # gpu_memory_utilization = 0.95
    # processing_class = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")
    # processor = AutoProcessor.from_pretrained("/data/sls/scratch/pschro/sole/checkpoints/checkpoint-85000")
    # llm = LLM(model="/data/sls/scratch/pschro/sole/checkpoints/checkpoint-85000", gpu_memory_utilization=0.95, max_model_len=16384,   enable_prefix_caching=True,)
    # sampling_params = SamplingParams(temperature=temperature_, top_p=0.9, top_k=50,max_tokens=200, ) # max_tokens=self.max_completion_length,
    load_model()   # ensures model is loaded once
    global processing_class, processor, llm, sampling_params
    # 
    # test_image = load_image(dataset_dict_list[0]['image'])
    # test_image.save('/data/sls/scratch/pschro/sole/test_image.jpg')
    # 
    text_input_list_batch, text_output_list_batch, example_list_batch, answer_list_batch = sole_batch_decode(processing_class, processor, llm, sampling_params, dataset_dict_list, video_count, video_step_count,)
    # 
    text_output_list_list, text_input_list_list, example_list_list, answer_list_list = get_output_across_videos(video_count, text_input_list_batch, text_output_list_batch, example_list_batch, answer_list_batch )
    # 
    # [len(x) for x in text_output_list_list]
    # [len(x) for x in text_input_list_list]
    # [len(x) for x in example_list_list]
    # [len(x) for x in answer_list_list]
    # 
    # text_output_list_list[0][0]
    # 
    valid_answer_list_list = []
    for episode in answer_list_list:
        valid_answer_list = []  
        for ans in episode:
            try:
                progress = int(ans)
            except (ValueError, TypeError):
                if len(valid_answer_list) > 0:
                    progress = valid_answer_list[-1]
                else:
                    progress = 0
            valid_answer_list.append(progress)
        valid_answer_list_list.append(valid_answer_list)
    # 
    return valid_answer_list_list, text_output_list_list







