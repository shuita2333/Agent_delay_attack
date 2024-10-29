import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import re
from config import VICUNA_PATH, LLAMA_PATH, ATTACK_TEMP, TARGET_TEMP, ATTACK_TOP_P, TARGET_TOP_P
import common
from language_models import GPT, Siliconflow
import json

from system_prompts import get_subtask_attacker_system_prompt, get_target_identity


def load_attack_and_target_models(args):
    '''
    加载攻击模型和 tokenizer
    :param args:
    :return:
    '''
    attackLM = AttackLM(model_name=args.attack_model,
                        max_n_tokens=args.attack_max_n_tokens,
                        max_n_attack_attempts=args.max_n_attack_attempts,
                        temperature=ATTACK_TEMP,  # init to 1
                        top_p=ATTACK_TOP_P,  # init to 0.9
                        )
    preloaded_model = None
    if args.attack_model == args.target_model:
        print("Using same attack and target model. Using previously loaded model.")
        preloaded_model = attackLM.model
    targetLM = TargetLM(model_name=args.target_model,
                        max_n_tokens=args.target_max_n_tokens,
                        temperature=TARGET_TEMP,  # init to 0
                        top_p=TARGET_TOP_P,  # init to 1
                        preloaded_model=preloaded_model,
                        )
    return attackLM, targetLM


class AttackLM():
    def __init__(self,
                 model_name: str,
                 max_n_tokens: int,
                 max_n_attack_attempts: int,
                 temperature: float,
                 top_p: float):
        self.model_name = model_name
        self.temperature = temperature
        self.max_n_tokens = max_n_tokens
        self.max_n_attack_attempts = max_n_attack_attempts
        self.top_p = top_p
        self.model, self.template = load_indiv_model(model_name)

        if "vicuna" in model_name or "llama" in model_name:
            self.model.extend_eos_tokens()

    def get_attack(self, convs_list, prompts_list):
        """
        使用语言模型为一批对话和提示生成响应。
        只返回正确JSON格式的有效输出。
        如果没有生成输出在max_n_attack_attempts之后成功返回None。

        Parameters:
        - convs_list: List of conversation objects.
        - prompts_list: List of prompts corresponding to each conversation.

        Returns:
        - List of generated outputs (dictionaries) or None for failed generations.
        """
        # 对话和提示的数量不匹配。
        assert len(convs_list) == len(prompts_list), "Mismatch between number of conversations and prompts."

        batchsize = len(convs_list)
        indices_to_regenerate = list(range(batchsize))
        valid_outputs = [None] * batchsize

        # # 初始化攻击模型生成的输出以匹配格式
        # if len(convs_list[0].messages) == 0:
        #     init_message = """{\"improvement\": \"\",\"prompt\": \""""
        # else:
        #     init_message = """{\"improvement\": \""""

        full_prompts = []
        # 向对话添加提示和初始播种消息（仅一次）
        for conv, prompt in zip(convs_list, prompts_list):
            conv.append_message(conv.roles[0], prompt)
            # Get prompts(需要根据模型修改)
            if "gpt" in self.model_name:
                full_prompts.append(conv.to_openai_api_messages())
            else:
                # conv.append_message(conv.roles[1], init_message)
                # full_prompts.append(conv.get_prompt()[:-len(conv.sep2)])
                full_prompts.append(conv.get_prompt()[:-len(conv.sep)])

        for attempt in range(self.max_n_attack_attempts):
            # 基于索引重新生成会话子集
            full_prompts_subset = [full_prompts[i] for i in indices_to_regenerate]

            # 生成输出
            outputs_list = self.model.batched_generate(full_prompts_subset,
                                                       max_n_tokens=self.max_n_tokens,
                                                       temperature=self.temperature,
                                                       top_p=self.top_p
                                                       )

            # 检查有效的输出并更新列表
            new_indices_to_regenerate = []
            for i, full_output in enumerate(outputs_list):
                orig_index = indices_to_regenerate[i]
                # if "gpt" not in self.model_name:
                #     full_output = init_message + full_output

                attack_dict, json_str = common.my_extract_json(full_output)

                if attack_dict is not None:
                    valid_outputs[orig_index] = attack_dict
                    convs_list[orig_index].update_last_message(
                        json_str)  # 使用有效生成更新对话
                else:
                    new_indices_to_regenerate.append(orig_index)

            # 更新索引以为下一次迭代重新生成索引
            indices_to_regenerate = new_indices_to_regenerate

            # If all outputs are valid, break
            if not indices_to_regenerate:
                break

        if any([output for output in valid_outputs if output is None]):
            print(f"Failed to generate output after {self.max_n_attack_attempts} attempts. Terminating.")
        return valid_outputs

    def get_integrate_attack(self, conv, prompt):
        conv.append_message(conv.roles[0], prompt)
        full_prompts = []
        if "gpt" in self.model_name:
            full_prompts.append(conv.to_openai_api_messages())
        else:
            # conv.append_message(conv.roles[1], init_message)
            # full_prompts.append(conv.get_prompt()[:-len(conv.sep2)])
            full_prompts.append(conv.get_prompt()[:-len(conv.sep)])

        outputs_list = self.model.batched_generate(full_prompts,
                                                   max_n_tokens=self.max_n_tokens,
                                                   temperature=self.temperature,
                                                   top_p=self.top_p
                                                   )

        outputs = outputs_list[0]
        if "Qwen" in self.model_name:
            prompt = common.extract_integrate_json(outputs)

        return prompt

    def get_background(self, convs_list, prompts_list):
        assert len(convs_list) == len(prompts_list), "Mismatch between number of conversations and prompts."

        batchsize = len(convs_list)
        indices_to_regenerate = list(range(batchsize))
        valid_outputs = [None] * batchsize

        full_prompts = []
        request_prompt = []
        length_lest = []
        # 向对话添加提示和初始播种消息（仅一次）
        for conv, prompt in zip(convs_list, prompts_list):
            conv.append_message(conv.roles[0], prompt)
            # Get prompts(需要根据模型修改)
            if "gpt" in self.model_name:
                full_prompts.append(conv.to_openai_api_messages())
            else:
                full_prompts.append(conv.get_prompt()[:-len(conv.sep)])

        # 基于索引重新生成会话子集
        full_prompts_subset = [full_prompts[i] for i in indices_to_regenerate]

        # 生成输出
        outputs_list = self.model.batched_generate(full_prompts_subset,
                                                   max_n_tokens=self.max_n_tokens,
                                                   temperature=self.temperature,
                                                   top_p=self.top_p
                                                   )

        # 检查有效的输出并更新列表
        for full_output in outputs_list:
            output, length = siliconCloud_outputs_list_extracted(full_output)
            request_prompt.append(output)
            length_lest.extend([length])
        return request_prompt, length_lest


class TargetLM():
    """
        目标模型类
        使用语言模型为提示生成响应.
        self.model 包括底层模型.
    """

    def __init__(self,
                 model_name: str,
                 max_n_tokens: int,
                 temperature: float,
                 top_p: float,
                 preloaded_model: object = None):

        self.model_name = model_name
        self.temperature = temperature
        self.max_n_tokens = max_n_tokens
        self.top_p = top_p
        if preloaded_model is None:
            self.model, self.template = load_indiv_model(model_name)
        else:
            self.model = preloaded_model
            _, self.template = get_model_path_and_template(model_name)

    def get_response(self, prompts_list, target_identity):
        batchsize = len(prompts_list)
        convs_list = [common.conv_template(self.template) for _ in range(batchsize)]
        full_prompts = []
        for conv, prompt in zip(convs_list, prompts_list):
            conv.append_message(conv.roles[0], prompt)
            if "gpt" in self.model_name:
                # Openai没有分隔符
                full_prompts.append(conv.to_openai_api_messages())
            elif "palm" in self.model_name:
                full_prompts.append(conv.messages[-1][1])
            else:
                conv.append_message(conv.roles[1], None)
                conv.append_message(conv.system_message, target_identity)
                full_prompts.append(conv.get_prompt())
        outputs_list = self.model.batched_generate(full_prompts,
                                                   max_n_tokens=self.max_n_tokens,
                                                   temperature=self.temperature,
                                                   top_p=self.top_p
                                                   )

        outputs = []

        outputs_length = []
        if "Qwen" in self.model_name:
            for output in outputs_list:
                print(output)
                output, length = siliconCloud_outputs_list_extracted(output)
                outputs_length.append(length)
                outputs.append(output)
        return outputs, outputs_length


def siliconCloud_outputs_list_extracted(outputs_list):
    # 将JSON数据加载为字典
    data = json.loads(outputs_list)

    # 提取choices列表中的第一个元素的message内容
    message = None
    completion_tokens = None
    prompt_content = None

    if 'choices' in data and len(data['choices']) > 0:
        message = data['choices'][0]['message']['content']

        # 使用正则表达式提取 "prompt" 的内容
        match = re.search(r'"prompt":\s*"(.+?)"', message, re.DOTALL)
        if match:
            prompt_content = match.group(1)

    # 提取completion_tokens的值
    if 'usage' in data and 'completion_tokens' in data['usage']:
        completion_tokens = data['usage']['completion_tokens']

    return prompt_content, completion_tokens


def subtask_generate(args, extracted_integrate_attack_prompt, attackLM, targetLM):
    subtask_prompt_list = []
    # #下一步通过总任务的prompt生成子任务
    batchsize = args.n_question
    subtask_list = [common.get_subtask_init_msg(extracted_integrate_attack_prompt["total_prompt"], prompt) for prompt in
                    extracted_integrate_attack_prompt["subtask_question"]]
    convs_list = [common.conv_template(attackLM.template) for _ in range(batchsize)]
    subtask_system_prompt = get_subtask_attacker_system_prompt(args.function_descript, args.target_length,
                                                               extracted_integrate_attack_prompt["total_prompt"])

    for conv in convs_list:
        ## 这里替换系统提示内容
        conv.set_system_message(subtask_system_prompt)

    if args.n_iterations == 1:
        print(f"""\n{'=' * 36}\nIteration: 1\n{'=' * 36}\n""")

        # 获得对抗性 prompt 和改进
        extracted_attack_list = attackLM.get_attack(convs_list, subtask_list)
        print("Finished getting adversarial prompts.")

        # 提取 prompt 和改进
        adv_prompt_list = [attack["prompt"] for attack in extracted_attack_list]

        subtask_prompt_list = adv_prompt_list


    elif args.n_iterations == 2:
        last_adv_prompt_list = []
        last_target_response_list = []
        last_target_response_length = []

        for iteration in range(1, args.n_iterations + 1):
            print(f"""\n{'=' * 36}\nIteration: {iteration}\n{'=' * 36}\n""")
            if iteration > 1:
                subtask_list = [
                    common.subtask_process_target_response(adv_prompt, target_response,
                                                           extracted_integrate_attack_prompt["total_prompt"], prompt,
                                                           length) for
                    adv_prompt, target_response, length, prompt in
                    zip(last_adv_prompt_list, last_target_response_list, last_target_response_length,
                        extracted_integrate_attack_prompt["subtask_question"])]

            # 获得对抗性 prompt 和改进
            extracted_attack_list = attackLM.get_attack(convs_list, subtask_list)
            print("Finished getting adversarial prompts.")

            # 提取 prompt 和改进
            adv_prompt_list = [attack["prompt"] for attack in extracted_attack_list]
            improv_list = [attack["improvement"] for attack in extracted_attack_list]

            # 获得目标响应
            target_identity = get_target_identity(args.function_descript)
            target_response_list, target_response_length = targetLM.get_response(adv_prompt_list, target_identity)
            print(f"Finished getting target responses.\ntarget_response_length: {target_response_length}")

            if iteration == 1:
                last_adv_prompt_list = adv_prompt_list
                last_target_response_list = target_response_list
                last_target_response_length = target_response_length

            if iteration == args.n_iterations:
                for i in range(len(target_response_length)):
                    # 比较两个length列表中的对应值
                    if target_response_length[i] >= last_target_response_length[i]:
                        # 如果 target_response_length 的值较大或相等，存入 adv_prompt_list 中的对应元素
                        subtask_prompt_list.append(adv_prompt_list[i])
                    else:
                        # 如果 last_target_response_length 的值较大，存入 last_adv_prompt_list 中的对应元素
                        subtask_prompt_list.append(last_adv_prompt_list[i])

    return subtask_prompt_list


def load_indiv_model(model_name):
    '''
    构造模型，定义模板名称
    :param self:
    :param model_name:
    :return:
    '''
    model_path, template = get_model_path_and_template(model_name)
    if model_name in ["gpt-3.5-turbo", "gpt-4"]:
        lm = GPT(model_name)
    elif model_name in ["Qwen2.5-7B"]:
        lm = Siliconflow(model_path)
    # elif model_name in ["claude-2", "claude-instant-1"]:
    #     lm = Claude(model_name)
    # elif model_name in ["palm-2"]:
    #     lm = PaLM(model_name)
    # else:
    #     model = AutoModelForCausalLM.from_pretrained(
    #         model_path,
    #         torch_dtype=torch.float16,
    #         low_cpu_mem_usage=True, device_map="auto").eval()
    #
    #     tokenizer = AutoTokenizer.from_pretrained(
    #         model_path,
    #         use_fast=False
    #     )
    #
    #     if 'llama-2' in model_path.lower():
    #         tokenizer.pad_token = tokenizer.unk_token
    #         tokenizer.padding_side = 'left'
    #     if 'vicuna' in model_path.lower():
    #         tokenizer.pad_token = tokenizer.eos_token
    #         tokenizer.padding_side = 'left'
    #     if not tokenizer.pad_token:
    #         tokenizer.pad_token = tokenizer.eos_token
    #
    #     lm = HuggingFace(model_name, model, tokenizer)

    return lm, template


def get_model_path_and_template(model_name):
    '''
    定义模型路径
    :param model_name:
    :return:
    '''
    full_model_dict = {
        "gpt-4": {
            "path": "gpt-4",
            "template": "gpt-4"
        },
        "Qwen2.5-7B": {
            "path": "Qwen/Qwen2.5-7B-Instruct",
            "template": "Qwen2"
        }
        # "gpt-3.5-turbo": {
        #     "path":"gpt-3.5-turbo",
        #     "template":"gpt-3.5-turbo"
        # },
        # "vicuna":{
        #     "path":VICUNA_PATH,
        #     "template":"vicuna_v1.1"
        # },
        # "llama-2":{
        #     "path":LLAMA_PATH,
        #     "template":"llama-2"
        # },
        # "claude-instant-1":{
        #     "path":"claude-instant-1",
        #     "template":"claude-instant-1"
        # },
        # "claude-2":{
        #     "path":"claude-2",
        #     "template":"claude-2"
        # },
        # "palm-2":{
        #     "path":"palm-2",
        #     "template":"palm-2"
        # }
    }
    path, template = full_model_dict[model_name]["path"], full_model_dict[model_name]["template"]
    return path, template
