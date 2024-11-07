import json
import re
import traceback
from abc import ABC, abstractmethod
from json import JSONDecodeError

from utils.conversers import load_indiv_model, conv_template
import copy


class BaseAgent(ABC):
    def __init__(self,
                 model_name: str,
                 max_n_tokens: int,
                 max_n_attack_attempts: int,
                 temperature: float,
                 top_p: float,
                 args=None):
        self.model_name = model_name
        self.temperature = temperature
        self.max_n_tokens = max_n_tokens
        self.max_n_attack_attempts = max_n_attack_attempts
        self.top_p = top_p
        self.model, self.template = load_indiv_model(model_name)
        self.args = args
        if "vicuna" in model_name or "llama" in model_name:
            self.model.extend_eos_tokens()

    def get_conv_list(self, batch_size):
        conv_list = [conv_template(self.template) for _ in range(batch_size)]
        for conv in conv_list:
            conv.set_system_message(self._get_system_message())
        return conv_list

    @abstractmethod
    def _get_system_message(self):
        pass

    def get_response(self, conv_list, prompts_list):
        """
        使用语言模型为一批对话和提示生成响应。
        只返回正确JSON格式的有效输出。
        如果没有生成输出在max_n_attack_attempts之后成功返回None。

        Parameters:
        - conv_list: List of conversation objects.
        - prompts_list: List of prompts corresponding to each conversation.

        Returns:
        - List of generated outputs (dictionaries) or None for failed generations.
        """
        # 对话和提示的数量不匹配。
        assert len(conv_list) == len(prompts_list), "Mismatch between number of conversations and prompts."
        batch_size = len(conv_list)
        indices_to_regenerate = list(range(batch_size))
        full_prompts = []
        # 向对话添加提示和初始播种消息（仅一次）
        for conv, prompt in zip(conv_list, prompts_list):
            # Get prompts (需要根据模型修改)
            if "gpt" in self.model_name:
                full_prompts.append(conv.to_openai_api_messages())
            else:
                full_prompts.append(conv)

        return self._iterative_try_get_proper_format(conv_list, full_prompts, indices_to_regenerate)

    def _iterative_try_get_proper_format(self, conv_list, full_prompts, indices_to_regenerate):
        batch_size = len(conv_list)
        valid_outputs = [None] * batch_size
        valid_times = [None] * batch_size
        for attempt in range(self.max_n_attack_attempts):
            # 基于索引重新生成会话子集
            full_prompts_subset = [full_prompts[i] for i in indices_to_regenerate]

            # 生成输出
            outputs_list, output_times = self.model.batched_generate(full_prompts_subset,
                                                                     max_n_tokens=self.max_n_tokens,
                                                                     temperature=self.temperature,
                                                                     top_p=self.top_p
                                                                     )

            # 检查有效的输出并更新列表
            new_indices_to_regenerate = []
            for i, full_output in enumerate(outputs_list):
                orig_index = indices_to_regenerate[i]
                try:
                    attack_dict, json_str = self._extract_json(full_output)
                    valid_outputs[orig_index] = attack_dict
                    valid_times[orig_index] = output_times[i]
                    conv_list[orig_index].append_message(conv_list[orig_index].roles[1], json_str)  # 使用有效生成更新对话
                except (JSONDecodeError, KeyError, TypeError) as e:
                    # 如果出现异常，重新输出
                    traceback.print_exc()
                    print(f"index 为{orig_index} 解析过程中出现了异常:{e}.正在重新生成。。。。。")
                    new_indices_to_regenerate.append(orig_index)

            # 更新索引以为下一次迭代重新生成索引
            indices_to_regenerate = new_indices_to_regenerate

            # If all outputs are valid, break
            if not indices_to_regenerate:
                break

        if any([output for output in valid_outputs if output is None]):
            print(f"Failed to generate output after {self.max_n_attack_attempts} attempts. Terminating.")
        return valid_outputs,valid_times

    def _extract_json(self, s):
        """
            按照指定格式提取
            Args:
                s (str): The string containing the potential JSON structure.

            Returns:
                dict: A dictionary containing the extracted values.包含提取值的字典。
                str: The cleaned JSON string.清理后的JSON字符串。
            """
        # 解析整个返回值字符串为一个字典
        response = json.loads(s)

        try:
            # 提取content字段中的嵌套JSON字符串
            content_str = response['choices'][0]['message']['content']
        except KeyError as e:
            traceback.print_exc()
            print(f"KeyError! content_str: {response}")
            raise KeyError

        # 去除外围的```json\n和\n```部分
        json_str = content_str.strip('```json\n').strip('\n```').strip()
        # 去掉所有反斜杠
        json_str = re.sub(r'\\', '', json_str)
        # 解析嵌套的JSON字符串
        json_str = re.sub(r'[\x00-\x1F\x7F]', '', json_str)
        if '{{' in json_str and '}}' in json_str:
            json_str = json_str.replace('{{', '{').replace('}}', '}')
        if json_str.endswith("."):
            json_str = json_str + '"}'
        elif json_str.endswith('"'):
            # 如果字符串以 '"' 结尾但缺少 '}'
            json_str = json_str + '}'
        elif json_str.endswith('}'):
            # 检查末尾是否符合 ]\s*} 的格式
            if not re.search(r'\]\s*}$', json_str):
                # 使用正则表达式处理，确保在 } 一定有一个 *
                json_str = re.sub(r'([^\s"])(\s*)(})$', r'\1"\3', json_str)
        # 如果最后一个json元素有逗号，去除
        json_str = re.sub(r',\s*}', '}', json_str)
        # 提取最外层的{}里面的内容
        json_str = re.sub(r'^(.*?)(\{.*\})(.*?$)', r'\2', json_str, flags=re.DOTALL)
        try:
            nested_json = json.loads(json_str)
            return self._extract(nested_json)
        except JSONDecodeError:
            print(f"JSONDecodeError! Attempted to decode: {json_str}")
            raise JSONDecodeError

    @abstractmethod
    def _extract(self, nested_json):
        pass
