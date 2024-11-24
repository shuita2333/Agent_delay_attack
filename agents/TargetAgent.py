import json
import copy
import re
from agents.BaseAgent import BaseAgent
from prompt.messages import add_tool_answer
from prompt.system_prompts import get_target_agent_system_prompt, get_targetAgent_agent_system_prompt
import openai
import time
import traceback
from json import JSONDecodeError

from utils.tool_invoke import invoke


class TargetAgent(BaseAgent):
    def _get_system_message(self):
        return get_target_agent_system_prompt(self.args)

    def _extract_json(self, s):
        # 解析整个返回值字符串为一个字典
        response = json.loads(s)
        try:
            # 提取content字段中的嵌套JSON字符串
            prompt_length = response['usage']['prompt_tokens']
            content_str = response['choices'][0]['message']['content']
            content_length = response['usage']['completion_tokens']
        except KeyError as e:
            print(f"KeyError! : {e}")
            raise KeyError
        return {'content_str': content_str, 'content_length': content_length,'prompt_length': prompt_length}, None


    # def _iterative_try_get_proper_format(self, conv_list, full_prompts, indices_to_regenerate):
    #     batch_size = len(conv_list)
    #     valid_outputs = [None] * batch_size
    #     valid_times = [None] * batch_size
    #
    #     # 基于索引重新生成会话子集
    #     full_prompts_subset = [full_prompts[i] for i in indices_to_regenerate]
    #
    #     for i, full_prompt in enumerate(full_prompts_subset):
    #         valid_outputs[i], valid_times[i] = self.Agent_work(full_prompt,indices_to_regenerate)
    #
    #     if any([output for output in valid_outputs if output is None]):
    #         print(f"Failed to generate output after {self.max_n_attack_attempts} attempts. Terminating.")
    #     return valid_outputs, valid_times


    def _extract(self, nested_json):
        pass

    # def Agent_work(self, full_prompt,indices_to_regenerate):
    #     agent_system_prompt = get_targetAgent_agent_system_prompt()
    #     Agent_prompt = copy.deepcopy(full_prompt)
    #     Agent_prompt_user_message=Agent_prompt.messages[0][1]
    #     Agent_prompt.messages = []
    #     Agent_prompt_user_message=agent_system_prompt+"\n\n"+Agent_prompt_user_message
    #     Agent_prompt.append_message(Agent_prompt.roles[0],Agent_prompt_user_message)
    #     batch_size = 1
    #     valid_outputs = [None] * batch_size
    #     valid_times = [None] * batch_size
    #     for attempt in range(self.max_n_attack_attempts):
    #         # 生成输出
    #         outputs_list, output_times = self.model.batched_generate([Agent_prompt],
    #                                                                  max_n_tokens=self.max_n_tokens,
    #                                                                  temperature=self.temperature,
    #                                                                  top_p=self.top_p
    #                                                                  )
    #         # 检查有效的输出并更新列表
    #         new_indices_to_regenerate = []
    #         for i, full_output in enumerate(outputs_list):
    #             orig_index = indices_to_regenerate[i]
    #             try:
    #                 attack_dict, json_str = self._extract_json(full_output)
    #                 pattern = r'\{\s*"Action":\s*".*?",\s*"Action_Input":\s*(\{.*?\}|\s*".*?")\s*\}'
    #                 match = re.search(pattern, attack_dict['content_str'], re.DOTALL)
    #                 target_work=json.loads(match.group(0))
    #                 if target_work["Action"] == "tool":
    #                     invoke_result = invoke(target_work["Action_Input"])
    #                 if target_work["Action"] == "answer" or target_work["Action"] == "tool":
    #                     if target_work["Action"] == "tool":
    #                         full_prompt_append = add_tool_answer(invoke_result,
    #                                                        target_work["Action_Input"]["tool_name"])
    #                         full_prompt_user_message=full_prompt.messages[0][1]
    #                         full_prompt.messages=[]
    #                         full_prompt_user_message=full_prompt_append+full_prompt_user_message
    #                         full_prompt.append_message(full_prompt.roles[0], full_prompt_user_message)
    #                     valid_output, valid_time = self.get_answer_generate(full_prompt,list(range(1)))
    #                 else:
    #                     raise JSONDecodeError("Action format error", target_work["Action"])
    #             except (JSONDecodeError, KeyError, TypeError) as e:
    #                 # 如果出现异常，重新输出
    #                 traceback.print_exc()
    #                 print(f"index 为{orig_index} 解析过程中出现了异常:{e}.正在重新生成。。。。。")
    #                 new_indices_to_regenerate.append(orig_index)
    #         # 更新索引以为下一次迭代重新生成索引
    #         indices_to_regenerate = new_indices_to_regenerate
    #         # If all outputs are valid, break
    #         if not indices_to_regenerate:
    #             break
    #     return valid_output, valid_time
    #
    # def get_answer_generate(self, Agent_prompt,indices_to_regenerate):
    #     global valid_time
    #     for attempt in range(self.max_n_attack_attempts):
    #         # 生成输出
    #         outputs_list, output_times = self.model.batched_generate([Agent_prompt],
    #                                                                  max_n_tokens=self.max_n_tokens,
    #                                                                  temperature=self.temperature,
    #                                                                  top_p=self.top_p
    #                                                                  )
    #
    #         # 检查有效的输出并更新列表
    #         new_indices_to_regenerate = []
    #         for i, full_output in enumerate(outputs_list):
    #             orig_index = indices_to_regenerate[i]
    #             try:
    #                 attack_dict, json_str = self._extract_json(full_output)
    #                 valid_output = attack_dict
    #                 valid_time = output_times[i]
    #             except (JSONDecodeError, KeyError, TypeError) as e:
    #                 # 如果出现异常，重新输出
    #                 traceback.print_exc()
    #                 print(f"index 为{orig_index} 解析过程中出现了异常:{e}.正在重新生成。。。。。")
    #                 new_indices_to_regenerate.append(orig_index)
    #
    #         # 更新索引以为下一次迭代重新生成索引
    #         indices_to_regenerate = new_indices_to_regenerate
    #
    #         # If all outputs are valid, break
    #         if not indices_to_regenerate:
    #             break
    #     return valid_output, valid_time
