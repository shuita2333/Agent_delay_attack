import json

from agents.BaseAgent import BaseAgent
from prompt.system_prompts import get_target_agent_system_prompt


class TargetAgent(BaseAgent):
    def _get_system_message(self):
        get_target_agent_system_prompt(self.args.function_descript)

    def _extract_json(self, s):
        # 解析整个返回值字符串为一个字典
        response = json.loads(s)
        try:
            # 提取content字段中的嵌套JSON字符串
            content_str = response['choices'][0]['message']['content']
            content_length = response['usage']['completion_tokens']
        except KeyError as e:
            print(f"KeyError! : {e}")
            raise KeyError
        return {'content_str': content_str, 'content_length': content_length}, None

    def _extract(self, nested_json):
        pass
