import json
from abc import abstractmethod, ABC

import openai
import os
import time
import requests
from typing import Dict, List

from fastchat.conversation import Conversation
from openai import OpenAI

from API_key import Siliconflow_BASE_URL, Siliconflow_API_KEY, Mistral_API, Mistral_BASE_URL, OPENAI_API_KEY, \
    ALIYUN_API_KEY, ALIYUN_BASE_URL


class GPT:
    API_RETRY_SLEEP = 10
    API_ERROR_OUTPUT = "$ERROR$"
    API_QUERY_SLEEP = 0.5
    API_MAX_RETRY = 5
    API_TIMEOUT = 10000

    def __init__(self, model_name):
        self.model_name = model_name
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.max_tokens = 16384

    def generate(self, conv: Conversation,
                 max_n_tokens: int,
                 temperature: float,
                 top_p: float):
        for _ in range(self.API_MAX_RETRY):
            try:
                start_time = time.time()
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=conv.to_openai_api_messages(),
                    max_tokens=self.max_tokens,
                    temperature=temperature,
                    timeout=self.API_TIMEOUT
                )
                end_time = time.time()
                sum_time = end_time - start_time
                # response = openai.ChatCompletion.create(
                #     model=self.model_name,
                #     messages=conv,
                #     max_tokens=max_n_tokens,
                #     temperature=temperature,
                #     top_p=top_p,
                #     request_timeout=self.API_TIMEOUT,
                # )
                # # 检查 response 对象的结构
                # choices = response.choices
                # output = choices[0].message.content
                break
            except openai.OpenAIError as e:
                print(f"An error occurred: {e}")
                time.sleep(self.API_RETRY_SLEEP)
            time.sleep(self.API_QUERY_SLEEP)
        return response.to_json(), sum_time

    def batched_generate(self,
                         conv_list: List[List[Dict]],
                         max_n_tokens: int,
                         temperature: float,
                         top_p: float = 1.0, ):
        texts = []
        times = []
        output_content = []
        for conv in conv_list:
            text, sum_time = self.generate(conv, max_n_tokens, temperature, top_p)
            texts.append(text)
            times.append(float(sum_time))
        return texts, times

class StreamGPT(GPT):
    def __init__(self,model_name):
        super().__init__(model_name)
        self.client = OpenAI(api_key=ALIYUN_API_KEY, base_url=ALIYUN_BASE_URL)

    def generate(self, conv: Conversation,
                 max_n_tokens: int,
                 temperature: float,
                 top_p: float):
        for _ in range(self.API_MAX_RETRY):
            try:
                full_content = ""  # 用来累积所有的 chunk 内容
                content_length = 0  # 用来记录 total token 数
                response = None  # 最终的完整 response 对象
                start_time = time.time()
                response_stream = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=conv.to_openai_api_messages(),
                    max_tokens=self.max_tokens,
                    temperature=temperature,
                    stream=True,
                    stream_options={"include_usage": True}
                )
                # 遍历流式返回的 chunk
                for chunk in response_stream:
                    # 逐个解析 chunk，提取内容
                    chunk_data = chunk.model_dump_json()  # 或者根据你的具体格式来解析
                    chunk_json = json.loads(chunk_data)
                    if chunk_json['choices'] is not None and len(chunk_json['choices'])>0 and chunk_json['choices'][0] is not None:
                        full_content += chunk_json['choices'][0]['delta']['content']
                    if chunk_json['usage'] is not None:
                        completion_length = chunk_json['usage']['completion_tokens']
                        prompt_length = chunk_json['usage']['prompt_tokens']
                    # 构造最终的 response 对象，模拟你的原始结构
                end_time = time.time()
                response = {
                    'choices': [{
                        'message': {'content': full_content}
                    }],
                    'usage': {'completion_tokens': completion_length,
                              'prompt_tokens': prompt_length
                              }
                }
                sum_time = end_time - start_time
                break
            except openai.OpenAIError as e:
                print(f"An error occurred: {e}")
                time.sleep(self.API_RETRY_SLEEP)
            time.sleep(self.API_QUERY_SLEEP)
        return json.dumps(response), sum_time

class Qwen(StreamGPT):
    def __init__(self,model_name):
        super().__init__(model_name)
        self.model_name = model_name.lower()+"-instruct"
        self.max_tokens = 8192

class Llama(StreamGPT):
    def __init__(self, model_name):
        super().__init__(model_name)
        self.model_name = "llama" + model_name.strip("Meta-Llama-").lower() + "-instruct"
        self.max_tokens = 8192


class Api(ABC):
    def __init__(self, api_key, base_url, model_name):
        self.api_key = api_key
        self.base_url = base_url
        self.model_name = model_name

    def batched_generate(self,
                         conv_list: List[Conversation],
                         max_n_tokens: int,
                         temperature: float,
                         top_p: float = 1.0, ):
        """
        模型调用规范接口
        :param conv_list:
        :param max_n_tokens:
        :param temperature:
        :param top_p:
        :return:

        Args:
            conv_list:
        """
        texts = []
        times = []
        for conv in conv_list:
            text, sum_time = self._call(conv, max_n_tokens, temperature, top_p)
            texts.append(text)
            times.append(float(sum_time))
        return texts, times

    def _call(self, conv: Conversation,
              max_n_tokens: int,
              temperature: float,
              top_p: float) -> str:
        payload, headers = self.completions(conv=conv, max_n_tokens=max_n_tokens,
                                            temperature=temperature, top_p=top_p)
        start_time = time.time()
        response = requests.request("POST", self.base_url, json=payload, headers=headers)
        end_time = time.time()
        sum_time = end_time - start_time
        if self.base_url == Mistral_BASE_URL:
            time.sleep(1)
        # if stop is not None:
        #     response = enforce_stop_tokens(response, stop)

        return response.text, sum_time

    def process_conv(self, conv: Conversation):
        message = [{"role": "system",
                    "content": conv.system_message}]
        index = 0
        roles = ["user", "assistant"]
        for (r, m) in conv.messages:
            message.append({"role": roles[index],
                            "content": m})
            index = (index + 1) % 2

        return message

    @abstractmethod
    def completions(self, conv: Conversation,
                    max_n_tokens: int,
                    temperature: float,
                    top_p: float):
        pass


class Siliconflow(Api):

    def __init__(self, model_name):
        super().__init__(Siliconflow_API_KEY, Siliconflow_BASE_URL, model_name)

    def completions(self,
                    conv: Conversation,
                    max_n_tokens: int,
                    temperature: float,
                    top_p: float):
        """
        api 调用指令
        :param prompt:
        :param max_n_tokens:
        :param temperature:
        :param top_p:
        :return:
        """
        payload = {
            "model": self.model_name,
            "messages": self.process_conv(conv),
            "stream": False,
            "max_tokens": max_n_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        return payload, headers


class Mistral(Api):

    def __init__(self, model_name):
        super().__init__(Mistral_API, Mistral_BASE_URL, model_name)

    def completions(self,
                    conv: Conversation,
                    max_n_tokens: int,
                    temperature: float,
                    top_p: float):
        payload = {
            "model": self.model_name,
            "messages": self.process_conv(conv),
            "stream": False,
            "max_tokens": max_n_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        return payload, headers
