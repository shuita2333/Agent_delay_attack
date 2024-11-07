import json
from abc import abstractmethod, ABC

import openai
import os
import time
import requests
from typing import Dict, List

from fastchat.conversation import Conversation
from openai import OpenAI

from API_key import Siliconflow_BASE_URL, Siliconflow_API_KEY, Mistral_API, Mistral_BASE_URL, OPENAI_API_KEY


# import google.generativeai as palm
# from langchain.llms.base import LLM
# from langchain_core.callbacks import CallbackManagerForLLMRun


class LanguageModel:
    def __init__(self, model_name):
        self.model_name = model_name

    def batched_generate(self, prompts_list: List, max_n_tokens: int, temperature: float):
        """
        Generates responses for a batch of prompts using a language model.
        """
        raise NotImplementedError


# class HuggingFace(LanguageModel):
#     def __init__(self,model_name, model, tokenizer):
#         self.model_name = model_name
#         self.model = model
#         self.tokenizer = tokenizer
#         self.eos_token_ids = [self.tokenizer.eos_token_id]
#
#     def batched_generate(self,
#                         full_prompts_list,
#                         max_n_tokens: int,
#                         temperature: float,
#                         top_p: float = 1.0,):
#         inputs = self.tokenizer(full_prompts_list, return_tensors='pt', padding=True)
#         inputs = {k: v.to(self.model.device.index) for k, v in inputs.items()}
#
#         # Batch generation
#         if temperature > 0:
#             output_ids = self.model.generate(
#                 **inputs,
#                 max_new_tokens=max_n_tokens,
#                 do_sample=True,
#                 temperature=temperature,
#                 eos_token_id=self.eos_token_ids,
#                 top_p=top_p,
#             )
#         else:
#             output_ids = self.model.generate(
#                 **inputs,
#                 max_new_tokens=max_n_tokens,
#                 do_sample=False,
#                 eos_token_id=self.eos_token_ids,
#                 top_p=1,
#                 temperature=1, # To prevent warning messages
#             )
#
#         # If the model is not an encoder-decoder type, slice off the input tokens
#         if not self.model.config.is_encoder_decoder:
#             output_ids = output_ids[:, inputs["input_ids"].shape[1]:]
#
#         # Batch decoding
#         outputs_list = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
#
#         for key in inputs:
#             inputs[key].to('cpu')
#         output_ids.to('cpu')
#         del inputs, output_ids
#         gc.collect()
#         torch.cuda.empty_cache()
#
#         return outputs_list
#
#     def extend_eos_tokens(self):
#         # Add closing braces for Vicuna/Llama eos when using attacker model
#         self.eos_token_ids.extend([
#             self.tokenizer.encode("}")[1],
#             29913,
#             9092,
#             16675])

class GPT(LanguageModel):
    API_RETRY_SLEEP = 10
    API_ERROR_OUTPUT = "$ERROR$"
    API_QUERY_SLEEP = 0.5
    API_MAX_RETRY = 5
    API_TIMEOUT = 100
    openai.api_key = OPENAI_API_KEY

    def generate(self, conv: List[Dict],
                 max_n_tokens: int,
                 temperature: float,
                 top_p: float):
        """
        Args:
            conv: List of dictionaries, OpenAI API format
            max_n_tokens: int, max number of tokens to generate
            temperature: float, temperature for sampling
            top_p: float, top p for sampling
        Returns:
            str: generated response
        """
        output = self.API_ERROR_OUTPUT
        for _ in range(self.API_MAX_RETRY):
            try:
                client = OpenAI(api_key=openai.api_key)
                start_time = time.time()
                response = client.chat.completions.create(
                    model=self.model_name,
                    messages=conv,
                    max_tokens=max_n_tokens,
                    temperature=temperature,
                    top_p=top_p,
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
        return json.dumps(response.to_dict()), sum_time

    def batched_generate(self,
                         conv_list: List[List[Dict]],
                         max_n_tokens: int,
                         temperature: float,
                         top_p: float = 1.0, ):
        texts = []
        times = []
        for conv in conv_list:
            text, sum_time = self.generate(conv, max_n_tokens, temperature, top_p)
            texts.append(text)
            times.append(float(sum_time))
        return texts, times


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
