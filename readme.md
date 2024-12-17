# Crabs: Consuming Resrouce via Auto-generation for LLM-DoS Attack under Black-box Settings

## Abstract

Large Language Models (LLMs) have demonstrated remarkable performance across diverse tasks. LLMs continue to be vulnerable to external threats, particularly Denial-of-Service (DoS) attacks. Specifically, LLM-DoS attacks aim to exhaust computational resources and block services. However, prior works tend to focus on performing white-box attacks, overlooking black-box settings. In this work, we propose an automated algorithm designed for black-box LLMs, called Auto-Generation for LLM-DoS Attack (**AutoDoS**). AutoDoS introduces **DoS Attack Tree** and optimizes the prompt node coverage to enhance effectiveness under black-box conditions. Our method can bypass existing defense with enhanced stealthiness via semantic improvement of prompt nodes. Furthermore, we reveal that implanting **Length Trojan** in Basic DoS Prompt aids in achieving higher attack efficacy. Experimental results show that AutoDoS amplifies service response latency by over **250** $\times \uparrow$, leading to severe resource consumption in terms of GPU utilization and memory usage. 

## Install

Clone this project

```
git clone https://github.com/shuita2333/Agent_delay_attack.git
cd Agent_delay_attack
```

Create virtual environment

```
conda create -n AutoDoS python=3.9 -y
conda activate AutoDoS
```

Install requirements

```
pip install -r requirements.txt
```

Create a new `API_kye.py` in the root directory

```
cd Agent_delay_attack
touch API_key.py
```

Please enter the key of the model to be tested into the `API_key.py`

```
#Calling GPT from "openAI"
OPENAI_API_KEY=""

#Calling Qwen and Llama from "https://www.aliyun.com/"
# ALIYUN_API_KEY = ""
ALIYUN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

#Calling Ministral from platform "https://mistral.ai/"
Mistral_API = ""
Mistral_BASE_URL = "https://api.mistral.ai/v1/chat/completions"

#Calling DeepSeek from "https://api.deepseek.com"
DEEPSEEK_API_KEY = ""
DEEPSEEK_BASE_URL = "https://api.deepseek.com/beta"

#Calling other models from platform "https://siliconflow.cn/"
Siliconflow_API_KEY="""
Siliconflow_BASE_URL= "https://api.siliconflow.cn/v1/chat/completions"
```

## Run Experiments

To run AutoDoS, run:

```
python3 professional_iterative_generation.py --attack-model [ATTACK MODEL] --target-model [TARGET MODEL] --judge-model [JUDGE MODEL] --goal [GOAL STRING] --target-str [TARGET STRING]
```







