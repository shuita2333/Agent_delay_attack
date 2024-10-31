from utils.language_models import GPT, Siliconflow
from fastchat.model import get_conversation_template


def load_indiv_model(model_name):
    """
    构造模型，定义模板名称
    :param model_name:
    :return:
    """
    model_path, template = get_model_path_and_template(model_name)
    if model_name in ["gpt-3.5-turbo", "gpt-4"]:
        lm = GPT(model_name)
    elif model_name in ["Qwen2.5-7B", "Qwen2.5-32B", "DeepSeek-V2.5"]:
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
    """
    定义模型路径
    :param model_name:
    :return:
    """
    full_model_dict = {
        "gpt-4": {
            "path": "gpt-4",
            "template": "gpt-4"
        },
        "Qwen2.5-7B": {
            "path": "Qwen/Qwen2.5-7B-Instruct",
            "template": "Qwen2"
        },
        "Qwen2.5-32B": {
            "path": "Qwen/Qwen2.5-32B-Instruct",
            "template": "Qwen2"
        },
        "DeepSeek-V2.5": {
            "path": "deepseek-ai/DeepSeek-V2.5",
            "template": "DeepSeek-llm-chat"
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


def conv_template(template_name):
    """
    获取对应的对话模板
    :param template_name:
    :return:
    """
    template = get_conversation_template(template_name)
    if template.name == 'llama-2':
        template.sep2 = template.sep2.strip()
    return template