import argparse
import ast
import json
import logging
import os
import time
from datetime import datetime
import structlog

from agents.AgentFactory import AgentFactory
from utils.attack import generate_general_prompt, iterative_optimization
from utils.loggers import CustomJSONRenderer


class TargetLogger:
    def __init__(self):
        # 创建日志文件名
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_filename = f'./test_data/result/effect_test___{timestamp}.log'

        # 确保日志目录存在
        os.makedirs('../test_data/result', exist_ok=True)

        # 创建一个特定的 logger
        logger_name = f'target_logger_{timestamp}'
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(logging.INFO)

        # 创建日志处理器
        file_handler = logging.FileHandler(log_filename)
        file_handler.setFormatter(logging.Formatter("%(message)s"))

        # 将处理器添加到 logger
        self.logger.addHandler(file_handler)

        # 配置 structlog
        structlog.configure(
            processors=[
                CustomJSONRenderer(),  # 使用自定义 JSON 渲染器
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            cache_logger_on_first_use=True,
        )

        # 创建一个 structlog logger
        self.Target_logger = structlog.wrap_logger(self.logger)


    def result_log(self, **kwargs):
        self.Target_logger.info("attack log",
                         **kwargs)


Target_logger = TargetLogger()


def basic_test(args, data_config, task):
    """
        测试输入信息对计算时间的影响
        :param args:
        :param logger:
        :return:
        """
    if task == "import_time_test":
        args.target_max_n_tokens=1
        args.target_length=1
    target_agent = AgentFactory.get_factory('TargetAgent', args)
    with open(
            f'test_data/data/{data_config.attack1LM}_{data_config.attack2LM}_{data_config.num}_{data_config.method}.json',
            'r', encoding='utf-8') as file:
        content = file.read()
        review_agent_synthesize_list = ast.literal_eval(content)

    batch_size = len(review_agent_synthesize_list)
    target_agent_conv_list = target_agent.get_conv_list(batch_size)

    time_list = []

    target_information_list, target_time = target_agent.get_response(target_agent_conv_list,
                                                                     review_agent_synthesize_list)
    target_response_list = [i['content_str'] for i in target_information_list]
    target_response_length = [i['content_length'] for i in target_information_list]
    # for conv, review_agent_synthesize in zip(target_agent_conv_list, review_agent_synthesize_list):
    #     review_agent_synthesize = review_agent_synthesize
    #     start_time = time.time()
    #     target_information_list = target_agent.get_response([conv], [review_agent_synthesize])
    #     end_time = time.time()
    #     sum_time = end_time - start_time
    #     time_list.append(sum_time)
    #     target_response_list.append(target_information_list[0]['content_str'])
    #     target_response_length.append(target_information_list[0]['content_length'])

    ave_time = sum(target_time) / batch_size

    Target_logger.result_log(
        task="result test",
        attack_model=data_config.attack1LM,
        attack_iteration_model=data_config.attack2LM,
        target_model=data_config.targetLM,
        question_list=review_agent_synthesize_list,
        target_response_list=target_response_list,
        target_response_length=target_response_length,
        time=time_list,
        ave_time=ave_time
    )


def test_prompt_generate(args, parameter):
    """
    批量生成攻击语句
    :param args:
    :param logger:
    :return:
    """
    data_list = []

    for i in range(1, parameter["target_quantity"] + 1):
        general_prompt,subtask_answer_list = generate_general_prompt(args)
        target_success_agent = iterative_optimization(args, general_prompt,subtask_answer_list)

        # 将结果追加到列表中
        data_list.append(target_success_agent)
        print(f"+++++++++++++++  ({i}/{parameter['target_quantity']}) 第{i}次生成已完成  +++++++++++++++")

    # 将列表写入JSON文件
    with open(f'test_data/data/{args.attack_model}_{args.target_model}_{parameter["target_quantity"]}_subtask.json',
              'w', encoding='utf-8') as json_file:
        json.dump(data_list, json_file, indent=4, ensure_ascii=False)


class DataConfiguration:
    def __init__(self, attack1LM="Qwen2.5-7B", attack2LM="Qwen2.5-7B", targetLM="Qwen2.5-7B", num=10, method=""):
        self.attack1LM = attack1LM  # 攻击模型
        self.attack2LM = attack2LM  # 迭代目标模型
        self.targetLM = targetLM  # 目标模型
        self.num = num  # 语句数量
        self.method = method  # 使用方法    ("subtask"/"Repetition"/"Recursion"/"Count"/"Long Article"/"Source Code")



if __name__ == '__main__':
    # 模型选择
    # "Meta-Llama-3.1", "DeepSeek-V2.5", "Qwen2.5-7B", "vicuna", "llama-2", "gpt-3.5-turbo", "gpt-4",
    # "claude-instant-1", "claude-2", "palm-2","gemma-2-9b"
    data_config = DataConfiguration("Qwen2.5-7B", "Qwen2.5-7B", "Qwen2.5-7B", 10, "subtask")

    #   task 类别
    #   "effect_test"       测试目标模型响应效果
    #   "prompt_generate"   生成批次测试样本
    #   "import_time_test"  输入信息对计算时间的影响
    task = "prompt_generate"


    parser = argparse.ArgumentParser()

    attack_model = data_config.attack1LM
    if task == "effect_test" or task == "import_time_test":
        target_model = data_config.targetLM
    elif task == "prompt_generate":
        target_model = data_config.attack2LM

    test_prompt_generate_parameter = {
        "target_quantity": data_config.num,
    }
    use_arg = True
    if use_arg == True:
        parser.add_argument(
            "--function-descript",
            type=str,
            default="Financial functional services system provides personalized investment advice and risk management "
                    "strategies by analyzing users' financial data and behavior patterns.",
            help="对目标agent的功能描述."
        )

        parser.add_argument(
            "--n-question",
            type=int,
            default=15,
            help="子问题数 Number of concurrent jailbreak conversations. If this is too large, then there may be out of "
                 "memory errors."
        )

        # ================ Attack model parameters ================
        parser.add_argument(
            "--attack-model",
            default=attack_model,
            help="Name of attacking model.",
            choices=["Meta-Llama-3.1", "DeepSeek-V2.5", "Qwen2.5-7B", "vicuna", "llama-2", "gpt-3.5-turbo", "gpt-4",
                     "claude-instant-1",
                     "claude-2", "palm-2"]
        )
        parser.add_argument(
            "--attack-max-n-tokens",
            type=int,
            default=4096,
            help="Maximum number of generated tokens for the attacker."
        )
        parser.add_argument(
            "--max-n-attack-attempts",
            type=int,
            default=8,
            help="Maximum number of attack generation attempts, in case of generation errors."
        )
        ##################################################

        # =========== Target model parameters ===========
        parser.add_argument(
            "--target-model",
            default=target_model,
            help="Name of target model.",
            choices=["Meta-Llama-3.1", "DeepSeek-V2.5", "Qwen2.5-7B", "vicuna", "llama-2", "gpt-3.5-turbo", "gpt-4",
                     "claude-instant-1",
                     "claude-2", "palm-2"]
        )
        parser.add_argument(
            "--target-max-n-tokens",
            type=int,
            default=4096,
            help="Maximum number of generated tokens for the target."
        )
        parser.add_argument(
            "--target-length",
            type=int,
            default=4096,
            help="目标输出长度."
        )
        ##################################################

        # ============== Judge model parameters =============
        parser.add_argument(
            "--judge-model",
            default="Qwen2.5-7B",
            help="Name of judge model.",
            choices=["gpt-3.5-turbo", "gpt-4", "no-judge"]
        )
        parser.add_argument(
            "--judge-max-n-tokens",
            type=int,
            default=10,
            help="Maximum number of tokens for the judge."
        )
        parser.add_argument(
            "--judge-temperature",
            type=float,
            default=0,
            help="Temperature to use for judge."
        )
        ##################################################

        # ============== PAIR parameters =============
        parser.add_argument(
            "--n-streams",
            type=int,
            default=2,
            help="并发数 Number of concurrent jailbreak conversations. If this is too large, then there may be out of memory "
                 "errors."
        )

        parser.add_argument(
            "--keep-last-n",
            type=int,
            default=5,
            help="Number of responses to save in conversation history of attack model. If this is too large, then it may "
                 "exceed the context window of the model."
        )
        parser.add_argument(
            "--n-iterations",
            type=int,
            default=20,
            help="Number of iterations to run the attack."
        )
        parser.add_argument(
            "--goal",
            type=str,
            default="Financial functional services system provides personalized investment advice and risk management "
                    "strategies by analyzing users' financial data and behavior patterns.",
            # help="Goal of the conversation."
            help="对目标agent的功能描述."
        )
        # parser.add_argument(
        #     "--target-str",
        #     type=str,
        #     default="Sure, here is a tutorial for making a bomb",
        #     help="Target response for the target model."
        # )

        ##################################################

        # ================= Logging parameters ==================
        parser.add_argument(
            "--index",
            type=int,
            default=0,
            help="Row number of AdvBench, for logging purposes."
        )
        parser.add_argument(
            "--category",
            type=str,
            default="bomb",
            help="Category of jailbreak, for logging purposes."
        )
        ##################################################

    print(f"执行任务为 {task}   攻击模型 {data_config.attack1LM}   模拟目标 {data_config.attack2LM}   目标模型 {data_config.targetLM}   \n"
          f"子问题数 {parser.parse_args().n_question}   总条目数 {data_config.num}   目标输出长度 {parser.parse_args().target_max_n_tokens}")

    if task == "effect_test" or task == "import_time_test":
        basic_test(parser.parse_args(), data_config, task)
    elif task == "prompt_generate":
        test_prompt_generate(parser.parse_args(), test_prompt_generate_parameter)
