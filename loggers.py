import json
import logging
import os
from datetime import datetime
import structlog


class CustomJSONRenderer:
    def __call__(self, logger, name, event_dict):
        # 格式化为 JSON 字符串，每个 k:v 后面加换行
        json_output = json.dumps(event_dict, indent=4)
        return f"{json_output}\n\n\n\n{'-' * 120}\n\n\n\n"  # 添加分隔符


class AttackLogger:
    def __init__(self):
        # 创建日志文件名
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_filename = f'./log/{timestamp}.log'

        # 确保日志目录存在
        os.makedirs('./log', exist_ok=True)

        # 创建日志处理器
        file_handler = logging.FileHandler(log_filename)
        file_handler.setFormatter(logging.Formatter("%(message)s"))

        # 配置标准库的 logging
        logging.basicConfig(level=logging.INFO, handlers=[file_handler])

        # 配置 structlog
        structlog.configure(
            processors=[
                CustomJSONRenderer(),  # 使用自定义 JSON 渲染器
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
        )

        # 创建一个 logger
        self.logger = structlog.get_logger()

    def log(self, iteration: int, **kwargs):
        self.logger.info("attack log",
                         Desciption=f"iteration:{iteration}",
                         **kwargs)

    def subproblems_optimize_initial(self, args):
        '''
        拆分子问题优化法 初始信息日志记录
        :param args:
        :return:
        '''
        self.logger.info("init",
                         Desciption="Model Args",
                         attack_model=args.attack_model,
                         target_model=args.target_model,
                         model_function_descript=args.function_descript,
                         n_iter=args.n_iterations,
                         n_subtesk=args.n_question,
                         )

    def integrate_log(self, integrate, n_question):
        '''
        拆分子问题优化法 总任务目标记录
        :param integrate:
        :param n_question:
        :return:
        '''
        subtask_questions = {}
        for i, question in enumerate(integrate["subtask_question"], start=1):
            subtask_questions[f"subtask_question{i}"] = question

        self.logger.info(event="integrate_system_prompt",
                         Desciption="The problem framework is built, a broad topic is generated, and sub-questions "
                                    "are generated according to the topic.",
                         n_question=n_question,
                         total_task_prompt=integrate["total_prompt"],
                         **subtask_questions)

    def subtask_log(self, n_iterations, subtask_prompt_list):
        '''
        拆分子问题优化法 子问题生成日志记录
        :param n_iterations:
        :param subtask_prompt_list:
        :return:
        '''
        subtask_prompt = {}
        for i, question in enumerate(subtask_prompt_list, start=1):
            subtask_prompt[f"subtask_prompt{i}"] = question
        self.logger.info(event="subtask_log",
                         Desciption="A subproblem that causes the target model to extend the output",
                         n_iterations=n_iterations,
                         **subtask_prompt
                         )

    def background_log(self, subtask_prompt_list, general_assignment, length_lest):
        '''
        拆分子问题优化法 子问题添加背景延长及 总问题prompt 日志记录
        :param subtask_prompt_list:
        :param general_assignment:
        :param length_lest:
        :return:
        '''
        subtask_prompt = {}
        for i, question in enumerate(subtask_prompt_list, start=1):
            subtask_prompt[f"background_prompt{i}"] = question
        self.logger.info(event="background_log",
                         Desciption="Write an extended backstory to the problem",
                         **subtask_prompt,
                         ooutput_length=length_lest,
                         general_assignment=general_assignment,
                         general_assignment_length=len(general_assignment)
                         )
