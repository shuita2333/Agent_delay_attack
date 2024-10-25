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
    def __init__(self, args, system_prompt, ):
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

        self.logger.info("init",
                         Desciption="Model Args",
                         attack_model=args.attack_model,
                         target_model=args.target_model,
                         judge_model=args.judge_model,
                         keep_last_n=args.keep_last_n,
                         system_prompt=system_prompt,
                         index=args.index,
                         category=args.category,
                         goal=args.goal,
                         n_iter=args.n_iterations,
                         n_streams=args.n_streams,
                         )

    def log(self, iteration: int, attack_list: list, response_list: list, judge_scores: list):
        self.logger.info("attack log",
                         Desciption=f"iteration:{iteration}",
                         attack_list=attack_list,
                         response_list=response_list,
                         judge_scores=judge_scores)
