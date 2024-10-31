import json
import logging
import os
from datetime import datetime
import structlog


class CustomJSONRenderer:
    def __call__(self, logger, name, event_dict):
        # 格式化为 JSON 字符串，每个 k:v 后面加换行
        json_output = json.dumps(event_dict, indent=4)
        return f"{json_output}\n\n\n\n{'=' * 120}\n\n\n\n"  # 添加分隔符


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

    def log(self, **kwargs):
        self.logger.info("attack log",
                         **kwargs)
