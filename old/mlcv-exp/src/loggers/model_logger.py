import logging
import sys

class Model_Logger(object):
    def __init__(self):
        self.logger     = self.setup_logger()
        self.log_dict   = {}

    def setup_logger(self):
        logger = logging.getLogger("base")
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "%(asctime)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    def log(self, log_str):
            self.logger.info(log_str)

    def log_step(self, cur_step):
        log_str = "["
        for key, value in self.log_dict.items():
            log_str += f"| {key}: {value} |"
        log_str += "]"
        self.logger.info(log_str)
