import sys
import tensorflow as tf
import logging

class TBLogger(object):
    def __init__(self, log_dir):
        """Create a summary writer logging to log_dir."""
        self.writer = tf.summary.FileWriter(log_dir)
        self.log_dict = {}
        
    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
        self.writer.add_summary(summary, step)
        
    def update_scalar_summary(self, epoch):
        print("TENSORBOARD UPDATE SUMMARY EPOCH {}".format(epoch))
        for tag, value in self.log_dict.items():
            self.scalar_summary(tag, value, epoch) 
        self.log_dict = {}
                      
class Logger(object):
    def __init__(self, conf, rank):
        self.logger = self.get_logger()
        self.print_freq = conf["print_freq"]
        self.log_dict = {}
        self.rank = rank
        
    def get_logger(self):
        logger = logging.getLogger("base")
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    def log(self, log_str):
        if self.rank <=0:
            self.logger.info(log_str)        

    def log_step(self, cur_step):
        if self.rank <= 0:
            log_str = "["
            for key, value in self.log_dict.items():
                log_str += f"| {key}: {value} |" 
            log_str += "]"
            self.logger.info(log_str)
            