from tensorboardX import SummaryWriter

class TBX_Logger(object):
    def __init__(self, log_dir):
        self.logger = SummaryWriter(log_dir)
        self.log_dict = {}

    def add_summary(self, iteration):
        for k,v in self.log_dict.items():
            self.logger.add_scalar(k, v, iteration)
        self.log_dict = {}