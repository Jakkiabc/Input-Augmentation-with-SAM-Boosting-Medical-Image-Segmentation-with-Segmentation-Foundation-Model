# _*_ coding: utf-8 _*_
import logging
import os.path
import time


class Logger(object):
    def __init__(self, loggername):
        '''
            指定保存日志的文件路径，日志级别，以及调用文件
            将日志存入到指定的文件中
        '''

        # 创建一个logger
        self.logger = logging.getLogger(loggername)
        self.logger.setLevel(logging.DEBUG)

        # 创建一个handler，用于写入日志文件
        rq = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
        log_path = os.path.dirname(os.path.abspath('.')) + '/logs/'
        log_name = log_path + rq + '.log'
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')#输出格式

        fh = logging.FileHandler(log_name)
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)

        # 再创建一个handler，用于输出到控制台
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)

    def getlog(self):
        return self.logger
import logging
import time
import os

def log_creater(output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    log_name = '{}.log'.format(time.strftime('%Y-%m-%d-%H-%M'))
    final_log_file = os.path.join(output_dir,log_name)


    # creat a log
    log = logging.getLogger('train_log')
    log.setLevel(logging.DEBUG)

    # FileHandler
    file = logging.FileHandler(final_log_file)
    file.setLevel(logging.DEBUG)

    # StreamHandler
    stream = logging.StreamHandler()
    stream.setLevel(logging.DEBUG)

    # Formatter
    formatter = logging.Formatter(
        '[%(asctime)s][line: %(lineno)d] ==> %(message)s')

    # setFormatter
    file.setFormatter(formatter)
    stream.setFormatter(formatter)

    # addHandler
    log.addHandler(file)
    log.addHandler(stream)

    log.info('creating {}'.format(final_log_file))
    return log
class Logger(object):
    def __init__(self, log_file_name, log_level, logger_name):
        # firstly, create a logger
        self.__logger = logging.getLogger(logger_name)
        self.__logger.setLevel(log_level)
        # secondly, create a handler
        file_handler = logging.FileHandler(log_file_name)
        console_handler = logging.StreamHandler()
        # thirdly, define the output form of handler
        formatter = logging.Formatter(
            "[%(asctime)s]-[%(filename)s line:%(lineno)d]:%(message)s "
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        # finally, add the Hander to logger
        self.__logger.addHandler(file_handler)
        self.__logger.addHandler(console_handler)

    def get_log(self):
        return self.__logger