import logging
import logging.handlers as handlers
import time


class LogManager(object):
    # 基礎設定
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
        datefmt='%m-%d %H:%M',
        handlers=[
            logging.FileHandler(
                'log/{}.log'.format(
                    time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())), 'w',
                'utf-8'),
        ])

    @staticmethod
    def error(msg):
        logging.getLogger().error(msg)

    @staticmethod
    def info(msg):
        logging.getLogger().info(msg)