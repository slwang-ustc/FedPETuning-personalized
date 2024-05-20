""" pretty logging for FedETuning """

import sys
from loguru import logger
from utils.register import registry
import os


def formatter(record):
    # default format
    time_format = "<green>{time:MM-DD/HH:mm:ss}</>"
    lvl_format = "<lvl><i>{level:^5}</></>"
    rcd_format = "<cyan>{file}:{line:}</>"
    msg_format = "<lvl>{message}</>"

    if record["level"].name in ["WARNING", "CRITICAL"]:
        lvl_format = "<l>" + lvl_format + "</>"

    return "|".join([time_format, lvl_format, rcd_format, msg_format]) + "\n"


def setup_logger():

    # RESULT_PATH = os.path.join(os.getcwd(), 'output/server_client_log/')
    # if not os.path.exists(RESULT_PATH):
    #     os.makedirs(RESULT_PATH, exist_ok=True)
    # # init logger
    # logger = logging.getLogger(os.path.basename(__file__).split('.')[0])
    # logger.setLevel(logging.INFO)
    # now = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(time.time()))
    # filename = RESULT_PATH + now + "_" + os.path.basename(__file__).split('.')[0] + '.log'
    # fileHandler = logging.FileHandler(filename=filename)
    # formatter = logging.Formatter("%(message)s")
    # fileHandler.setFormatter(formatter)
    # logger.addHandler(fileHandler)


    logger.remove()

    logger.add(
        sys.stderr, format=formatter,
        colorize=True, enqueue=True
    )

    registry.register("logger", logger)

    return logger
