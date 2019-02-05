import os
import logging
from . import model

MODEL_LIST = dict(
    dann=model.DANN,
    deep_jdot=model.DeepJDOT
)


def create_log(out_file_path: str=None):
    """ Logging
    If `out_file_path` is None, only show in terminal or else save log file in `out_file_path`. To avoid duplicate log,
    use one if exist.

     Parameter
    ------------------
    out_file_path: path to output log file

     Usage
    -------------------
    logger.info(message)
    logger.error(error)
    """

    # handler to record log to a log file
    if out_file_path is not None:
        if os.path.exists(out_file_path):
            os.remove(out_file_path)
        logger = logging.getLogger(out_file_path)

        if len(logger.handlers) > 1:  # if there are already handler, return it
            return logger
        else:
            logger.setLevel(logging.DEBUG)
            formatter = logging.Formatter("H1, %(asctime)s %(levelname)8s %(message)s")

            handler = logging.FileHandler(out_file_path)
            handler.setFormatter(formatter)
            logger.addHandler(handler)

            logger_stream = logging.getLogger()
            # check if some stream handlers are already
            if len(logger_stream.handlers) > 0:
                return logger
            else:
                handler = logging.StreamHandler()
                handler.setFormatter(formatter)
                logger.addHandler(handler)

                return logger
    else:
        # handler to output
        handler = logging.StreamHandler()
        logger = logging.getLogger()

        if len(logger.handlers) > 0:  # if there are already handler, return it
            return logger
        else:  # in case of no, make new output handler
            logger.setLevel(logging.DEBUG)
            formatter = logging.Formatter("H1, %(asctime)s %(levelname)8s %(message)s")
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            return logger


def raise_error(condition: bool, msg: str):
    """ function to raise ValueError error

     Parameter
    -------------
    condition: if condition is True, raise ValueError
    msg: manual error message
    """
    if condition:
        raise ValueError(msg)


def mkdir(path: str):
    """ mkdir command

     Parameter
    -------------
    path: path to make directory
    """
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
