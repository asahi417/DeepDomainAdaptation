import os
import logging
from . import model


def get_model_instance(name: str=None):
    """ Get model instance
     Parameter
    ---------------
    name: str
        name of algorithm in ['dann', 'deep_jdot']

     Return
    --------------
    model_instance
    """

    model_list = dict(
        dann=model.DANN,
        source_only=model.SourceOnly
        # deep_jdot=model.deep_jdot.DeepJDOT
    )

    if name is None:
        return model_list
    else:
        return model_list[name]


def create_log(out_file_path: str=None):
    """ Logging
    If `out_file_path` is None, only show in terminal or else save log file in `out_file_path`. To avoid duplicate log,
    use one if exist.

     Parameter
    ------------------
    out_file_path: str
        path to output log file

     Usage
    -------------------
    >>> logger.info(message)
    >>> logger.error(error)
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

