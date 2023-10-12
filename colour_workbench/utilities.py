import logging
import sys

BASE_LOGGER_NAME = "colour_workbench"

__all__ = ["get_logger"]


def get_logger(name: str = ""):
    """Creates / returns loggers for the colour_workbench module

    Parameters
    ----------
    name : str, default ""
        Names a sub-logger for level management. Default "" returns base logger
        for colour_workbench

    Returns
    -------
    logging.Logger
    """
    if name == "":
        return logging.getLogger(f"{BASE_LOGGER_NAME}")
    return logging.getLogger(f"{BASE_LOGGER_NAME}.{name}")


BASE_LOGGER = get_logger()
BASE_LOGGER.setLevel("INFO")
BASE_LOGGER.addHandler(logging.StreamHandler(sys.stdout))
