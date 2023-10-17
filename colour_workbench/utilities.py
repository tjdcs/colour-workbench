import logging
import re
import sys

BASE_LOGGER_NAME = "colour_workbench"

__all__ = ["get_logger", "get_valid_filename"]


class SuspiciousFileOperation(Exception):
    """Generated when a user does something suspicious with file names"""

    pass


def get_valid_filename(name: str) -> str:
    """Clean / validate filename string

    Parameters
    ----------
    name : str
        The string to be cleaned for file name validity

    Returns
    -------
    str
        A clean filename

    Raises
    ------
    SuspiciousFileOperation
        if the cleaned string looks like a spooky filepath (i.e. '/', '.', etc...)
    """
    s = str(name).strip().replace(" ", "_")
    s = re.sub(r"(?u)[^-\w.]", "", s)
    if s in {"", ".", ".."}:
        raise SuspiciousFileOperation("Could not derive file name from '%s'" % name)
    return s


def get_logger(name: str = "") -> logging.Logger:
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
