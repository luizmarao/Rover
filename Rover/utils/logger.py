import datetime
import os
import sys
import tempfile
import warnings
from typing import Any, Dict, List, Mapping, Optional, Sequence, TextIO, Tuple, Union
from stable_baselines3.common.logger import Logger, HumanOutputFormat, Video, FormatUnsupportedError, Figure, Image, \
    HParam, KVWriter, JSONOutputFormat, CSVOutputFormat, TensorBoardOutputFormat

import numpy as np
import pandas
import torch as th
from matplotlib import pyplot as plt

try:
    from torch.utils.tensorboard import SummaryWriter
    from torch.utils.tensorboard.summary import hparams
except ImportError:
    SummaryWriter = None  # type: ignore[misc, assignment]

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

DEBUG = 10
INFO = 20
WARN = 30
ERROR = 40
DISABLED = 50


def configure(folder: Optional[str] = None, format_strings: Optional[List[str]] = None, exp_name: Optional[str] = None) -> Logger:
    """
    Configure the current logger.

    :param folder: the save location
        (if None, $SB3_LOGDIR, if still None, tempdir/SB3-[date & time])
    :param format_strings: the output logging format
        (if None, $SB3_LOG_FORMAT, if still None, ['stdout', 'log', 'csv'])
    :return: The logger object.
    """
    if folder is None:
        folder = os.getenv("SB3_LOGDIR")
    if folder is None:
        folder = os.path.join(tempfile.gettempdir(), datetime.datetime.now().strftime("SB3-%Y-%m-%d-%H-%M-%S"))
    elif exp_name is not None:
        folder = os.path.join(folder, exp_name + datetime.datetime.now().strftime("-%Y-%m-%d-%H-%M-%S"))
    else:
        folder = os.path.join(folder, datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))  #TODO: what in case of loading and retrain?
    assert isinstance(folder, str)
    os.makedirs(folder, exist_ok=True)

    log_suffix = ""
    if format_strings is None:
        format_strings = os.getenv("SB3_LOG_FORMAT", "stdout,log,csv").split(",")

    format_strings = list(filter(None, format_strings))
    output_formats = [make_output_format(f, folder, log_suffix) for f in format_strings]

    logger = Logger(folder=folder, output_formats=output_formats)
    # Only print when some files will be saved
    if len(format_strings) > 0 and format_strings != ["stdout"]:
        logger.log(f"Logging to {folder}")
    return logger

def make_output_format(_format: str, log_dir: str, log_suffix: str = "") -> KVWriter:
    """
    return a logger for the requested format

    :param _format: the requested format to log to ('stdout', 'log', 'json' or 'csv' or 'tensorboard')
    :param log_dir: the logging directory
    :param log_suffix: the suffix for the log file
    :return: the logger
    """
    os.makedirs(log_dir, exist_ok=True)
    if _format == "stdout":
        return HumanOutputFormat4Rover(sys.stdout)
    elif _format == "log":
        return HumanOutputFormat4Rover(os.path.join(log_dir, f"log{log_suffix}.txt"))
    elif _format == "json":
        return JSONOutputFormat(os.path.join(log_dir, f"progress{log_suffix}.json"))
    elif _format == "csv":
        return CSVOutputFormat(os.path.join(log_dir, f"progress{log_suffix}.csv"))
    elif _format == "tensorboard":
        return TensorBoardOutputFormat(log_dir)
    else:
        raise ValueError(f"Unknown format specified: {_format}")


class HumanOutputFormat4Rover(HumanOutputFormat):
    def write(self, key_values: Dict, key_excluded: Dict, step: int = 0) -> None:
        # Create strings for printing
        key2str = {}
        tag = None
        for (key, value), (_, excluded) in zip(sorted(key_values.items()), sorted(key_excluded.items())):
            if excluded is not None and ("stdout" in excluded or "log" in excluded):
                continue

            elif isinstance(value, Video):
                raise FormatUnsupportedError(["stdout", "log"], "video")

            elif isinstance(value, Figure):
                raise FormatUnsupportedError(["stdout", "log"], "figure")

            elif isinstance(value, Image):
                raise FormatUnsupportedError(["stdout", "log"], "image")

            elif isinstance(value, HParam):
                raise FormatUnsupportedError(["stdout", "log"], "hparam")

            elif isinstance(value, float):
                # Align left
                value_str = f"{value:<8.3g}"
            else:
                value_str = str(value)

            if key.find("/") > 0:  # Find tag and add it to the dict
                tag = key[: key.find("/") + 1]
                key2str[(tag, self._truncate(tag))] = ""
            # Remove tag from key
            if tag is not None and tag in key:
                key = str("   " + key[len(tag) :])

            truncated_key = self._truncate(key)
            if (tag, truncated_key) in key2str:
                raise ValueError(
                    f"Key '{key}' truncated to '{truncated_key}' that already exists. Consider increasing `max_length`."
                )
            key2str[(tag, truncated_key)] = self._truncate(value_str)

        # Find max widths
        if len(key2str) == 0:
            warnings.warn("Tried to write empty key-value dict")
            return
        else:
            tagless_keys = map(lambda x: x[1], key2str.keys())
            key_width = max(map(len, tagless_keys))
            val_width = max(map(len, key2str.values()))

        # Write out the data
        dashes = "-" * (key_width + val_width + 7)
        lines = [dashes]
        for (_, key), value in key2str.items():
            key_space = " " * (key_width - len(key))
            val_space = " " * (val_width - len(value))
            lines.append(f"| {key}{key_space} | {value}{val_space} |")
        lines.append(dashes)

        if tqdm is not None and hasattr(self.file, "name") and self.file.name == "<stdout>":
            # Do not mess up with progress bar
            tqdm.write("\n".join(lines) + "\n", file=sys.stdout, end="")
        else:
            self.file.write("\n".join(lines) + "\n")

        # Flush the output to the file
        self.file.flush()
