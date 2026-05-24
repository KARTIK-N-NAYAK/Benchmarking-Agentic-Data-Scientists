from __future__ import annotations

import os
import pickle
import time
from datetime import timedelta
from enum import Enum
from pathlib import Path
from typing import T, Callable

import numpy as np
import pandas as pd

NO_HUMAN_ERROR_STR = "Queried human input in NO_HUMAN mode!"
RUNNING_CMDS_DIRNAME = "running_commands"


class UnauthorizedHumanHelpError(Exception):
    """When the agent needs human helps but is not authorized to bother a human"""

    pass


class SubmissionFormatError(Exception):
    """When all attempts to create the submission format have failed"""

    pass


def get_path_to_python(path_to_python: str) -> str:
    while not os.path.exists(path_to_python):
        os.makedirs(os.path.dirname(path_to_python), exist_ok=True)
        python_exe = input(
            rf"/!\ file {os.path.abspath(path_to_python)} should contain the absolute path to the python"
            f" executable to use, but {path_to_python} does not exists...\n"
            f"Enter here the path to the python executable you want to use:\n"
        )
        os.system(f'echo "{python_exe}" > {path_to_python}')
    with open(path_to_python) as f:
        python_exe = f.readline().replace("\n", "")
    return python_exe


class ListableEnum(Enum):

    def _generate_next_value_(name, start: ..., count: int, last_values: ...) -> str:
        """ Generate the next value when not given. """
        return name

    @classmethod
    def list(cls: T) -> list[T]:
        return list(map(lambda c: c.value, cls))

    @classmethod
    def _rev_dict(cls) -> dict[str, ...]:
        if not hasattr(cls, "_lookup"):
            # Create the reverse lookup only once per class
            cls._lookup = {member.value: member for member in cls}
        return cls._lookup

    @classmethod
    def get_enum_element(cls: T, value: str) -> T:
        if value not in cls._rev_dict():
            msg = f'Element with value "{value}" not in enum class {cls}.\nOnly has:\n\t- '
            msg += "\n\t- ".join(cls._rev_dict())
            raise ValueError(msg)
        return cls._rev_dict()[value]


def time_formatter(t: float, show_ms: bool = False) -> str:
    """Convert a duration in seconds to a str `dd:hh:mm:ss`

    Args:
        t: time in seconds
        show_ms: whether to show ms on top of dd:hh:mm:ss
    """
    n_day = time.gmtime(t).tm_yday - 1
    if n_day > 0:
        ts = time.strftime("%H:%M:%S", time.gmtime(t))
        ts = f"{n_day}:{ts}"
    else:
        ts = time.strftime("%H:%M:%S", time.gmtime(t))
    if show_ms:
        ts += f"{t - int(t):.3f}".replace("0.", ".")
    return ts


def get_df_stats(
        df: pd.DataFrame, columns_subset: list[str] | None = None,
        sorted_by: str | None = None
) -> pd.DataFrame:
    """
    Get a summary of each column values containing min, q1, q2, q3, max, mean, std

    Args:
        df: Dataframe containing column values to summarize.
        columns_subset: Subset of columns to summarize.
        sorted_by: criterion to sort the columns to show
    """
    if columns_subset is None:
        columns_subset = df.columns
    df = df[columns_subset]

    if sorted_by == "std":
        df = df[df.columns[np.argsort(df.std().values)[::-1]]]
    else:
        assert sorted_by is None

    summary = df.describe(percentiles=[0.25, 0.5, 0.75])

    # Get the mean and standard deviation separately
    mean_std = df.agg(['mean', 'std'])

    # Combine the summary and the mean/std data
    combined_stats = pd.concat([summary.loc[['min', '25%', '50%', '75%', 'max']], mean_std])

    # Rename the percentiles for better understanding
    combined_stats.rename(index={'25%': 'q1', '50%': 'q2', '75%': 'q3'}, inplace=True)
    combined_stats.loc['n_nans'] = df.isna().sum(0).astype(int)

    return combined_stats


def set_pd_options(
        max_rows: int | None, max_columns: int | None, float_format: Callable[[float], str],
        width: int | None = None
) -> None:
    """
    Set some options of pandas display
    """
    pd.set_option("display.max_rows", max_rows)
    pd.set_option("display.max_columns", max_columns)
    pd.set_option("display.width", width)
    pd.set_option("display.float_format", float_format)
    pd.set_option("display.max_colwidth", width)


def reset_pd_options() -> None:
    """
    Reset to default
    """
    pd.reset_option("display.max_rows")
    pd.reset_option("display.max_columns")
    pd.reset_option("display.width")
    pd.reset_option("display.float_format")
    pd.reset_option("display.max_colwidth")


class StringColors(Enum):
    RED = "\033[91m"
    GREEN = "\033[92m"
    BLUE = "\033[94m"
    YELLOW = "\033[93m"
    CYAN = "\033[96m"
    MAGENTA = "\033[95m"
    ORANGE = "\033[38;5;214m"
    PURPLE = "\033[38;5;141m"
    BROWN = "\033[38;5;94m"
    BLACK = "\033[30m"
    FOREST = "\033[38;5;70m"
    GREY = "\033[38;5;8m"
    WHITE = "\033[97m"
    PINK = "\033[38;5;218m"
    LIGHT_BLUE = "\033[38;5;117m"
    TEAL = "\033[38;5;37m"
    LIME = "\033[38;5;118m"
    GOLD = "\033[38;5;220m"
    SALMON = "\033[38;5;209m"
    VIOLET = "\033[38;5;177m"
    INDIGO = "\033[38;5;54m"
    MAROON = "\033[38;5;1m"
    NAVY = "\033[38;5;17m"
    SILVER = "\033[38;5;7m"
    BEIGE = "\033[38;5;230m"
    TURQUOISE = "\033[38;5;80m"

    def to_hex(self) -> str:
        """ Converts the ANSI escape sequence to a hex color. """
        # ANSI escape codes for 8-bit colors
        if '38;5' in self.value:
            color_code = int(self.value.split(';')[-1].replace('m', ''))
            return self._ansi_to_hex_8bit(color_code)
        elif self == StringColors.RED:
            return "#FF0000"
        elif self == StringColors.GREEN:
            return "#00FF00"
        elif self == StringColors.BLUE:
            return "#0000FF"
        elif self == StringColors.YELLOW:
            return "#FFFF00"
        elif self == StringColors.CYAN:
            return "#00FFFF"
        elif self == StringColors.MAGENTA:
            return "#FF00FF"
        elif self == StringColors.ORANGE:
            return "#FF7F00"
        elif self == StringColors.PURPLE:
            return "#800080"
        elif self == StringColors.BROWN:
            return "#A52A2A"
        elif self == StringColors.BLACK:
            return "#000000"
        elif self == StringColors.FOREST:
            return "#228B22"
        elif self == StringColors.GREY:
            return "#808080"
        raise ValueError(f"Color {self} not defined in mapping.")

    @staticmethod
    def _ansi_to_hex_8bit(color_code: int) -> str:
        """Convert ANSI 256 color code to RGB hex."""
        if 0 <= color_code <= 15:
            # Basic colors (0-15)
            basic_colors = [
                (0, 0, 0), (255, 0, 0), (0, 255, 0), (255, 255, 0),
                (0, 0, 255), (255, 0, 255), (0, 255, 255), (192, 192, 192),
                (128, 128, 128), (255, 128, 0), (0, 128, 0), (128, 0, 128),
                (0, 128, 128), (128, 128, 0), (255, 255, 255)
            ]
            r, g, b = basic_colors[color_code]
        elif 16 <= color_code <= 231:
            # 6x6x6 RGB cube (16-231)
            color_code -= 16
            r = ((color_code // 36) * 51)
            g = (((color_code // 6) % 6) * 51)
            b = ((color_code % 6) * 51)
        elif 232 <= color_code <= 255:
            # Grayscale (232-255)
            gray = (color_code - 232) * 10 + 8
            r = g = b = gray
        else:
            return "#000000"  # Default fallback
        return f"#{r:02X}{g:02X}{b:02X}"


def string_w_color(text: str, color: StringColors | None, bold: bool = False, min_length=0) -> str:
    bold_code = "\033[1m" if bold else ""
    reset_code = "\033[0m"

    if len(text) < min_length:
        text = text + " " * (min_length - len(text))
    if color is None:
        return text
    return f"{bold_code}{color.value}{text}{reset_code}"


def format_timedelta(delta: timedelta) -> str:
    days = delta.days
    hours = delta.seconds // 3600  # Divide by 3600 to get the number of hours
    return f"{days} days {hours} hours"


def save_w_pickle(obj: ..., path: str, filename: str | None = None, overwrite: bool = True) -> None:
    """Save object obj in file exp_path/filename.pkl
    Args:
        obj: object to save
        path: path to save to
        filename: name of the file (if not included in the path)
        overwrite: whether to overwrite existing file
    """
    if filename is None:
        filename = os.path.basename(path)
        path = os.path.dirname(path)
    if len(filename) < 4 or filename[-4:] != ".pkl":
        filename += ".pkl"
    if not os.path.exists(path):
        os.makedirs(path)
        os.chmod(path, 0o777)
    filepath = os.path.join(path, filename)
    if not os.path.exists(filepath) or overwrite:
        if os.path.exists(filepath):
            os.remove(filepath)
        with open(filepath, "wb") as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
        os.chmod(filepath, 0o777)


def load_w_pickle(path: str, filename: str | None = None) -> ...:
    """ Load object from file exp_path/filename.pkl """
    if filename is None:
        filename = os.path.basename(path)
        path = os.path.dirname(path)
    if len(filename) < 4 or filename[-4:] != '.pkl':
        filename += '.pkl'
    p = os.path.join(path, filename)
    with open(p, 'rb') as f:
        try:
            return pickle.load(f)
        except EOFError:
            raise Exception(f"EOFError with {p}")
        except UnicodeDecodeError:
            raise Exception(f"UnicodeDecodeError with {p}")
        except pickle.UnpicklingError:
            raise Exception(f"UnpicklingError with {p}")


def get_last_k_lines(file_path: str | Path, k: int) -> list[str]:
    """ Returns the last k lines of a file """
    lines = []

    with open(file_path, 'rb') as f:
        f.seek(-2, os.SEEK_END)  # Move to the second-to-last byte

        # Read backwards until we have enough lines
        while len(lines) < k:
            line = b""
            while f.read(1) != b'\n' and f.tell() > 0:  # Read one byte at a time backwards
                f.seek(-2, os.SEEK_CUR)  # Move back by 1 byte
                line = f.read(1) + line  # Prepend the byte to the line

            # Add the line to the list, after decoding it to string
            lines.append(line.decode())

            # Stop if we've reached the beginning of the file, and we don't have enough lines
            if f.tell() == 1:  # If we're at the very start, exit loop
                break

        lines.reverse()  # Reverse to return the lines in the correct order
    return lines
