import enum
import json
import time
from hashlib import sha256
from pathlib import Path
from typing import TypedDict

import rich
import rich.tree
from omegaconf import DictConfig
from omegaconf import OmegaConf
from rich.markup import escape

from agent.loggers.utils import JSONEncoderV2

# Cost per token
_BASE_INPUT_LLM_COST = 0.03 / 1000
_BASE_OUTPUT_LLM_COST = 0.06 / 1000

_BASE_LLM_PARAMS = int(1.7 * 1e12)


class Message(TypedDict):
    role: str
    content: str


class Tag(str, enum.Enum):
    EPISODE = "episode"
    TIMESTEP = "timestep"
    TIMESTAMP = "timestamp"
    DONE = "done"
    REWARD = "reward"
    ERROR = "error"
    LLM = "llm"

    @staticmethod
    def universal_tags(episode: int = None, timestep: int = None, timestamp: float = None):
        return {
            Tag.EPISODE: episode,
            Tag.TIMESTEP: timestep,
            Tag.TIMESTAMP: timestamp if timestamp is not None else time.time(),
        }


class Logger:
    def __init__(self, project_cfg: DictConfig):
        self._logs = []

        self.config_hash = sha256(
            json.dumps(
                {
                    k: v
                    for k, v in OmegaConf.to_container(project_cfg).items()
                    if k not in ["seed", "extras", "paths", "tags"]
                },
                sort_keys=True,
            ).encode("utf8")
        ).hexdigest()[-10:]

        self._start_time = time.time()
        self._prev_time = None
        self._cur_timestep = 0
        self._message_timestep = 0
        self._cur_episode = 0

    def update_state(self, *args, **kwargs):
        pass

    def log_metrics(
            self,
            data: dict[str, str | float | int | list[Message]],
            episode: int | None = None,
            timestep: int | None = None,
    ):
        assert episode is None or episode >= self._cur_episode
        assert timestep is None or (timestep >= self._cur_timestep or episode > self._cur_episode)

        self._cur_timestep = self._cur_timestep if timestep is None else timestep
        self._cur_episode = self._cur_episode if episode is None else episode

        universal_tags = Tag.universal_tags(self._cur_episode, self._cur_timestep)

        for key, value in data.items():
            self._logs.append({**universal_tags, key: value})
            if key == "llm:input":
                self._message_timestep += 1

    def save_metrics(self):
        self._logs = []
        pass


class ManyLoggers(Logger):
    def __init__(self, loggers: list[Logger]):
        self.loggers = loggers
        self.fs_loggers: list[FileSystemLogger] = list(filter(lambda l: isinstance(l, FileSystemLogger), loggers))

    def update_state(self, *args, **kwargs):
        for logger in self.loggers:
            logger.update_state(*args, **kwargs)

    def log_metrics(self, *args, **kwargs):
        for logger in self.loggers:
            logger.log_metrics(*args, **kwargs)

    def save_metrics(self):
        for logger in self.loggers:
            logger.save_metrics()


class StdoutLogger(Logger):
    def save_metrics(self) -> None:
        tree = rich.tree.Tree("Logs")
        for log in self._logs:
            keys = set(log.keys()) - set(Tag.universal_tags().keys())
            for key in keys:
                tree.add(key).add(escape(str(log[key])))

        rich.print(tree)
        print(f"Number of llm inputs so far: {self._message_timestep}")
        self._logs = []


class FileSystemLogger(Logger):
    def __init__(self, project_cfg: DictConfig, save_to_file: str, **kwargs):
        super().__init__(project_cfg=project_cfg)
        self.output_dir = project_cfg.paths.output_dir
        self.save_to_file = save_to_file

    @property
    def log_path(self):
        return Path(self.output_dir) / self.save_to_file

    def save_metrics(self):
        with self.log_path.open("a") as outfile:
            for record in self._logs:
                json.dump(obj=record, fp=outfile, cls=JSONEncoderV2)
                outfile.write("\n")
        self._logs = []


class APIUsageLogger(Logger):
    def __init__(self, project_cfg: DictConfig):
        super().__init__(project_cfg=project_cfg)
        self.input_usage = 0
        self.output_usage = 0
        self.llm_num_params = None

    def update_state(self, llm_num_params: int):
        self.llm_num_params = llm_num_params
        print(f"llm_num_params: {self.llm_num_params}")

    def log_metrics(
            self,
            data: dict[str, str | float | int | list[Message]],
            episode: int | None = None,
            timestep: int | None = None,
    ):
        for key, value in data.items():
            if key.startswith("api_usage"):
                self._logs.append({key: value})
            if value is not None:
                if key.startswith("api_usage:input"):
                    self.input_usage += value
                elif key.startswith("api_usage:output"):
                    self.output_usage += value

    def save_metrics(self):
        print(self._logs)
        self._logs = []

    def reset_usage_metrics(self):
        self.input_usage = 0
        self.output_usage = 0

    def get_cost(self):
        return ((self.input_usage * _BASE_INPUT_LLM_COST) + (self.output_usage * _BASE_OUTPUT_LLM_COST)) * (
                self.llm_num_params / _BASE_LLM_PARAMS
        )


class FakeLogger(Logger):
    def __init__(self):
        pass

    def log_metrics(
            self,
            data: dict[str, str | float | int | list[Message]],
            episode: int | None = None,
            timestep: int | None = None,
    ):
        pass

    def save_metrics(self):
        pass
