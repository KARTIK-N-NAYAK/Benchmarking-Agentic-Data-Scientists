from abc import ABC
from abc import abstractmethod
from enum import Enum
from typing import Any

from agent.loggers.base import Logger
from agent.memory import MemKey


class DatasetOutOfBoundsException(Exception):
    """Raised when dataset index is greater than dataset length."""
    pass


class ActionSpace(Enum):
    DISCRETE = 1
    CONTINUOUS = 2


class AnswerType(Enum):
    TEXT = 1
    CODE = 2
    DIGIT = 3


class Task(ABC):
    def __init__(
            self,
            name: str,
            subtask: str | None,
            version: str,
            description: str,
            logger: Logger,
            **args,
    ):
        self.name = name
        self.subtask = subtask
        self.version = version
        self.description = description
        self.logger = logger
        self.answer_type = AnswerType.TEXT
        self.id = None
        self.action_space = NotImplemented

    @abstractmethod
    def step(self, action) -> tuple[dict[MemKey, Any], float, bool]:
        """Perform an action and return the next observation, reward, and done."""
        pass

    @abstractmethod
    def reset(self, next_subtask: str | None) -> dict[MemKey, Any]:
        """Reset the environment and return the initial observation."""
        pass

    @abstractmethod
    def answer_parser(self, raw_response: str) -> str:
        """Return a parsed response."""
        pass
