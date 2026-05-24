from abc import ABC
from abc import abstractmethod
from typing import Dict


class Tool(ABC):
    requires_llm_prompt: bool  # whether the tool needs llm output as input when it is called.

    @abstractmethod
    def __call__(self, agent_input: str) -> dict[str, str]:
        pass
