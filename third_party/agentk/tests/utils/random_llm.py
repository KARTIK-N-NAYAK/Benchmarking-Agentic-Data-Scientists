from typing import Callable

from agent.models.llm import LanguageBackend


class RandomLanguageBackend(LanguageBackend):
    def __init__(self, **kwargs):
        super().__init__("random", None, 0)

    def _chat_completion(self, messages: list[dict[str, str]], parse_func: Callable, **kwargs) -> tuple[str, float]:
        return "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do....", 0

    def count_tokens(self, prompt: list[dict[str, str]]) -> int:
        return len(prompt)
