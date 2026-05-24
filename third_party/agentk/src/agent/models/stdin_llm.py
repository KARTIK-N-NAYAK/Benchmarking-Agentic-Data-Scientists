import time
from typing import Any, Callable

from rich import print

from agent.models.llm import LanguageBackend


class StdinLanguageBackend(LanguageBackend):

    def __init__(self, model_id: str, logger: Any, context_length: int, responses_from_file: list[str] | None = None,
                 **kwargs):
        """
        Args:
            responses_from_file:  if set, the answers will be taken from the list rather than queried from the LLM
        """
        super().__init__(model_id=model_id, logger=logger, context_length=context_length)
        self.responses_from_file = responses_from_file

    def count_tokens(self, messages: list[dict[str, str]]) -> int:
        return 0

    def _chat_completion(self, messages: list[dict[str, str]], parse_func: Callable, **kwargs) -> tuple[str, float]:
        """Fakes an LLM generation by getting input from the keyboard.

        Args:
            messages (list[dict[str, str]]): The input text prompt to generate completion for.
            parse_func (Callable): A function to parse the model's response.
            **kwargs: Additional keyword arguments that may be required for the generation,
                      such as temperature, max_tokens, etc.

        Returns:
            str: The text completion.
            runtime: time taken to execute this function (won't take retrials into account)
        """
        start_time = time.time()
        if not self.human_mode:
            print(f"\n{'-' * 20}\nHere is a list of messages that would normally be fed to the LLM.\n{'-' * 20}")
            for m in messages:
                for k, v in m.items():
                    print(k, ":", v)

            reply = ""
            stop_signal = "[STOP]"
            output = input(f"Please enter a reply (write {stop_signal} at the end): ")
            n_empty_lines_in_a_raw = 0
            while stop_signal not in output[-len(stop_signal):]:
                reply += output + "\n"
                if output == "":
                    n_empty_lines_in_a_raw += 1
                else:
                    n_empty_lines_in_a_raw = 0
                if n_empty_lines_in_a_raw >= 5:
                    msg = f"print what's next (just print {stop_signal} to end):\n"
                else:
                    msg = ""
                output = input(msg)
            output = output[:len(output) - len(stop_signal)]
            reply += output
        else:
            reply = super().human_chat_completion(messages=messages, parse_func=lambda x: x)

        parsed_response = parse_func(reply)

        self.history.append({"input": messages, "output": reply, "parsed_response": parsed_response})

        self.logger.log_metrics(
            {
                "llm:input": messages,
                "llm:output": reply,
                "llm:parsed_response": parsed_response,
            }
        )

        return parsed_response, time.time() - start_time
