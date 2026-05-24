import hashlib
import logging
import os
import time
from typing import Any, Callable

import numpy as np
import openai
from openai import OpenAI
import requests.exceptions
from huggingface_hub import (
    ChatCompletionOutput,
    ChatCompletionOutputComplete,
    ChatCompletionOutputMessage,
    ChatCompletionOutputUsage,
    InferenceClient,
)
# from huggingface_hub.errors import HfHubHTTPError
# Compatibility import for different huggingface_hub versions
try:
    # Some older versions
    from huggingface_hub.errors import HfHubHTTPError  # type: ignore
except ImportError:
    try:
        # Newer 0.2x+ versions (like 0.23.4) keep it under utils._errors
        from huggingface_hub.utils._errors import HfHubHTTPError  # type: ignore
    except ImportError:
        # Extreme fallback: define a dummy error so code still runs
        class HfHubHTTPError(RuntimeError):
            pass

from openai import OpenAI
from requests import HTTPError
from requests.exceptions import ProxyError
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    wait_fixed,
)

from agent.models.embeddings import EmbeddingBackend
from agent.models.llm import (
    HumanContinueAction,
    HumanFixAction,
    HumanGuidanceAction,
    HumanResampleLLMAction,
    HumanSeqPrintModeAction,
    HumanSupervisionChecker,
    LanguageBackend, HumanIncreaseWindowSizeAction, HumanDecreaseWindowSizeAction
)
from agent.utils.utils import (
    HumanInputCancelError,
    get_terminal_width,
    human_input,
    print_in_two_cols,
)
from ds_agent.utils import save_w_pickle, load_w_pickle

logger_ = logging.getLogger(__name__)


def _num_tokens_from_messages(messages: list[dict[str, str]], model: str) -> int:
    """Return the number of tokens used by a list of messages.

    Args:
        messages (list[dict[str, str]]): The list of messages to count tokens for.

    Returns:
        int: The number of tokens after encoding the prompt.

    Taken from OpenAI's Cookbook: https://github.com/openai/openai-
    cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
    """
    import tiktoken

    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    if model in {
        "gpt-3.5-turbo-0613",
        "gpt-3.5-turbo-16k-0613",
        "gpt-4-0314",
        "gpt-4-32k-0314",
        "gpt-4-0613",
        "gpt-4-32k-0613",
    }:
        tokens_per_message = 3
        tokens_per_name = 1
    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif "gpt-3.5-turbo" in model:
        print("Warning: gpt-3.5-turbo may update over time. Returning num tokens assuming gpt-3.5-turbo-0613.")
        return _num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613")
    elif "gpt-4" in model:
        print("Warning: gpt-4 may update over time. Returning num tokens assuming gpt-4-0613.")
        return _num_tokens_from_messages(messages, model="gpt-4-0613")
    else:
        raise NotImplementedError(
            f"""num_tokens_from_messages() is not implemented for model {model}.
            See https://github.com/openai/openai-python/blob/main/chatml.md for
            information on how messages are converted to tokens."""
        )
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens


def is_hf_hub_rate_limit_exception(e: BaseException) -> bool:
    print(e, flush=True)
    return isinstance(e, HfHubHTTPError) and e.response.status_code == 429 and "Rate limit reached" in e.server_message


def is_internal_server_error_exception(e: BaseException) -> bool:
    """Check for internal server error 500"""
    print(e, flush=True)
    return isinstance(e, HTTPError) and e.response.status_code == 500


class OpenAIAPILanguageBackend(LanguageBackend):
    def __init__(self, client: InferenceClient | OpenAI, model_id: str, context_length: int, logger: Any = None,
                 responses_from_file: list[str] | None = None, continue_after_end_of_file: bool = True,
                 no_system_prompt: bool = False, **kwargs):
        """
        Args:
            responses_from_file:  if set, the answers will be taken from the list rather than queried from the LLM
            continue_after_end_of_file: the LLM will be queried after the responses from `responses_from_file` are
                                        exhausted - set it to False if you want to raise an error instead and
                                        avoid continuing the episode by querying the LLM.
            no_system_prompt: If set, it combines system prompt and user prompt into one, and removes system prompt.
        """
        self.logger = logger or logging.getLogger(__name__)
        super().__init__(model_id=model_id, logger=self.logger, context_length=context_length)
        
        self.client = client
        self.generation_kwargs = kwargs
        self.responses_from_file = responses_from_file
        self.continue_after_end_of_file = continue_after_end_of_file
        self.human_guidance = ""  # save the potential guidance added during human supervision
        self.screen_width = get_terminal_width()
        self.no_system_prompt = no_system_prompt

        
        # 👇 ADD THIS HERE — perfect placement
        # self.api_key = os.getenv("OPENAI_API_KEY", "api")
        # self.base_url = os.getenv("OPENAI_BASE_URL", "http://releagpu01.tf.ki.privat:8010/v1")
        # self.base_url = os.getenv("OPENAI_BASE_URL", "http://localhost:8000/v1")
        self.base_url = os.getenv("OPENAI_BASE_URL", "https://openwebui.uni-freiburg.de/api")
        self.api_key = os.getenv("OPENAI_API_KEY", "sk-494ce4dab1034d1e8965f07c445be4c8")

        # DEBUGGING CODE
        # print("OpenAIAPILanguageBackend INIT CALLED")

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_fixed(5),
        retry=retry_if_exception_type(
            (openai.APIConnectionError,
             openai.RateLimitError,
             HTTPError,
             HfHubHTTPError,
             ProxyError,
             requests.exceptions.ConnectionError)
        ),
        before_sleep=before_sleep_log(logger=logger_, log_level=logging.WARNING)
    )
    @retry(
        retry=retry_if_exception(lambda e: is_hf_hub_rate_limit_exception(e) or is_internal_server_error_exception(e)),
        wait=wait_exponential(max=3600),
        before_sleep=before_sleep_log(logger=logger_, log_level=logging.WARNING)
    )
    def _chat_completion(self, messages: list[dict[str, str]], parse_func: Callable, **kwargs) -> tuple[str, float]:
        """Generates a text completion for a given prompt in a chat-like interaction.

        Args:
            messages (list[dict[str, str]]): The input text prompt to generate completion for.
            parse_func (Callable): A function to parse the model's response.
            **kwargs: Additional keyword arguments that may be required for the generation,
                      such as temperature, max_tokens, etc.

        Returns:
            str: The generated text completion.
            runtime: time taken to execute this function (won't take retrials into account)
        """
        start_time = time.time()
        
        # # DEBUGGING CODE
        # print("🚀 ENTERED CHAT COMPLETION")
        # print("MODEL USED:", self.model_id)

         # Ensure we have an OpenAI client even if none was passed from Hydra
        # if getattr(self, "client", None) is None:
        #     self.client = OpenAI()  # uses OPENAI_API_KEY from env
        if getattr(self, "client", None) is None:
            self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        

        # Check if responses are already in cache
        path_to_saved_responses: str | None = kwargs.pop("path_to_saved_responses", None)

        # Check is response is pre-specified in the argument
        predefined_response: str | None = kwargs.pop("predefined_response", None)

        if path_to_saved_responses is not None and self.responses_from_file is not None:
            raise ValueError("Choose between reading answers from file or from cache!")
        try:
            if not self.human_mode:
                try:
                    save_response_filepath = None
                    response = None

                    response_content = None
                    response_id = None
                    if predefined_response is not None:
                        response_content = predefined_response
                        response_id = "PREDEFINED_RESPONSE"
                    elif self.responses_from_file is not None:
                        if len(self.responses_from_file) > 0:
                            response_content = self.responses_from_file.pop(0)
                            response_id = "READ_FROM_FILE"
                        elif not os.getenv("FILE_RESPONSES_FALLBACK_TO_LLM"):
                            raise RuntimeError("No more responses in file! Set FILE_RESPONSES_FALLBACK_TO_LLM env "
                                               "var if you want to use the LLM at this point")

                    if response_content is not None:
                        resp_message = ChatCompletionOutputMessage(content=response_content, role="assistant")
                        choices = [ChatCompletionOutputComplete(finish_reason="stop", index=0, message=resp_message)]
                        response = ChatCompletionOutput(
                            id=response_id,
                            choices=choices,
                            created=int(time.time()),
                            model=response_id,
                            system_fingerprint="",
                            usage=ChatCompletionOutputUsage(completion_tokens=0, prompt_tokens=0, total_tokens=0)
                        )
                    elif path_to_saved_responses is not None:
                        # Before querying the LLM, check if the response was not given previously
                        name = self.get_cached_response_path(messages=messages)
                        save_response_filepath = os.path.join(path_to_saved_responses, name)
                        if os.path.exists(save_response_filepath):
                            try:
                                response = load_w_pickle(path=save_response_filepath)
                                self.logger.log_metrics({"llm:retrieved_response": True})
                            except Exception as e:
                                response = None
                                self.logger.log_metrics({"llm:retrieved_response": str(e)})

                    if response is None:
                        if not self.continue_after_end_of_file and self.responses_from_file is not None and len(
                                self.responses_from_file) == 0:
                            raise RuntimeError("Responses from file exhausted "
                                               "and `continue_after_end_of_file` set to `False`. Stopping.")

                        # Combine system prompt and user prompt message and remove system prompt
                        if self.no_system_prompt and messages[0].get('role') == 'system':
                            new_content = messages[0].get('content') + '\n\n' + messages[1].get('content')
                            messages[1]['content'] = new_content
                            messages.pop(0)

                        # Query the LLM
                        # Merge default generation kwargs with call-time kwargs
                        merged_kwargs = {**self.generation_kwargs, **kwargs}

                        # The OpenAI chat.completions API does not accept tokenizer_kwargs
                        merged_kwargs.pop("tokenizer_kwargs", None)
                        merged_kwargs.pop("base_url", None)
                        merged_kwargs.pop("api_key", None)

                        # print("🚀 ABOUT TO CALL LLM")
                        # print("MODEL:", self.model_id)
                        # print("First message:", messages[0])
                        
                        try:
                            response = self.client.chat.completions.create(
                                model=self.model_id,
                                messages=messages,
                                **merged_kwargs,
                            )
                            # print("✅ LLM CALL SUCCESS")
                        except Exception as e:
                            print("🔥 LLM CALL FAILED")
                            import traceback
                            traceback.print_exc()
                            raise e
                        
                        # print("📦 RESPONSE RECEIVED")
                        # print(response)
                        # response = self.client.chat.completions.create(
                        #     model=self.model_id,
                        #     messages=messages,
                        #     **{**self.generation_kwargs, **kwargs},  # This allows kwargs to overwrite generation_kwargs
                        # )
                        if save_response_filepath is not None:
                            # couldn't use json with ChatCompletion type...
                            save_w_pickle(obj=response, path=save_response_filepath, filename=None, overwrite=False)

                except openai.BadRequestError as e:
                    # check if the message is too long
                    import re
                    max_len_err = re.findall("This model\'s maximum context length is [0-9]+ tokens", e.message)
                    if len(max_len_err) > 0:
                        raise
                    raise
                # raw_output_text = response.choices[0].message.content.strip()
                raw_output_text = response.choices[0].message.content

                raw_output_text = raw_output_text.strip()

                # print("\n===== RAW MODEL OUTPUT =====")
                # print(raw_output_text)
                # print("============================\n")

                if raw_output_text is None:
                    raise ValueError("Model returned empty content")

                raw_output_text = raw_output_text.strip()

                api_usage_input = response.usage.prompt_tokens
                api_usage_output = response.usage.completion_tokens
            else:
                raw_output_text = super().human_chat_completion(
                    messages=messages, parse_func=lambda x: x
                )
                api_usage_input = 0
                api_usage_output = 0
        except openai.APIConnectionError as e:
            print(f"API Connection Error: {e}")
            raise
        except openai.RateLimitError as e:
            print(f"Rate Limit Exceeded: {e}")
            raise
        except openai.OpenAIError as e:
            print(f"General OpenAI API Error: {e}")
            raise
        human_mode = self.human_mode

        self.logger.log_metrics(
            {
                "llm:input": messages,
                "llm:output": raw_output_text,
                "api_usage:input": api_usage_input,
                "api_usage:output": api_usage_output,
            }
        )

        human_supervise = os.getenv("HUMAN_SUPERVISE", False)
        if human_supervise in ["0", "False"]:
            human_supervise = False

        prompt_to_display = ""
        raw_response_to_display = ""

        if human_supervise and not self.human_mode:
            prompt_to_display += "================================================\n"
            prompt_to_display += "==================== PROMPT ====================\n"
            prompt_to_display += "================================================\n"
            for m in messages:
                prompt_to_display += f"[[ {m['role']} ]]\n"
                prompt_to_display += f"{m['content']}\n"

            raw_response_to_display += "====================================================\n"
            raw_response_to_display += "=================== RAW RESPONSE ===================\n"
            raw_response_to_display += "====================================================\n"
            raw_response_to_display += raw_output_text + "\n"
        try:
            parsed_response = parse_func(raw_output_text)

        except Exception as e:
            print("\n❌ PARSING FAILED ❌")
            print("RAW OUTPUT FROM MODEL:")
            print("--------------------------------------------------")
            print(raw_output_text)
            print("--------------------------------------------------\n")

            if human_supervise and not self.human_mode:
                print_in_two_cols(t1=prompt_to_display, t2=raw_response_to_display, w=self.screen_width)

            self.logger.log_metrics(
                {
                    "llm:human_mode": human_mode,
                    "llm:parsed_error": str(e),
                }
            )
            self.logger.save_metrics()
            raise

        if human_supervise and not self.human_mode:
            response_to_display = ""
            if raw_output_text != parsed_response:
                response_to_display += raw_response_to_display + "\n"

            response_to_display += "=======================================================\n"
            response_to_display += "=================== PARSED RESPONSE ===================\n"
            response_to_display += "=======================================================\n"
            response_to_display += str(parsed_response)
            print_in_two_cols(t1=prompt_to_display, t2=response_to_display, w=self.screen_width)

            human_supervision_handler = HumanSupervisionChecker(
                allow_fix=True, allow_continue=True, allow_resampling=True, allow_guidance=True, allow_seq_print=True,
                length=self.screen_width - 1
            )

            while True:
                human_supervision_handler.length = self.screen_width
                action = human_supervision_handler.query_human()

                if isinstance(action, HumanIncreaseWindowSizeAction):
                    self.screen_width = int(1.5 * self.screen_width)
                    print_in_two_cols(t1=prompt_to_display, t2=response_to_display, w=self.screen_width)
                    continue

                if isinstance(action, HumanDecreaseWindowSizeAction):
                    self.screen_width = max(5, int(0.8 * self.screen_width))
                    print_in_two_cols(t1=prompt_to_display, t2=response_to_display, w=self.screen_width)
                    continue

                if isinstance(action, HumanSeqPrintModeAction):
                    human_supervision_handler.allow_seq_print = False
                    print(prompt_to_display)
                    print(response_to_display)
                    action = human_supervision_handler.query_human()

                if isinstance(action, HumanContinueAction):
                    print("Continuing!\n")
                    break
                elif isinstance(action, HumanFixAction):
                    print("Get human fix!")
                    try:
                        fixed_reply = human_input(allow_cancel=True, w=self.screen_width)
                    except HumanInputCancelError:
                        print("Cancel human fix!")
                        continue
                    raw_output_text = fixed_reply
                    parsed_response = parse_func(raw_output_text)
                    human_mode = True

                    break
                elif isinstance(action, HumanResampleLLMAction):
                    print("Get a new answer!")
                    return self._chat_completion(messages=messages, parse_func=parse_func,
                                                 path_to_saved_responses=path_to_saved_responses, **kwargs)
                elif isinstance(action, HumanGuidanceAction):
                    print("Provide guidance to get a better answer!")
                    assert messages[-1]["role"] == "user", messages[-1]
                    try:
                        guidance = "\n" + human_input(allow_cancel=True, w=self.screen_width)
                    except HumanInputCancelError:
                        print("Cancel human guidance!")
                        continue
                    last_content = messages[-1]["content"]
                    if len(self.human_guidance) > 0 and last_content[-len(self.human_guidance):] == self.human_guidance:
                        last_content = messages[-1]["content"][:-len(self.human_guidance)]
                    self.human_guidance = guidance
                    last_content += guidance
                    messages[-1]["content"] = last_content
                    return self._chat_completion(
                        messages=messages, parse_func=parse_func, path_to_saved_responses=path_to_saved_responses,
                        **kwargs
                    )
                else:
                    raise ValueError(action)

        self.history.append({"input": messages, "output": raw_output_text, "parsed_response": parsed_response})
        self.logger.log_metrics(
            {
                "llm:parsed_response": parsed_response,
                "llm:human_mode": human_mode,
            }
        )
        self.logger.save_metrics()

        return parsed_response, time.time() - start_time

    def get_cached_response_path(self, messages: list[dict[str, str]]) -> str:
        # We manually map the model id from hf-hub to llm-playground to share the cache
        # Actually: we don't because quantisation
        # model_name = {
        #     "Qwen/Qwen2.5-72B-Instruct": "qwen2.5-72b-instruct",
        #     "Qwen/Qwen2-72B-Instruct": "qwen2-72b-32k"
        # }.get(self.model_id, self.model_id)
        model_name = self.model_id
        messages_hash = hashlib.md5(str(messages).encode('utf-8')).digest().hex()
        return f"{model_name}_{messages_hash}.pkl"

    def count_tokens(self, prompt: list[dict[str, str]]) -> int:
        return _num_tokens_from_messages(prompt, self.model_id)


class OpenAIAPIEmbeddingBackend(EmbeddingBackend):
    """Embedding backend using OpenAI's API."""

    def __init__(self, api_key: str | None, server_ip: str, **kwargs):
        super().__init__(**kwargs)
        self.api_key = api_key
        self.base_url = server_ip
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def embed_text(self, text: str) -> list[float]:
        text = text.replace("\n", " ")
        ret_val = self.client.embeddings.create(input=[text], model=self.model_id)
        return np.array(ret_val.data[0].embedding)

    def batch_embed_texts(self, texts: list[str]) -> list[list[float]]:
        texts = [text.replace("\n", " ") for text in texts]
        ret_val = self.client.embeddings.create(input=texts, model=self.model_id)
        return np.vstack(ret_val.data[0].embedding)
