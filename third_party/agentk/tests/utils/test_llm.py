import threading
import time

import hydra
from agent.loggers.base import FakeLogger
from agent.models.llm import LanguageBackend
from hydra.utils import instantiate
from omegaconf import DictConfig

done = False


def test_llm_response(llm: LanguageBackend) -> bool:
    """
    Test querying the LLM with a sample question.
    Args:
        llm: Initialised language backend instance.

    Returns:
        Status of the LLM query - True/False
    """
    system_message = "Answer the question"
    query = "what is the capital of UK"

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": query}
    ]

    response_text = llm._chat_completion(messages=messages, parse_func=lambda r: r)
    if "london" in response_text[0].lower():
        return True
    print(f"LLM answer to {query} is: {response_text}")
    return False


def spinner() -> None:
    """
    Display spinner animation
    Returns:
        None
    """
    while not done:
        for ch in '|/-\\':
            print(f'\rChecking LLM... {ch} ', end='', flush=True)
            time.sleep(0.1)
            if done:
                break
    print('\r' + ' ' * 20 + '\r', end='', flush=True)


@hydra.main(config_path="../../configs", config_name="test_llm_config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    """
    Instantiate the LLM based on the provided config
    Args:
        cfg: The configuration containing model config, client settings, and other relevant options.

    Returns:
        None
    """
    print("🔧 config loaded")

    # Instantiate config
    client = instantiate(cfg)
    llm_backend = client['llm'](logger=FakeLogger())

    global done
    done = False
    t = threading.Thread(target=spinner)
    t.start()

    # Check LLM query
    check_status = test_llm_response(llm=llm_backend)

    done = True
    t.join()

    if check_status:
        print(f"✅ LLM query test is successful!")
    else:
        print(f"❌ LLM query test failed!")


if __name__ == "__main__":
    """
    Check LLM query

    Example:
        python tests/utils/test_llm.py llm=hf/example_openchat-3.5
    """
    main()
