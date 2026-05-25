import logging
import os
from typing import Any, Dict, List

from langchain_openai import ChatOpenAI
from openai import OpenAI, OpenAIError


from .base_chat import BaseAssistantChat

logger = logging.getLogger(__name__)
# Added basicConfig to see log output
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# --- CONSTANTS for your local server ---
# LOCAL_SERVER_URL = "http://localhost:8010/v1"
LOCAL_SERVER_URL = os.getenv("LOCAL_SERVER_URL")
# API key set via job.slurm
LOCAL_API_KEY = os.getenv("LOCAL_API_KEY") 


class AssistantMLLABChatOpenAI(ChatOpenAI, BaseAssistantChat):
    """OpenAI chat model with LangGraph support."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.initialize_conversation(self)

    def describe(self) -> Dict[str, Any]:
        base_desc = super().describe()
        # In your original code, you used self.openai_proxy.
        # The actual parameter in ChatOpenAI is `openai_api_base`.
        return {**base_desc, "model": self.model_name, "proxy": getattr(self, "openai_api_base", "N/A")}


def get_local_models() -> List[str]:
    """Fetches available models from your local server."""
    try:
        client = OpenAI(
            base_url=LOCAL_SERVER_URL,
            api_key=LOCAL_API_KEY
        )
        models = client.models.list()
        return [model.id for model in models]
    except OpenAIError as e:
        logger.error(f"Error fetching local models from {LOCAL_SERVER_URL}: {e}")
        return []


def create_local_chat(config, session_name: str) -> AssistantMLLABChatOpenAI:
    """Create a chat model instance pointed at your local server."""
    model = config.model

    # We don't need to check for a real OPENAI_API_KEY for a local server
    
    logger.info(f"Using LOCAL model: {model} for session: {session_name}")
    logger.info(f"Connecting to server at: {LOCAL_SERVER_URL}")

    kwargs = {
        "model_name": model,
        "openai_api_key": LOCAL_API_KEY,
        "openai_api_base": LOCAL_SERVER_URL, 
        "session_name": session_name,
        "max_tokens": config.max_tokens,
    }

    if hasattr(config, "temperature"):
        kwargs["temperature"] = config.temperature

    if hasattr(config, "verbose"):
        kwargs["verbose"] = config.verbose

    return AssistantMLLABChatOpenAI(**kwargs)
