"""
OneAPI Module
"""

from langchain_openai import ChatOpenAI


class OneApi(ChatOpenAI):
    """
    A wrapper for the OneApi class that provides default configuration
    and could be extended with additional methods if needed.

    Args:
        llm_config (dict): Configuration parameters for the language model.
    """

    def __init__(self, **llm_config):
        """Initialize OneApi wrapper mapping generic keys to ChatOpenAI-compatible keys

        - Maps `api_key` to `openai_api_key`
        - Maps `base_url` (DashScope/OpenAI-compatible endpoint) to `openai_api_base`
        """
        if "api_key" in llm_config:
            llm_config["openai_api_key"] = llm_config.pop("api_key")
        if "base_url" in llm_config and "openai_api_base" not in llm_config:
            llm_config["openai_api_base"] = llm_config.pop("base_url")
        super().__init__(**llm_config)
