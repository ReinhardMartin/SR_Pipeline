import os
import threading

from openai import OpenAI

from core.retry import retry_call


class ChatClient:
    def __init__(self, config: dict):
        self.config = config
        key_name = config["api_key_env"]
        key = os.environ.get(key_name) if key_name else "unused"
        self._client = OpenAI(
            api_key=key or "unused", base_url=config["base_url"],
            timeout=config["timeout"], max_retries=0,
        )
        self._semaphore = threading.BoundedSemaphore(config["max_concurrent_requests"])

    def complete(self, system: str, user: str, max_tokens: int) -> str:
        cfg = self.config
        if cfg["api_key_env"] and not os.environ.get(cfg["api_key_env"]):
            raise ValueError(f"Missing LLM credential: {cfg['api_key_env']}")
        request = {
            "model": cfg["model"],
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            cfg["token_parameter"]: max_tokens,
        }
        if cfg["temperature"] is not None:
            request["temperature"] = cfg["temperature"]
        if cfg["json_mode"]:
            request["response_format"] = {"type": "json_object"}

        def send():
            with self._semaphore:
                return self._client.chat.completions.create(**request)

        response = retry_call(
            send, retries=cfg["retries"], initial=cfg["retry_initial_seconds"],
            maximum=cfg["retry_max_seconds"],
        )
        if not response.choices:
            raise ValueError("LLM returned no choices")
        choice = response.choices[0]
        if choice.finish_reason != "stop" or not choice.message.content:
            raise ValueError(f"LLM returned incomplete content ({choice.finish_reason})")
        return choice.message.content

    def close(self):
        self._client.close()
