from typing import Any, List, Mapping, Optional
import random
import time

from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
import requests
import json
import openai

class CustomLLM(LLM):
    max_token: int
    URL: str = "http://xxxxx"
    api_key: str = ""
    max_retries: int = 0
    retry_min_seconds: float = 0.5
    retry_max_seconds: float = 8.0
    headers: dict = {"Content-Type": "application/json"}
    payload: dict = {"prompt": "", "history": []}
    logger: Any
    model: str

    @property
    def _llm_type(self) -> str:
        return "CustomLLM"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        history: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
    ) -> str:
        attempt = 0
        while True:
            try:
                openai.api_key = self.api_key
                openai.api_base = self.URL
                response = openai.ChatCompletion.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    stop=stop,
                    n=1,
                    max_tokens=self.max_token,
                    temperature=0.7,
                    top_p=0.8,
                    top_k=20,
                    extra_body={"enable_thinking": False},
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                if attempt < self.max_retries:
                    wait_seconds = min(self.retry_min_seconds * (2 ** attempt), self.retry_max_seconds)
                    wait_seconds = wait_seconds * (1 + random.random() * 0.2)
                    self.logger.warning(
                        f"CustomLLM call failed, retrying in {wait_seconds:.2f}s "
                        f"(attempt {attempt + 1}/{self.max_retries}): {e}"
                    )
                    time.sleep(wait_seconds)
                    attempt += 1
                    continue
                self.logger.error(f"CustomLLM error occurred: {e}")
                return str(e)

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {
            "max_token": self.max_token,
            "URL": self.URL,
            "max_retries": self.max_retries,
            "headers": self.headers,
            "payload": self.payload,
        }
