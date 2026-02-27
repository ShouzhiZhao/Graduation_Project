from typing import Any, List, Mapping, Optional

from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
import requests
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
import threading
import re

class SingletonLocalLLM:
    _instance = None
    _lock = threading.Lock()

    @classmethod
    def get_instance(cls, config, logger, api_key, api_base) -> 'LocalLLM':
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = LocalLLM(
                        max_token=config["max_token"],
                        model_path=api_base,
                        logger=logger,
                    )
        return cls._instance


class LocalLLM(LLM):
   
    model_path: str = "local"
    max_token: int
    model_name: str=""
    tokenizer: Any= None
    model: Any= None
    logger: Any= None

    def __init__(self, **data):
        super().__init__(**data)
        self.model_name = data['model_path'].split("/")[-1]
        self.tokenizer = AutoTokenizer.from_pretrained(data['model_path'], trust_remote_code=True)
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        use_cuda = torch.cuda.is_available()
        torch_dtype = torch.float16 if use_cuda else torch.float32
        device_map = 'auto' if use_cuda else None
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                data['model_path'],
                trust_remote_code=True,
                torch_dtype=torch_dtype,
                device_map=device_map,
            )
        except Exception:
            device = 'cuda' if use_cuda else 'cpu'
            self.model = AutoModelForCausalLM.from_pretrained(
                data['model_path'],
                trust_remote_code=True,
                torch_dtype=torch_dtype,
            ).to(device)
        self.model.eval()

    @property
    def _llm_type(self) -> str:
        return self.model_name

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        history: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
    ) -> str:
        message=[{"role": "user", "content": prompt}]
        device = self.model.get_input_embeddings().weight.device
        # Based on Qwen3 official docs: enable_thinking should be passed to apply_chat_template
        text = self.tokenizer.apply_chat_template(
            message,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(device)
        input_ids = model_inputs.input_ids
        attention_mask = model_inputs.attention_mask
        prompt_len = len(input_ids[0])
        output_ids = self.model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=self.max_token,
            do_sample=True,
            num_beams=1,
            temperature=0.7,
            top_p=0.8,
            top_k=20,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        new_ids = output_ids[0][prompt_len:]
        response = self.tokenizer.decode(new_ids, skip_special_tokens=True)
        response = self._strip_think(response)
        response = self._apply_stop(response, stop)
        return response

    def _apply_stop(self, text: str, stop: Optional[List[str]]) -> str:
        if not stop:
            return text
        earliest = None
        for s in stop:
            if not s:
                continue
            idx = text.find(s)
            if idx == -1:
                continue
            if earliest is None or idx < earliest:
                earliest = idx
        if earliest is None:
            return text
        return text[:earliest].rstrip()

    def _strip_think(self, text: str) -> str:
        text = re.sub(r"<\|begin_of_thought\|>[\s\S]*?<\|end_of_thought\|>", "", text, flags=re.IGNORECASE)
        for tag in ["think", "thought", "analysis", "assistant_thought", "assistant_analysis", "internal"]:
            text = re.sub(fr"<{tag}>[\s\S]*?</{tag}>", "", text, flags=re.IGNORECASE)
            text = re.sub(fr"</?{tag}>", "", text, flags=re.IGNORECASE)
        return text.strip()

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {
            "max_token": self.max_token,
            "model Path": self.model_path,
        }
