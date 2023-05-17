
import os
from typing import Dict, List, Optional, Tuple, Union

from langchain.llms.base import LLM
from langchain.llms.utils import enforce_stop_tokens
from paddlenlp import Taskflow

chatbot = Taskflow("text2text_generation", batch_size=2)


class ChatLLM(LLM):
    max_token: int = 10000
    temperature: float = 0.1
    top_p = 0.9
    history = []
    tokenizer: object = None
    model: object = None

    def __init__(self):
        super().__init__()

    @property
    def _llm_type(self) -> str:
        return "ChatLLM"

    def _call(self,
              prompt: str,
              stop: Optional[List[str]] = None) -> str:
        print(prompt)
        prompt_list = []
        prompt_list.append(prompt)
        print(prompt_list)

        results = chatbot(prompt_list)

        response = results['result'][0]
        print(response)
        if stop is not None:
            response = enforce_stop_tokens(response, stop)
        return response