import os
from typing import List, Optional

import torch
from langchain.llms.base import LLM
from langchain.llms.utils import enforce_stop_tokens

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

os.environ["TOKENIZERS_PARALLELISM"] = "false"

DEVICE = "cuda"
DEVICE_ID = "0"
CUDA_DEVICE = f"{DEVICE}:{DEVICE_ID}" if DEVICE_ID else DEVICE


def torch_gc():
    if torch.cuda.is_available():
        with torch.cuda.device(CUDA_DEVICE):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


class ChatGLM(LLM):
    history = []
    max_length = 10000
    temperature: float = 0.01
    top_p = 0.9
    model_name = 'ZhipuAI/ChatGLM-6B'
    model_revision = 'v1.0.13'

    def __init__(self):
        super().__init__()

    @property
    def _llm_type(self) -> str:
        return "ChatGLM"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:

        inputs = {
            'text': prompt,
            'history': self.history,
            'max_length': self.max_length,
            'temperature': self.temperature,
            'top_p': self.top_p
        }
        pipe = self.pipe()
        result = pipe(inputs)
        response = result['response']
        updated_history = result['history']
        torch_gc()
        if stop is not None:
            response = enforce_stop_tokens(response, stop)
        self.history = updated_history
        return response

    def pipe(self):
        pipe = pipeline(task=Tasks.chat,
                        model=self.model_name,
                        model_revision=self.model_revision)
        return pipe
