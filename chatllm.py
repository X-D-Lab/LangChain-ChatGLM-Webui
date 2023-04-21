
import os
from typing import Dict, List, Optional, Tuple, Union

import torch
from fastchat.conversation import (compute_skip_echo_len,
                                   get_default_conv_template)
from fastchat.serve.inference import load_model
from langchain.llms.base import LLM
from langchain.llms.utils import enforce_stop_tokens
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

os.environ["TOKENIZERS_PARALLELISM"] = "false"

DEVICE = "cuda"
DEVICE_ID = "0"
CUDA_DEVICE = f"{DEVICE}:{DEVICE_ID}" if DEVICE_ID else DEVICE


def torch_gc():
    if torch.cuda.is_available():
        with torch.cuda.device(CUDA_DEVICE):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()



class ChatLLM(LLM):
    max_token: int = 10000
    temperature: float = 0.1
    top_p = 0.9
    history = []
    model_name_or_path: str = "THUDM/chatglm-6b",
    tokenizer: object = None
    model: object = None

    def __init__(self):
        super().__init__()

    @property
    def _llm_type(self) -> str:
        return "ChatLLM"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:

        if 'vicuna' in self.model_name_or_path:
            conv = get_default_conv_template(self.model_name_or_path).copy()
            conv.append_message(conv.roles[0], prompt)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            inputs = self.tokenizer([prompt])
            output_ids = self.model.generate(
                torch.as_tensor(inputs.input_ids).cuda(),
                do_sample=True,
                temperature=self.temperature,
                max_new_tokens=self.max_token,
            )
            outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
            skip_echo_len = compute_skip_echo_len(self.model_name_or_path, conv, prompt)
            response = outputs[skip_echo_len:]
            self.history = self.history + [[None, response]]
            
        else:        
            response, _ = self.model.chat(
                self.tokenizer,
                prompt,
                history=self.history,
                max_length=self.max_token,
                temperature=self.temperature,
            )
            torch_gc()
            if stop is not None:
                response = enforce_stop_tokens(response, stop)
            self.history = self.history + [[None, response]]

        return response

    def load_llm(self,
                   llm_device=DEVICE,
                   num_gpus='auto',
                   **kwargs):
        self.model, self.tokenizer = load_model(
                model_path = self.model_name_or_path,
                device = llm_device,
                num_gpus = num_gpus
            )
