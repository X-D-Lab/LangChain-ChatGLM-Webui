import os
import time
import zipfile
from datetime import date, datetime, timedelta

import requests
import torch
from tqdm import tqdm

EMBEDDING_DEVICE = "cuda" if torch.cuda.is_available(
) else "mps" if torch.backends.mps.is_available() else "cpu"

LLM_DEVICE = "cuda" if torch.cuda.is_available(
) else "mps" if torch.backends.mps.is_available() else "cpu"

num_gpus = torch.cuda.device_count()

INIT_LLM_MODEL = 'ChatGLM-6B-int8'
INIT_EMBEDDING_MODEL = 'text2vec-base'

HUGGINGFACE_MODEL_DOWNLOAD = True

embedding_model_dict = {
    "ernie-nano": "nghuyong/ernie-3.0-nano-zh",
    "ernie-base": "nghuyong/ernie-3.0-base-zh",
    "ernie-xbase": "nghuyong/ernie-3.0-xbase-zh",
    "text2vec-base": "GanymedeNil/text2vec-base-chinese"
}

llm_model_dict = {
    "ChatGLM-6B": "THUDM/chatglm-6b",
    "ChatGLM-6B-int4": "THUDM/chatglm-6b-int4",
    "ChatGLM-6B-int8": "THUDM/chatglm-6b-int8",
    "ChatGLM-6b-int4-qe": "THUDM/chatglm-6b-int4-qe",
    #"vicuna-7b-1.1": "/data/vicuna-7b-1.1/", # 需要用户自行提供本地路径
}

embedding_model_download_dict = {
    "ernie-nano":
    'https://s3.openi.org.cn/opendata/attachment/2/2/22833889-1683-422e-a44c-929bc379904c?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=1fa9e58b6899afd26dd3%2F20230416%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20230416T044402Z&X-Amz-Expires=604800&X-Amz-SignedHeaders=host&response-content-disposition=attachment%3B%20filename%3D%22ernie-3.0-nano-zh.zip%22&X-Amz-Signature=6599e60b224d0fc05d13dac7a3648f24c2cba0462f39220142cb91923cfdc3c5',
    "ernie-base":
    'https://s3.openi.org.cn/opendata/attachment/7/3/733fe6e4-2c29-46d8-93e8-6be16194a204?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=1fa9e58b6899afd26dd3%2F20230416%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20230416T044454Z&X-Amz-Expires=604800&X-Amz-SignedHeaders=host&response-content-disposition=attachment%3B%20filename%3D%22ernie-3.0-base-zh.zip%22&X-Amz-Signature=554428b51410671dfc5dd6c928cb3e1291b0235abf7e418894bd4d5ac218123e',
    "ernie-xbase":
    'https://s3.openi.org.cn/opendata/attachment/c/5/c5f746c3-4c60-4fb7-8424-8f7e40f3cce8?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=1fa9e58b6899afd26dd3%2F20230416%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20230416T063343Z&X-Amz-Expires=604800&X-Amz-SignedHeaders=host&response-content-disposition=attachment%3B%20filename%3D%22ernie-3.0-xbase-zh.zip%22&X-Amz-Signature=f2e153cb75ea2dd520b03be88a2e50922c6ca8b86281ebb0b207a9a83254a016',
    "text2vec-base":
    'https://s3.openi.org.cn/opendata/attachment/a/2/a2f0edca-1b7b-4dfc-b7c8-15730d33cc3e?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=1fa9e58b6899afd26dd3%2F20230416%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20230416T044328Z&X-Amz-Expires=604800&X-Amz-SignedHeaders=host&response-content-disposition=attachment%3B%20filename%3D%22text2vec-large-chinese.zip%22&X-Amz-Signature=7468efbc7700f652e61386fe0d04b4d36dbd6cb8ff46d4cfd17c0f37bbaf868e'
}
llm_model_download_dict = {
    "ChatGLM-6B":
    'https://s3.openi.org.cn/opendata/attachment/b/3/b33c55bb-8e7c-4e9d-90e5-c310dcc776d9?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=1fa9e58b6899afd26dd3%2F20230416%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20230416T025911Z&X-Amz-Expires=604800&X-Amz-SignedHeaders=host&response-content-disposition=attachment%3B%20filename%3D%22chatglm-6b.zip%22&X-Amz-Signature=89de83c6dae3702387d14078845b3728a6b09e5e84fc57dbe66c1566f43482a7',
    "ChatGLM-6B-int4":
    'https://s3.openi.org.cn/opendata/attachment/b/2/b2c7f23f-6864-40da-9c81-2c0607cb1d02?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=1fa9e58b6899afd26dd3%2F20230415%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20230415T155352Z&X-Amz-Expires=604800&X-Amz-SignedHeaders=host&response-content-disposition=attachment%3B%20filename%3D%22chatglm-6b-int4.zip%22&X-Amz-Signature=0488bd8a55e0b52c846630d609e68d2fa05bd0f0b057059f4f94133a17fbd35b',
    "ChatGLM-6B-int8":
    'https://s3.openi.org.cn/opendata/attachment/3/a/3aad10d1-ac8e-48f8-ac5f-cea8b54cf41b?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=1fa9e58b6899afd26dd3%2F20230416%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20230416T032447Z&X-Amz-Expires=604800&X-Amz-SignedHeaders=host&response-content-disposition=attachment%3B%20filename%3D%22chatglm-6b-int8.zip%22&X-Amz-Signature=d58c08158ef8550719f934916fe4b6afe67220a9b84036f660e952c07b8b44f6',
    "ChatGLM-6b-int4-qe":
    'https://s3.openi.org.cn/opendata/attachment/b/f/bf5131da-62e0-4b57-b52a-4135c273b4fc?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=1fa9e58b6899afd26dd3%2F20230416%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20230416T051728Z&X-Amz-Expires=604800&X-Amz-SignedHeaders=host&response-content-disposition=attachment%3B%20filename%3D%22chatglm-6b-int4-qe.zip%22&X-Amz-Signature=9a137b222f4e0b39c369966c1c1c1d02712728d06185e4e6501a4ae22566c3dc',
    "Vicuna-7b-1.1":
    'https://s3.openi.org.cn/opendata/attachment/2/5/25854cfb-3d57-44ff-a842-2a98e1a2dafe?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=1fa9e58b6899afd26dd3%2F20230421%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20230421T110022Z&X-Amz-Expires=604800&X-Amz-SignedHeaders=host&response-content-disposition=attachment%3B%20filename%3D%22vicuna-7b-1.1.zip%22&X-Amz-Signature=c0fc5e9cbc48194ffa38d9d87cd2c476230c6536440d3daf961384b4f7f25871',
    "BELLE-LLaMA-7B-2M":
    "https://s3.openi.org.cn/opendata/attachment/2/6/26f570ea-03c8-4e48-8058-e90b4854edfb?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=1fa9e58b6899afd26dd3%2F20230422%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20230422T092629Z&X-Amz-Expires=604800&X-Amz-SignedHeaders=host&response-content-disposition=attachment%3B%20filename%3D%22BELLE-LLaMA-7B-2M.zip%22&X-Amz-Signature=c8a3f1c6afe3735134b39c7267a55cfe02ec33121307b7f27867576ea0cd85ae"
}
for llm_model_name in llm_model_download_dict:
    url = llm_model_download_dict[llm_model_name]
    response = requests.get(url, stream=True)
    total = int(response.headers.get('content-length', 0))
    file_path = os.path.join('../model/', llm_model_name + ".zip")
    with open(file_path, "wb") as file, tqdm(
            desc=file_path,
            total=total,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)
    with zipfile.ZipFile(os.path.join('../model/', llm_model_name + ".zip"),
                         'r') as zip_ref:
        zip_ref.extractall('../model/')
    os.remove(os.path.join('../model/', llm_model_name + ".zip"))
    print("Downloaded and extracted " + llm_model_name)
