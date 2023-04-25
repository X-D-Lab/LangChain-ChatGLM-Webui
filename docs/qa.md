# 常见问题

### 模型下载问题

对于国内用户来说, 直接从HuggingFace下载模型可能会遇到网络阻碍, 您可以先通过以下链接提前将模型下载并解压到本地:  
| large language model | Embedding model |
| :----: | :----: |
| [ChatGLM-6B](https://s3.openi.org.cn/opendata/attachment/b/3/b33c55bb-8e7c-4e9d-90e5-c310dcc776d9?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=1fa9e58b6899afd26dd3%2F20230424%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20230424T014727Z&X-Amz-Expires=604800&X-Amz-SignedHeaders=host&response-content-disposition=attachment%3B%20filename%3D%22chatglm-6b.zip%22&X-Amz-Signature=7324f73e66ee6ec9b955023d4f56076e3817b7daf14e874865c45f409094adf3) | [text2vec-large-chinese](https://s3.openi.org.cn/opendata/attachment/a/2/a2f0edca-1b7b-4dfc-b7c8-15730d33cc3e?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=1fa9e58b6899afd26dd3%2F20230424%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20230424T050110Z&X-Amz-Expires=604800&X-Amz-SignedHeaders=host&response-content-disposition=attachment%3B%20filename%3D%22text2vec-large-chinese.zip%22&X-Amz-Signature=a2e1bdb16f7b55fa05e134649ea1967c0be32d7afbcd300ea82202cc3a7aae6c) |
| [ChatGLM-6B-int8](https://s3.openi.org.cn/opendata/attachment/3/a/3aad10d1-ac8e-48f8-ac5f-cea8b54cf41b?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=1fa9e58b6899afd26dd3%2F20230424%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20230424T014606Z&X-Amz-Expires=604800&X-Amz-SignedHeaders=host&response-content-disposition=attachment%3B%20filename%3D%22chatglm-6b-int8.zip%22&X-Amz-Signature=50f15ed60e4feaffb0984feafd5b2627fa8b5b4105c04a2516b122fb251eedc8) | [ernie-3.0-base-zh](https://s3.openi.org.cn/opendata/attachment/7/3/733fe6e4-2c29-46d8-93e8-6be16194a204?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=1fa9e58b6899afd26dd3%2F20230424%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20230424T050111Z&X-Amz-Expires=604800&X-Amz-SignedHeaders=host&response-content-disposition=attachment%3B%20filename%3D%22ernie-3.0-base-zh.zip%22&X-Amz-Signature=92290028b0a64def599f27804e9314972fd115724ed4ad312a48797d20c5feb1) |
| [ChatGLM-6B-int4](https://s3.openi.org.cn/opendata/attachment/b/2/b2c7f23f-6864-40da-9c81-2c0607cb1d02?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=1fa9e58b6899afd26dd3%2F20230424%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20230424T050113Z&X-Amz-Expires=604800&X-Amz-SignedHeaders=host&response-content-disposition=attachment%3B%20filename%3D%22chatglm-6b-int4.zip%22&X-Amz-Signature=e8204284dcb2138e6fdce87d1b704a39f0dbe362512c28cef5a51cdea78a2858) | [ernie-3.0-nano-zh](https://s3.openi.org.cn/opendata/attachment/2/2/22833889-1683-422e-a44c-929bc379904c?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=1fa9e58b6899afd26dd3%2F20230422%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20230422T152948Z&X-Amz-Expires=604800&X-Amz-SignedHeaders=host&response-content-disposition=attachment%3B%20filename%3D%22ernie-3.0-nano-zh.zip%22&X-Amz-Signature=c8b213d627efb8518c8e54a857c7c323e5e0451a08a3a473d37e2372aabd182f) |
| [ChatGLM-6B-int4-qe](https://s3.openi.org.cn/opendata/attachment/b/f/bf5131da-62e0-4b57-b52a-4135c273b4fc?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=1fa9e58b6899afd26dd3%2F20230424%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20230424T050105Z&X-Amz-Expires=604800&X-Amz-SignedHeaders=host&response-content-disposition=attachment%3B%20filename%3D%22chatglm-6b-int4-qe.zip%22&X-Amz-Signature=3205ca3c5690a086eef95bd032a7314b258a7550ad88bb36c4b738cc5059fbee) | [ernie-3.0-xbase-zh](https://s3.openi.org.cn/opendata/attachment/c/5/c5f746c3-4c60-4fb7-8424-8f7e40f3cce8?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=1fa9e58b6899afd26dd3%2F20230424%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20230424T050103Z&X-Amz-Expires=604800&X-Amz-SignedHeaders=host&response-content-disposition=attachment%3B%20filename%3D%22ernie-3.0-xbase-zh.zip%22&X-Amz-Signature=edffb1ee1e7729fc45305750e1faaff54546683e1e1a983fce4cbff16d28e219) | 
| [Vicuna-7b-1.1](https://s3.openi.org.cn/opendata/attachment/2/5/25854cfb-3d57-44ff-a842-2a98e1a2dafe?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=1fa9e58b6899afd26dd3%2F20230423%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20230423T014232Z&X-Amz-Expires=604800&X-Amz-SignedHeaders=host&response-content-disposition=attachment%3B%20filename%3D%22vicuna-7b-1.1.zip%22&X-Amz-Signature=353d6295d5260d5c53ee512680b211b67fe91fab8376aaef4c17e477f09a666a) | [simbert-base-chinese](https://s3.openi.org.cn/opendata/attachment/1/9/19a54b2f-e527-47e1-aa16-62887498b7f7?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=1fa9e58b6899afd26dd3%2F20230423%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20230423T033222Z&X-Amz-Expires=604800&X-Amz-SignedHeaders=host&response-content-disposition=attachment%3B%20filename%3D%22simbert-base-chinese.zip%22&X-Amz-Signature=6ba81f63582fcb5a45fdc33415aabd40cd8ce0e803d79388390006f5feec5def) | 
| [BELLE-LLaMA-7B-2M](https://s3.openi.org.cn/opendata/attachment/2/6/26f570ea-03c8-4e48-8058-e90b4854edfb?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=1fa9e58b6899afd26dd3%2F20230424%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20230424T045945Z&X-Amz-Expires=604800&X-Amz-SignedHeaders=host&response-content-disposition=attachment%3B%20filename%3D%22BELLE-LLaMA-7B-2M.zip%22&X-Amz-Signature=a3a06bbce4389e21e384d5831f3a484bfae29a4af5a71fb043c26e6282ac00ee) | | 
| [BELLE-LLaMA-13B-2M](https://s3.openi.org.cn/opendata/attachment/a/c/acb0655f-4d3c-49c4-8320-f4b8584cf5bb?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=1fa9e58b6899afd26dd3%2F20230424%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20230424T014910Z&X-Amz-Expires=604800&X-Amz-SignedHeaders=host&response-content-disposition=attachment%3B%20filename%3D%22BELLE-LLaMA-13B-2M.zip%22&X-Amz-Signature=7409fd2eba9768e720380759601cd462deabb3ebb24f493b21e1762b5f3410da) | | 
| Minimax | |

然后在 `app.py` 文件中对以下字段进行修改:  

```python

embedding_model_dict = {

    "ernie-tiny": "your_model_path",
    "ernie-base": "your_model_path",
    "ernie-medium": "your_model_path",
    "ernie-xbase": "your_model_path",
    "text2vec": "your_model_path"
}

llm_model_dict = {

    "ChatGLM-6B": "your_model_path",
    "ChatGLM-6B-int4": "your_model_path",
    "ChatGLM-6B-int8": "your_model_path",
    "ChatGLM-6b-int4-qe": "your_model_path"
}
```

### 爆显存问题

* ChatGLM-6B 模型硬件需求
    | **量化等级**   | **最低 GPU 显存**（推理） | **最低 GPU 显存**（高效参数微调） |
    | -------------- | ------------------------- | --------------------------------- |
    | FP16（无量化） | 13 GB                     | 14 GB                             |
    | INT8           | 8 GB                     | 9 GB                             |
    | INT4           | 6 GB                      | 7 GB                              |

若您的设备显存有限

1. 可以选择 `ChatGLM-6B-int8` 或者 `ChatGLM-6B-int4` 以及选择较小的Embedding Model进行组合使用.
2. 参数选择时，可以选择叫小的history进行尝试.

### 常见的细节问题

1. 需要等文件完全上传之后再进行对话 
2. 若detectron2安装有问题, 可以执行:`pip install 'git+https://openi.pcl.ac.cn/Learning-Develop-Union/detectron2.git'`
