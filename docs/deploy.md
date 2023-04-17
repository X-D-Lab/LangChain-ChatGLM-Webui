# 部署文档

## 环境准备

项目需要Python>=3.8.1

1. git clone本项目, 您可以在自己的terminal中执行: `git clone https://github.com/thomas-yanxin/LangChain-ChatGLM-Webui.git`. 若国内用户访问Github存在网络问题, 也可以执行: `https://openi.pcl.ac.cn/Learning-Develop-Union/LangChain-ChatGLM-Webui.git`
2. 进入本项目目录：`cd LangChain-ChatGLM-Webui`
3. 安装依赖包：`pip install -r requirements.txt`, 国内用户可设置清华源加速下载.

另: 若您想要安装测试ModelScope版本, 需要额外安装ModelScope包: `pip install modelscope==1.4.3 -f https://modelscope.oss-cn-beijing.aliyuncs.com/releases/repo.html `

## 启动程序

### Huggingface版本

在terminal中执行命令: `python3 app.py`

### ModelScope版本

1. 进入modelscope文件目录：`cd modelscope`
2. 执行命令：`python3 app.py`

## 常见问题

### 模型下载问题

对于国内用户来说, 直接从HuggingFace下载模型可能会遇到网络阻碍, 您可以先通过以下链接提前将模型下载并解压到本地:  
| large language model | Embedding model |
| :----: | :----: |
| [ChatGLM-6B](https://s3.openi.org.cn/opendata/attachment/b/3/b33c55bb-8e7c-4e9d-90e5-c310dcc776d9?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=1fa9e58b6899afd26dd3%2F20230416%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20230416T025911Z&X-Amz-Expires=604800&X-Amz-SignedHeaders=host&response-content-disposition=attachment%3B%20filename%3D%22chatglm-6b.zip%22&X-Amz-Signature=89de83c6dae3702387d14078845b3728a6b09e5e84fc57dbe66c1566f43482a7) | [text2vec-large-chinese](https://s3.openi.org.cn/opendata/attachment/a/2/a2f0edca-1b7b-4dfc-b7c8-15730d33cc3e?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=1fa9e58b6899afd26dd3%2F20230416%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20230416T044328Z&X-Amz-Expires=604800&X-Amz-SignedHeaders=host&response-content-disposition=attachment%3B%20filename%3D%22text2vec-large-chinese.zip%22&X-Amz-Signature=7468efbc7700f652e61386fe0d04b4d36dbd6cb8ff46d4cfd17c0f37bbaf868e) |
| [ChatGLM-6B-int8](https://s3.openi.org.cn/opendata/attachment/3/a/3aad10d1-ac8e-48f8-ac5f-cea8b54cf41b?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=1fa9e58b6899afd26dd3%2F20230416%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20230416T032447Z&X-Amz-Expires=604800&X-Amz-SignedHeaders=host&response-content-disposition=attachment%3B%20filename%3D%22chatglm-6b-int8.zip%22&X-Amz-Signature=d58c08158ef8550719f934916fe4b6afe67220a9b84036f660e952c07b8b44f6) | [ernie-3.0-base-zh](https://s3.openi.org.cn/opendata/attachment/7/3/733fe6e4-2c29-46d8-93e8-6be16194a204?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=1fa9e58b6899afd26dd3%2F20230416%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20230416T044454Z&X-Amz-Expires=604800&X-Amz-SignedHeaders=host&response-content-disposition=attachment%3B%20filename%3D%22ernie-3.0-base-zh.zip%22&X-Amz-Signature=554428b51410671dfc5dd6c928cb3e1291b0235abf7e418894bd4d5ac218123e) |
| [ChatGLM-6B-int4](https://s3.openi.org.cn/opendata/attachment/b/2/b2c7f23f-6864-40da-9c81-2c0607cb1d02?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=1fa9e58b6899afd26dd3%2F20230415%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20230415T155352Z&X-Amz-Expires=604800&X-Amz-SignedHeaders=host&response-content-disposition=attachment%3B%20filename%3D%22chatglm-6b-int4.zip%22&X-Amz-Signature=0488bd8a55e0b52c846630d609e68d2fa05bd0f0b057059f4f94133a17fbd35b) | [ernie-3.0-nano-zh](https://s3.openi.org.cn/opendata/attachment/2/2/22833889-1683-422e-a44c-929bc379904c?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=1fa9e58b6899afd26dd3%2F20230416%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20230416T044402Z&X-Amz-Expires=604800&X-Amz-SignedHeaders=host&response-content-disposition=attachment%3B%20filename%3D%22ernie-3.0-nano-zh.zip%22&X-Amz-Signature=6599e60b224d0fc05d13dac7a3648f24c2cba0462f39220142cb91923cfdc3c5) |
| [ChatGLM-6B-int4-qe](https://s3.openi.org.cn/opendata/attachment/b/f/bf5131da-62e0-4b57-b52a-4135c273b4fc?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=1fa9e58b6899afd26dd3%2F20230416%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20230416T051728Z&X-Amz-Expires=604800&X-Amz-SignedHeaders=host&response-content-disposition=attachment%3B%20filename%3D%22chatglm-6b-int4-qe.zip%22&X-Amz-Signature=9a137b222f4e0b39c369966c1c1c1d02712728d06185e4e6501a4ae22566c3dc) | [ernie-3.0-xbase-zh](https://s3.openi.org.cn/opendata/attachment/c/5/c5f746c3-4c60-4fb7-8424-8f7e40f3cce8?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=1fa9e58b6899afd26dd3%2F20230416%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20230416T063343Z&X-Amz-Expires=604800&X-Amz-SignedHeaders=host&response-content-disposition=attachment%3B%20filename%3D%22ernie-3.0-xbase-zh.zip%22&X-Amz-Signature=f2e153cb75ea2dd520b03be88a2e50922c6ca8b86281ebb0b207a9a83254a016) | 
|  | [ernie-3.0-medium-zh](https://s3.openi.org.cn/opendata/attachment/8/e/8e57b1ad-f044-4fa8-ba8b-8ca1e8257313?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=1fa9e58b6899afd26dd3%2F20230416%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20230416T061240Z&X-Amz-Expires=604800&X-Amz-SignedHeaders=host&response-content-disposition=attachment%3B%20filename%3D%22ernie-3.0-medium-zh.zip%22&X-Amz-Signature=5af6f2d308cb81df949248f878889c7ceb90beb2e983666fdd695c8f9cf91832) |  

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
