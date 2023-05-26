# 部署文档

## 直接安装

### 环境准备

**项目需要Python>=3.8.1, 默认已安装torch**

1. git clone本项目, 您可以在自己的terminal中执行: `git clone https://github.com/thomas-yanxin/LangChain-ChatGLM-Webui.git`. 若国内用户访问Github存在网络问题, 也可以执行: `https://openi.pcl.ac.cn/Learning-Develop-Union/LangChain-ChatGLM-Webui.git`
2. 进入本项目目录：`cd LangChain-ChatGLM-Webui`
3. 安装依赖包：`pip install -r requirements.txt`, 国内用户可设置清华源加速下载.

另: 若您想要安装测试ModelScope版本, 需要额外安装ModelScope包: `pip install modelscope==1.4.3 -f https://modelscope.oss-cn-beijing.aliyuncs.com/releases/repo.html `

[OpenI启智社区](https://openi.pcl.ac.cn/Learning-Develop-Union/LangChain-ChatGLM-Webui)部署推荐的Docker镜像: `dockerhub.pcl.ac.cn:5000/user-images/openi:LangChain-ChatLLM-2.0`

### 启动程序

* Huggingface版本

在terminal中执行命令: `python3 app.py`

* ModelScope版本

1. 进入modelscope文件目录：`cd modelscope`
2. 执行命令：`python3 app.py`

## Docker 基础环境运行

1. 运行镜像：`docker run -it --rm --runtime=nvidia --gpus all --network host registry.cn-beijing.aliyuncs.com/public-development-resources/langchain-chatglm-webui:Base bash`
2. git clone项目: `git clone https://github.com/thomas-yanxin/LangChain-ChatGLM-Webui.git`
3. 进入本项目目录：`cd LangChain-ChatGLM-Webui`
4. 安装依赖包：`pip3 install -r requirements.txt`
5. 执行app.py：`python3 app.py`

## Docker 小白运行

1. 运行镜像：`docker run -d --name langchain-ChatGLM-webui --runtime=nvidia --gpus all --network host registry.cn-beijing.aliyuncs.com/public-development-resources/langchain-chatglm-webui:latest`
2. 访问服务：`http://ip:7860`
3. 运行环境，镜像大小约14G。
4. nvidia-runtime 请参考: [container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
5. 本地模型放置目录：  
 BELLE-LLaMA-Local:/pretrainmodel/belle  
 Vicuna-Local:/pretrainmodel/vicuna  
 ChatGLM-Local:/pretrainmodel/chatglm

6. 挂载cache目录，容器重启或更新无需重新下载相关模型。  
 `-v langchain-ChatGLM-webui-cache:/root/.cache/`

## Jina Serving API

1. 启动服务：`lc-serve deploy local jina_serving`

2. 执行curl初始化模型命令  

```bash
curl -X 'POST' \
  'http://localhost:8080/reinit_model' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
    "large_language_model": "ChatGLM-6B-int8",
    "embedding_model": "text2vec-base"
  }'
```

3. 执行curl构建向量库命令

```bash
curl -X 'POST' \
  'http://localhost:8080/vector_store' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
    "file_path": "./README.md"
  }'
```

4. 执行curl发送指令

```bash
curl -X 'POST' \
  'http://localhost:8080/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
    "input": "ChatGLM-6B的具体局限性？"
    "use_web": true, 
    "top_k": 3,  
    "history_len": 1, 
    "temperature": 0.01, 
    "top_p": 0.1, 
    "history": []
  }'
```

5. Docker API 服务快速启动

```bash
docker run -d --name LangChain-ChatGLM-Webui --runtime=nvidia --gpus all --network host registry.cn-beijing.aliyuncs.com/public-development-resources/langchain-chatglm-webui:api
```
