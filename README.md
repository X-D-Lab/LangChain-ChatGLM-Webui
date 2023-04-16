 # LangChain-ChatGLM-Webui
 <p align="center">
  <a href="https://github.com/thomas-yanxin/LangChain-ChatGLM-Webui"><img src="https://img.shields.io/badge/GitHub-24292e" alt="github"></a>
  <a href="https://modelscope.cn/studios/AI-ModelScope/LangChain-ChatLLM/summary"><img src="https://img.shields.io/badge/ModelScope-blueviolet" alt="gitee"></a>
  <a href="https://openi.pcl.ac.cn/Learning-Develop-Union/LangChain-ChatGLM-Webui"><img src="https://img.shields.io/badge/-OpenI-337AFF" alt="OpenI"></a>
   <a href="https://www.bilibili.com/video/BV1So4y1L7Hb/?share_source=copy_web&vd_source=8162f92b2a1a94035ca9e4e0f6e1860a"><img src="https://img.shields.io/badge/-bilibili-ff69b4" alt="bilibili"></a>
</p>

## 🔥项目体验

本项目提供基于ModelScope魔搭社区的[在线体验](https://modelscope.cn/studios/AI-ModelScope/LangChain-ChatLLM/summary), 欢迎尝试和反馈!

## 👏项目介绍

受[langchain-ChatGLM](https://github.com/imClumsyPanda/langchain-ChatGLM)启发, 利用LangChain和ChatGLM-6B系列模型制作的Webui, 提供基于本地知识的大模型应用.

目前支持上传 txt、docx、md 等文本格式文件, 提供包括ChatGLM-6B系列的模型文件以及[GanymedeNil/text2vec-large-chinese](https://huggingface.co/GanymedeNil/text2vec-large-chinese)、[nghuyong/ernie-3.0-base-zh](https://huggingface.co/nghuyong/ernie-3.0-base-zh)、[nghuyong/ernie-3.0-nano-zh](https://huggingface.co/nghuyong/ernie-3.0-nano-zh)的Embedding模型.

效果如下:
![](./img/demo_new.jpg)
![](./img/demo_ms.jpg)

## 🚀使用方式

提供ModelScope版本和HuggingFace版本. 其中ModelScope版本下载更稳定、效果更优.  
**需要Python>=3.8.1**

1. git clone本项目: `git clone https://github.com/thomas-yanxin/LangChain-ChatGLM-Webui.git`
2. 进入本项目目录：`cd LangChain-ChatGLM-Webui`
3. 安装依赖包：`pip3 install -r requirements.txt`
4. 执行app.py：`python3 app.py`

### 模型下载

若存在网络问题可点击以下链接快速下载:   
| large language model | Embedding model |
| :----: | :----: |
| [ChatGLM-6B](https://s3.openi.org.cn/opendata/attachment/b/3/b33c55bb-8e7c-4e9d-90e5-c310dcc776d9?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=1fa9e58b6899afd26dd3%2F20230416%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20230416T025911Z&X-Amz-Expires=604800&X-Amz-SignedHeaders=host&response-content-disposition=attachment%3B%20filename%3D%22chatglm-6b.zip%22&X-Amz-Signature=89de83c6dae3702387d14078845b3728a6b09e5e84fc57dbe66c1566f43482a7) | [text2vec-large-chinese](https://s3.openi.org.cn/opendata/attachment/a/2/a2f0edca-1b7b-4dfc-b7c8-15730d33cc3e?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=1fa9e58b6899afd26dd3%2F20230416%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20230416T044328Z&X-Amz-Expires=604800&X-Amz-SignedHeaders=host&response-content-disposition=attachment%3B%20filename%3D%22text2vec-large-chinese.zip%22&X-Amz-Signature=7468efbc7700f652e61386fe0d04b4d36dbd6cb8ff46d4cfd17c0f37bbaf868e) |
| [ChatGLM-6B-int4](https://s3.openi.org.cn/opendata/attachment/b/2/b2c7f23f-6864-40da-9c81-2c0607cb1d02?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=1fa9e58b6899afd26dd3%2F20230415%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20230415T155352Z&X-Amz-Expires=604800&X-Amz-SignedHeaders=host&response-content-disposition=attachment%3B%20filename%3D%22chatglm-6b-int4.zip%22&X-Amz-Signature=0488bd8a55e0b52c846630d609e68d2fa05bd0f0b057059f4f94133a17fbd35b) | [ernie-3.0-nano-zh](https://s3.openi.org.cn/opendata/attachment/2/2/22833889-1683-422e-a44c-929bc379904c?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=1fa9e58b6899afd26dd3%2F20230416%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20230416T044402Z&X-Amz-Expires=604800&X-Amz-SignedHeaders=host&response-content-disposition=attachment%3B%20filename%3D%22ernie-3.0-nano-zh.zip%22&X-Amz-Signature=6599e60b224d0fc05d13dac7a3648f24c2cba0462f39220142cb91923cfdc3c5) |
| [ChatGLM-6B-int8](https://s3.openi.org.cn/opendata/attachment/3/a/3aad10d1-ac8e-48f8-ac5f-cea8b54cf41b?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=1fa9e58b6899afd26dd3%2F20230416%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20230416T032447Z&X-Amz-Expires=604800&X-Amz-SignedHeaders=host&response-content-disposition=attachment%3B%20filename%3D%22chatglm-6b-int8.zip%22&X-Amz-Signature=d58c08158ef8550719f934916fe4b6afe67220a9b84036f660e952c07b8b44f6) | [ernie-3.0-base-zh](https://s3.openi.org.cn/opendata/attachment/7/3/733fe6e4-2c29-46d8-93e8-6be16194a204?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=1fa9e58b6899afd26dd3%2F20230416%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20230416T044454Z&X-Amz-Expires=604800&X-Amz-SignedHeaders=host&response-content-disposition=attachment%3B%20filename%3D%22ernie-3.0-base-zh.zip%22&X-Amz-Signature=554428b51410671dfc5dd6c928cb3e1291b0235abf7e418894bd4d5ac218123e) |

## 💪Todo

* [x] 多个模型选择
* [x] 支持上下文
* [ ] 支持上传多个文本文件
* [x] 提供ModelScope版本
* [ ] 支持用户自定义Embedding模型

## ❤️引用

1. [ChatGLM-6B](https://github.com/THUDM/ChatGLM-6B): ChatGLM-6B: 开源双语对话语言模型
2. [LangChain](https://github.com/hwchase17/langchain): Building applications with LLMs through composability
3. [langchain-ChatGLM](https://github.com/imClumsyPanda/langchain-ChatGLM): 基于本地知识的 ChatGLM 应用实现

## 🙇‍感谢

1. [langchain-ChatGLM](https://github.com/imClumsyPanda/langchain-ChatGLM)提供的基础框架
2. [魔搭ModelScope](https://modelscope.cn/home)提供展示空间
3. [OpenI启智社区](https://openi.pcl.ac.cn/)提供调试算力
