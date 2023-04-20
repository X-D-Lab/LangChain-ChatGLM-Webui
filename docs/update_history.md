# 更新日志

## Apr 20 2023

1. 优化对`.pdf`的支持
2. 优化UI设计
3. 修改对于Docker镜像的描述

## Apr 19 2023

1. 提供视频部署教程
2. 提供Docker部署及文字教程
3. 支持多卡推理ChatGLM-6B
4. 增加外部访问支持
5. 增加ChatGLM-6b-local以及本地模型读取路径
6. 修复text2vec无法加载的错误

上述2-6来自社区[@online2311](https://github.com/online2311)

## Apr 18 2023

1. 更新ModelScope版本
2. 完善Readme描述

## Apr 17 2023

1. 提供部署文档
2. 支持`.pdf`格式
3. 更新Docker镜像地址
4. 完善Readme描述

## Apr 16 2023

1. 提供模型下载链接
2. 完善Prompt设计
3. 修复上下文的bug
4. 支持更多的LLM和Embedding Model
5. 同步更新ModelScope版本
6. 完善Readme描述

## Apr 15 2023

1. 完善Readme描述

## Apr 14 2023

1. 提供离线的`nltk_data`依赖文件, 用户无需再次下载
2. 支持上下文
3. 增加支持的Embedding Model
4. 增加参数`history_len`、`temperature`、`top_p`
5. 同步更新ModelScope版本
6. 完善Readme描述

## Apr 13 2023

1. 提供ModelScope版本，支持在线体验
2. 使用langchain中的`RetrievalQA`替代之前选用的`ChatVectorDBChain`
3. 提供选择参数`VECTOR_SEARCH_TOP_K`
4. 删除提供的错误Docker地址

## Apr 12 2023

1. 增加对Embedding Model推理设备的定义

## Apr 11 2023

1. 提供部署镜像
2. 完善Readme描述

## Apr 10 2023

1. 提供多种LLM和Embedding Model可选择
2. 增加显存清理机制
3. 完善Readme描述

## Apr 10 2023

1. 初始化仓库, 提交Readme和Licence
2. 提交基础关键代码: app.py和chatglm_llm.py
3. 支持ChatGLM-6B的LLM和GanymedeNil/text2vec-large-chinese的Embedding Model
