<img src='./img/bg.jpg'>
 <p align="center">
  <a href="https://github.com/thomas-yanxin/LangChain-ChatGLM-Webui"><img src="https://img.shields.io/badge/GitHub-24292e" alt="github"></a>
  <a href="https://huggingface.co/spaces/thomas-yanxin/LangChain-ChatLLM"><img src="https://img.shields.io/badge/HuggingFace-yellow" alt="HuggingFace"></a>
  <a href="https://modelscope.cn/studios/AI-ModelScope/LangChain-ChatLLM/summary"><img src="https://img.shields.io/badge/ModelScope-blueviolet" alt="modelscope"></a>
  <a href="https://openi.pcl.ac.cn/Learning-Develop-Union/LangChain-ChatGLM-Webui"><img src="https://img.shields.io/badge/-OpenI-337AFF" alt="OpenI"></a>
   <a href="https://www.bilibili.com/video/BV1So4y1L7Hb/?share_source=copy_web&vd_source=8162f92b2a1a94035ca9e4e0f6e1860a"><img src="https://img.shields.io/badge/-bilibili-ff69b4" alt="bilibili"></a> 
</p> 
<p align="center">
<a href="https://github.com/thomas-yanxin/LangChain-ChatGLM-Webui/stargazers"><img alt="GitHub stars" src="https://img.shields.io/github/stars/thomas-yanxin/LangChain-ChatGLM-Webui?color=brightgreen" /></a>
<a href="https://github.com/thomas-yanxin/LangChain-ChatGLM-Webui/graphs/contributors">
<img alt="GitHub Contributors" src="https://img.shields.io/github/contributors/thomas-yanxin/LangChain-ChatGLM-Webui" />
</a>
<a href="https://github.com/thomas-yanxin/LangChain-ChatGLM-Webui/issues"><img alt="Issues" src="https://img.shields.io/github/issues/thomas-yanxin/LangChain-ChatGLM-Webui?color=0088ff" /></a>
<a href="https://github.com/thomas-yanxin/LangChain-ChatGLM-Webui/pulls"><img alt="GitHub pull requests" src="https://img.shields.io/github/issues-pr/thomas-yanxin/LangChain-ChatGLM-Webui?color=orange" /></a>
</p>
<p align="center">  
   <a href="https://www.bilibili.com/video/BV1So4y1L7Hb/?share_source=copy_web&vd_source=8162f92b2a1a94035ca9e4e0f6e1860a"><strong>视频链接</strong></a> | <a href="https://huggingface.co/spaces/thomas-yanxin/LangChain-ChatLLM"><strong>在线体验</strong></a> | <a href="https://github.com/thomas-yanxin/LangChain-ChatGLM-Webui/blob/master/docs/deploy.md"><strong>部署文档</strong></a>| <a href="https://github.com/thomas-yanxin/LangChain-ChatGLM-Webui/blob/master/docs/update_history.md"><strong>更新日志</strong></a> | <a href="https://github.com/thomas-yanxin/LangChain-ChatGLM-Webui/blob/master/docs/qa.md"><strong>常见问题</strong></a> 

</p>

## 🔥 项目体验

本项目提供基于[HuggingFace社区](https://huggingface.co/spaces/thomas-yanxin/LangChain-ChatLLM)和[ModelScope魔搭社区](https://modelscope.cn/studios/AI-ModelScope/LangChain-ChatLLM/summary)的在线体验, 欢迎尝试和反馈!  

## 👏 项目介绍

受[langchain-ChatGLM](https://github.com/imClumsyPanda/langchain-ChatGLM)启发, 利用LangChain和ChatGLM-6B系列模型制作的Webui, 提供基于本地知识的大模型应用.

目前支持上传 txt、docx、md、pdf等文本格式文件, 提供包括ChatGLM-6B系列、Belle系列等模型文件以及[GanymedeNil/text2vec-large-chinese](https://huggingface.co/GanymedeNil/text2vec-large-chinese)、[nghuyong/ernie-3.0-base-zh](https://huggingface.co/nghuyong/ernie-3.0-base-zh)、[nghuyong/ernie-3.0-nano-zh](https://huggingface.co/nghuyong/ernie-3.0-nano-zh)等Embedding模型.

<details><summary><b>HuggingFace效果</b></summary>

![](./img/demo_hf.jpg)

</details>
<details><summary><b>ModelScope效果</b></summary>

![](./img/demo_ms.jpg)

</details>

## 🚀 使用方式

提供ModelScope版本和HuggingFace版本.  
**需要Python>=3.8.1**  

详细部署教程可参考: [部署文档](./docs/deploy.md) | [视频教程](https://www.bilibili.com/video/BV1No4y1b7eu/)

### 支持模型

若存在网络问题可点击以下链接快速下载:   
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

## 💪 更新日志

详情请见: [更新日志](./docs/update_history.md)

项目处于初期阶段, 有很多可以做的地方和优化的空间, 欢迎感兴趣的社区大佬们一起加入!

## ❤️ 引用

1. [ChatGLM-6B](https://github.com/THUDM/ChatGLM-6B): ChatGLM-6B: 开源双语对话语言模型
2. [LangChain](https://github.com/hwchase17/langchain): Building applications with LLMs through composability
3. [langchain-ChatGLM](https://github.com/imClumsyPanda/langchain-ChatGLM): 基于本地知识的 ChatGLM 应用实现
<details><summary><b>ChatGLM论文引用</b></summary>

```
@inproceedings{
  zeng2023glm-130b,
  title={{GLM}-130B: An Open Bilingual Pre-trained Model},
  author={Aohan Zeng and Xiao Liu and Zhengxiao Du and Zihan Wang and Hanyu Lai and Ming Ding and Zhuoyi Yang and Yifan Xu and Wendi Zheng and Xiao Xia and Weng Lam Tam and Zixuan Ma and Yufei Xue and Jidong Zhai and Wenguang Chen and Zhiyuan Liu and Peng Zhang and Yuxiao Dong and Jie Tang},
  booktitle={The Eleventh International Conference on Learning Representations (ICLR)},
  year={2023},
  url={https://openreview.net/forum?id=-Aw0rrrPUF}
}
```

```
@inproceedings{du2022glm,
  title={GLM: General Language Model Pretraining with Autoregressive Blank Infilling},
  author={Du, Zhengxiao and Qian, Yujie and Liu, Xiao and Ding, Ming and Qiu, Jiezhong and Yang, Zhilin and Tang, Jie},
  booktitle={Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  pages={320--335},
  year={2022}
}
```

</details>
<details><summary><b>BELLE论文引用</b></summary>

```
@misc{BELLE,
  author = {Yunjie Ji, Yong Deng, Yan Gong, Yiping Peng, Qiang Niu, Baochang Ma and Xiangang Li},
  title = {BELLE: Be Everyone's Large Language model Engine },
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/LianjiaTech/BELLE}},
}
@article{belle2023exploring,
  title={Exploring the Impact of Instruction Data Scaling on Large Language Models: An Empirical Study on Real-World Use Cases},
  author={Yunjie Ji, Yong Deng, Yan Gong, Yiping Peng, Qiang Niu, Lei Zhang, Baochang Ma, Xiangang Li},
  journal={arXiv preprint arXiv:2303.14742},
  year={2023}
}
```

</details>

## 🙇‍ ‍感谢

1. [langchain-ChatGLM](https://github.com/imClumsyPanda/langchain-ChatGLM)提供的基础框架
2. [魔搭ModelScope](https://modelscope.cn/home)提供展示空间
3. [OpenI启智社区](https://openi.pcl.ac.cn/)提供调试算力
4. [langchain-serve](https://github.com/jina-ai/langchain-serve)提供十分简易的Serving方式

除此以外, 感谢来自社区的同学们对本项目的关注和支持!

## 🌟 Star History

[![Star History Chart](https://api.star-history.com/svg?repos=thomas-yanxin/LangChain-ChatGLM-Webui&type=Date)](https://star-history.com/#thomas-yanxin/LangChain-ChatGLM-Webui&Date)

## 😊 加群沟通

<div> <img src="./img/wechat_group.jpg" width = 50%/> </div>
