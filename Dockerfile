FROM registry.cn-beijing.aliyuncs.com/public-development-resources/langchain-chatglm-webui:Base
RUN mkdir -p /pretrainmodel/belle /pretrainmodel/vicuna /pretrainmodel/chatglm
RUN git clone https://github.com/thomas-yanxin/LangChain-ChatGLM-Webui.git /code
WORKDIR /code
RUN pip install -r requirements.txt
CMD ["python3", "app.py"]