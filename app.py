import os
from typing import List

import gradio as gr
import nltk
import sentence_transformers
from duckduckgo_search import ddg
from duckduckgo_search.utils import SESSION
from langchain.chains import RetrievalQA
from langchain.document_loaders import UnstructuredFileLoader
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.prompts.prompt import PromptTemplate
from langchain.vectorstores import FAISS

from chatllm import ChatLLM
from chinese_text_splitter import ChineseTextSplitter
from config import *

nltk.data.path = [os.path.join(os.path.dirname(__file__), "nltk_data")
                  ] + nltk.data.path

embedding_model_dict = embedding_model_dict
llm_model_dict = llm_model_dict
EMBEDDING_DEVICE = EMBEDDING_DEVICE
LLM_DEVICE = LLM_DEVICE
num_gpus = num_gpus
init_llm = init_llm
init_embedding_model = init_embedding_model

llm_model_list = []
llm_model_dict = llm_model_dict

for i in llm_model_dict:
    for j in llm_model_dict[i]:
        llm_model_list.append(j)


def search_web(query):

    SESSION.proxies = {
        "http": f"socks5h://localhost:7890",
        "https": f"socks5h://localhost:7890"
    }
    results = ddg(query)
    web_content = ''
    if results:
        for result in results:
            web_content += result['body']
    return web_content


class KnowledgeBasedChatLLM:

    llm: object = None
    embeddings: object = None

    def init_model_config(
        self,
        large_language_model: str = init_llm,
        embedding_model: str = init_embedding_model,
    ):

        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model_dict[embedding_model], )
        self.embeddings.client = sentence_transformers.SentenceTransformer(
            self.embeddings.model_name,
            device=EMBEDDING_DEVICE,
            cache_folder=os.path.join(MODEL_CACHE_PATH,
                                      self.embeddings.model_name))
        self.llm = ChatLLM()
        if 'chatglm' in large_language_model.lower():
            self.llm.model_type = 'chatglm'
            self.llm.model_name_or_path = llm_model_dict['chatglm'][
                large_language_model]
        elif 'belle' in large_language_model.lower():
            self.llm.model_type = 'belle'
            self.llm.model_name_or_path = llm_model_dict['belle'][
                large_language_model]
        elif 'vicuna' in large_language_model.lower():
            self.llm.model_type = 'vicuna'
            self.llm.model_name_or_path = llm_model_dict['vicuna'][
                large_language_model]
        self.llm.load_llm(llm_device=LLM_DEVICE, num_gpus=num_gpus)

    def init_knowledge_vector_store(self, filepath):

        docs = self.load_file(filepath)

        vector_store = FAISS.from_documents(docs, self.embeddings)
        vector_store.save_local('faiss_index')
        return vector_store

    def get_knowledge_based_answer(self,
                                   query,
                                   web_content,
                                   top_k: int = 6,
                                   history_len: int = 3,
                                   temperature: float = 0.01,
                                   top_p: float = 0.1,
                                   history=[]):
        self.llm.temperature = temperature
        self.llm.top_p = top_p
        self.history_len = history_len
        self.top_k = top_k
        if web_content:
            prompt_template = f"""基于以下已知信息，简洁和专业的来回答用户的问题。
                                如果无法从中得到答案，请说 "根据已知信息无法回答该问题" 或 "没有提供足够的相关信息"，不允许在答案中添加编造成分，答案请使用中文。
                                已知网络检索内容：{web_content}""" + """
                                已知内容:
                                {context}
                                问题:
                                {question}"""
        else:
            prompt_template = """基于以下已知信息，请简洁并专业地回答用户的问题。
                如果无法从中得到答案，请说 "根据已知信息无法回答该问题" 或 "没有提供足够的相关信息"。不允许在答案中添加编造成分。另外，答案请使用中文。

                已知内容:
                {context}

                问题:
                {question}"""
        prompt = PromptTemplate(template=prompt_template,
                                input_variables=["context", "question"])
        self.llm.history = history[
            -self.history_len:] if self.history_len > 0 else []
        vector_store = FAISS.load_local('faiss_index', self.embeddings)

        knowledge_chain = RetrievalQA.from_llm(
            llm=self.llm,
            retriever=vector_store.as_retriever(
                search_kwargs={"k": self.top_k}),
            prompt=prompt)
        knowledge_chain.combine_documents_chain.document_prompt = PromptTemplate(
            input_variables=["page_content"], template="{page_content}")

        knowledge_chain.return_source_documents = True

        result = knowledge_chain({"query": query})
        return result

    def load_file(self, filepath):
        if filepath.lower().endswith(".md"):
            loader = UnstructuredFileLoader(filepath, mode="elements")
            docs = loader.load()
        elif filepath.lower().endswith(".pdf"):
            loader = UnstructuredFileLoader(filepath)
            textsplitter = ChineseTextSplitter(pdf=True)
            docs = loader.load_and_split(textsplitter)
        else:
            loader = UnstructuredFileLoader(filepath, mode="elements")
            textsplitter = ChineseTextSplitter(pdf=False)
            docs = loader.load_and_split(text_splitter=textsplitter)
        return docs


def update_status(history, status):
    history = history + [[None, status]]
    print(status)
    return history


knowladge_based_chat_llm = KnowledgeBasedChatLLM()


def init_model():
    try:
        knowladge_based_chat_llm.init_model_config()
        knowladge_based_chat_llm.llm._call("你好")
        return """初始模型已成功加载，可以开始对话"""
    except Exception as e:

        return """模型未成功加载，请重新选择模型后点击"重新加载模型"按钮"""


def clear_session():
    return '', None


def reinit_model(large_language_model, embedding_model, history):
    try:
        knowladge_based_chat_llm.init_model_config(
            large_language_model=large_language_model,
            embedding_model=embedding_model)
        model_status = """模型已成功重新加载，可以开始对话"""
    except Exception as e:

        model_status = """模型未成功重新加载，请点击重新加载模型"""
    return history + [[None, model_status]]


def init_vector_store(file_obj):

    vector_store = knowladge_based_chat_llm.init_knowledge_vector_store(
        file_obj.name)

    return vector_store


def predict(input,
            use_web,
            top_k,
            history_len,
            temperature,
            top_p,
            history=None):
    if history == None:
        history = []

    if use_web == 'True':
        web_content = search_web(query=input)
    else:
        web_content = ''

    resp = knowladge_based_chat_llm.get_knowledge_based_answer(
        query=input,
        web_content=web_content,
        top_k=top_k,
        history_len=history_len,
        temperature=temperature,
        top_p=top_p,
        history=history)
    history.append((input, resp['result']))
    return '', history, history


model_status = init_model()

if __name__ == "__main__":
    block = gr.Blocks()
    with block as demo:

        gr.Markdown("""<h1><center>LangChain-ChatLLM-Webui</center></h1>
        <center><font size=3>
        本项目基于LangChain和大型语言模型系列模型, 提供基于本地知识的自动问答应用. <br>
        目前项目提供基于<a href='https://github.com/THUDM/ChatGLM-6B' target="_blank">ChatGLM-6B </a>的LLM和包括GanymedeNil/text2vec-large-chinese、nghuyong/ernie-3.0-base-zh、nghuyong/ernie-3.0-nano-zh在内的多个Embedding模型, 支持上传 txt、docx、md、pdf等文本格式文件. <br>
        后续将提供更加多样化的LLM、Embedding和参数选项供用户尝试, 欢迎关注<a href='https://github.com/thomas-yanxin/LangChain-ChatGLM-Webui' target="_blank">Github地址</a>.
        </center></font>
        """)
        model_status = gr.State(model_status)
        with gr.Row():
            with gr.Column(scale=1):
                model_choose = gr.Accordion("模型选择")
                with model_choose:
                    large_language_model = gr.Dropdown(
                        llm_model_list,
                        label="large language model",
                        value=init_llm)

                    embedding_model = gr.Dropdown(list(
                        embedding_model_dict.keys()),
                                                  label="Embedding model",
                                                  value=init_embedding_model)
                    load_model_button = gr.Button("重新加载模型")
                model_argument = gr.Accordion("模型参数配置")
                with model_argument:

                    top_k = gr.Slider(1,
                                      10,
                                      value=6,
                                      step=1,
                                      label="vector search top k",
                                      interactive=True)

                    history_len = gr.Slider(0,
                                            5,
                                            value=3,
                                            step=1,
                                            label="history len",
                                            interactive=True)

                    temperature = gr.Slider(0,
                                            1,
                                            value=0.01,
                                            step=0.01,
                                            label="temperature",
                                            interactive=True)
                    top_p = gr.Slider(0,
                                      1,
                                      value=0.9,
                                      step=0.1,
                                      label="top_p",
                                      interactive=True)

                file = gr.File(label='请上传知识库文件',
                               file_types=['.txt', '.md', '.docx', '.pdf'])

                init_vs = gr.Button("知识库文件向量化")

                use_web = gr.Radio(["True", "False"],
                                   label="Web Search",
                                   value="False")

            with gr.Column(scale=4):
                chatbot = gr.Chatbot([[None, model_status.value]],
                                     label='ChatLLM').style(height=750)
                message = gr.Textbox(label='请输入问题')
                state = gr.State()

                with gr.Row():
                    clear_history = gr.Button("🧹 清除历史对话")
                    send = gr.Button("🚀 发送")

            load_model_button.click(
                reinit_model,
                show_progress=True,
                inputs=[large_language_model, embedding_model, chatbot],
                outputs=chatbot,
            )
            init_vs.click(
                init_vector_store,
                show_progress=True,
                inputs=[file],
                outputs=[],
            )

            send.click(predict,
                       inputs=[
                           message, use_web, top_k, history_len, temperature,
                           top_p, state
                       ],
                       outputs=[message, chatbot, state])
            clear_history.click(fn=clear_session,
                                inputs=[],
                                outputs=[chatbot, state],
                                queue=False)

            message.submit(predict,
                           inputs=[
                               message, use_web, top_k, history_len,
                               temperature, top_p, state
                           ],
                           outputs=[message, chatbot, state])
        gr.Markdown("""提醒：<br>
        1. 使用时请先上传自己的知识文件，并且文件中不含某些特殊字符，否则将返回error. <br>
        2. 有任何使用问题，请通过[Github Issue区](https://github.com/thomas-yanxin/LangChain-ChatGLM-Webui/issues)进行反馈. <br>
        """)
    # threads to consume the request
    demo.queue(concurrency_count=3) \
        .launch(server_name='0.0.0.0', # ip for listening, 0.0.0.0 for every inbound traffic, 127.0.0.1 for local inbound
                server_port=7860, # the port for listening
                show_api=False, # if display the api document
                share=False, # if register a public url
                inbrowser=False) # if browser would be open automatically
