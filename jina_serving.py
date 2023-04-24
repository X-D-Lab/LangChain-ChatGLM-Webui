import os

import gradio as gr
import nltk
import sentence_transformers
import torch
from duckduckgo_search import ddg
from duckduckgo_search.utils import SESSION
from langchain.chains import RetrievalQA
from langchain.document_loaders import UnstructuredFileLoader
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.prompts.prompt import PromptTemplate
from langchain.vectorstores import FAISS
from lcserve import serving

from chatllm import ChatLLM
from chinese_text_splitter import ChineseTextSplitter

nltk.data.path.append('./nltk_data')
embedding_model_dict = {
    "ernie-tiny": "nghuyong/ernie-3.0-nano-zh",
    "ernie-base": "nghuyong/ernie-3.0-base-zh",
    "ernie-medium": "nghuyong/ernie-3.0-medium-zh",
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

EMBEDDING_DEVICE = "cuda" if torch.cuda.is_available(
) else "mps" if torch.backends.mps.is_available() else "cpu"

LLM_DEVICE = "cuda" if torch.cuda.is_available(
) else "mps" if torch.backends.mps.is_available() else "cpu"

num_gpus = torch.cuda.device_count()


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
        large_language_model: str = 'ChatGLM-6B-int8',
        embedding_model: str = 'text2vec-base',
    ):

        self.llm = ChatLLM()
        self.llm.model_name_or_path = llm_model_dict[large_language_model]
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model_dict[embedding_model], )
        self.embeddings.client = sentence_transformers.SentenceTransformer(
            self.embeddings.model_name, device=EMBEDDING_DEVICE)
        self.llm.load_llm(llm_device=LLM_DEVICE, num_gpus=num_gpus)

    def init_knowledge_vector_store(self, filepath):

        docs = self.load_file(filepath)

        vector_store = FAISS.from_documents(docs, self.embeddings)
        return vector_store

    def get_knowledge_based_answer(self,
                                   query,
                                   vector_store,
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
        if filepath.lower().endswith(".pdf"):
            loader = UnstructuredFileLoader(filepath)
            textsplitter = ChineseTextSplitter(pdf=True)
            docs = loader.load_and_split(textsplitter)
        else:
            loader = UnstructuredFileLoader(filepath, mode="elements")
            textsplitter = ChineseTextSplitter(pdf=False)
            docs = loader.load_and_split(text_splitter=textsplitter)
        return docs


knowladge_based_chat_llm = KnowledgeBasedChatLLM()


def init_model():
    try:
        knowladge_based_chat_llm.init_model_config()
        knowladge_based_chat_llm.llm._call("你好")
        return """初始模型已成功加载，可以开始对话"""
    except Exception as e:

        return """模型未成功加载，请重新选择模型后点击"重新加载模型"按钮"""

@serving
def reinit_model(large_language_model: str, 
                 embedding_model: str):
    try:
        knowladge_based_chat_llm.init_model_config(
            large_language_model=large_language_model,
            embedding_model=embedding_model)
        model_status = """模型已成功重新加载，可以开始对话"""
    except Exception as e:

        model_status = """模型未成功重新加载，请点击重新加载模型"""
    return model_status

@serving
def predict(input: str,
            file_path: str,
            use_web: str,
            top_k: int,
            history_len: int,
            temperature: float,
            top_p: float,
            history: list):
    if history == None:
        history = []
    vector_store = knowladge_based_chat_llm.init_knowledge_vector_store(
        file_path)
    if use_web == 'True':
        web_content = search_web(query=input)
    else:
        web_content = ''

    resp = knowladge_based_chat_llm.get_knowledge_based_answer(
        query=input,
        vector_store=vector_store,
        web_content=web_content,
        top_k=top_k,
        history_len=history_len,
        temperature=temperature,
        top_p=top_p,
        history=history)
    history.append((input, resp['result']))
    print(resp['result'])
    return resp['result']

if __name__ == "__main__":
    reinit_model(large_language_model='ChatGLM-6B', 
                 embedding_model='ernie-tiny')
    predict('chatglm-6b的局限性在哪里？',
            file_path='./README.md',
            use_web=True,
            top_k=6,
            history_len = 3,
            temperature = 0.01,
            top_p = 0.1,
            history = [])
