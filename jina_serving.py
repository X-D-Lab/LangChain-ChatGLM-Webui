import datetime
import os
from typing import List

import nltk
import qdrant_client
import sentence_transformers
import torch
from duckduckgo_search import ddg
from duckduckgo_search.utils import SESSION
from langchain.chains import RetrievalQA
from langchain.document_loaders import UnstructuredFileLoader
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.prompts.prompt import PromptTemplate
from langchain.vectorstores import Qdrant
from lcserve import serving

from chatllm import ChatLLM
from chinese_text_splitter import ChineseTextSplitter
from config import *

nltk.data.path = [os.path.join(os.path.dirname(__file__), "nltk_data")
                  ] + nltk.data.path

embedding_model_dict = embedding_model_dict
llm_model_dict = llm_model_dict
EMBEDDING_DEVICE = EMBEDDING_DEVICE
LLM_DEVICE = LLM_DEVICE
VECTOR_STORE_PATH = VECTOR_STORE_PATH
COLLECTION_NAME = COLLECTION_NAME
num_gpus = num_gpus
init_llm = init_llm
init_embedding_model = init_embedding_model



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
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model_dict[embedding_model], )
        self.embeddings.client = sentence_transformers.SentenceTransformer(
            self.embeddings.model_name, device=EMBEDDING_DEVICE)
        self.llm.load_llm(llm_device=LLM_DEVICE, num_gpus=num_gpus)

    def init_knowledge_vector_store(self,
                                    filepath: str or List[str],):
        loaded_files = []
        if isinstance(filepath, str):
            if not os.path.exists(filepath):
                return "路径不存在"
            elif os.path.isfile(filepath):
                file = os.path.split(filepath)[-1]
                try:
                    docs = self.load_file(filepath)
                    print(f"{file} 已成功加载")
                    loaded_files.append(filepath)
                except Exception as e:
                    print(e)
                    print(f"{file} 未能成功加载")
                    return f"{file} 未能成功加载"
            elif os.path.isdir(filepath):
                docs = []
                for file in os.listdir(filepath):
                    fullfilepath = os.path.join(filepath, file)
                    try:
                        docs += self.load_file(fullfilepath)
                        print(f"{file} 已成功加载")
                        loaded_files.append(fullfilepath)
                    except Exception as e:
                        print(e)
                        print(f"{file} 未能成功加载")
        else:
            docs = []
            for file in filepath:
                try:
                    docs += self.load_file(file)
                    print(f"{file} 已成功加载")
                    loaded_files.append(file)
                except Exception as e:
                    print(e)
                    print(f"{file} 未能成功加载")
        if len(docs) > 0:
            if VECTOR_STORE_PATH and os.path.isdir(VECTOR_STORE_PATH):
                vector_store = Qdrant.from_documents(
                    docs,
                    self.embeddings,
                    path=VECTOR_STORE_PATH,
                    collection_name=COLLECTION_NAME,
                )
                vector_store.add_documents(docs)
            else:
                vector_store = Qdrant.from_documents(
                    docs,
                    self.embeddings,
                    path=VECTOR_STORE_PATH,
                    collection_name=COLLECTION_NAME,
                )
            return "文件均未成功加载，请检查依赖包或文件路径。", loaded_files
        else:
            print("文件均未成功加载，请检查依赖包或文件路径。")
            return "文件均未成功加载，请检查依赖包或文件路径。", loaded_files

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
        client = qdrant_client.QdrantClient(path=VECTOR_STORE_PATH,
                                            prefer_grpc=True)
        qdrant = Qdrant(client=client,
                        collection_name=COLLECTION_NAME,
                        embedding_function=self.embeddings.embed_query)
        knowledge_chain = RetrievalQA.from_llm(
            llm=self.llm,
            retriever=qdrant.as_retriever(search_kwargs={"k": self.top_k}),
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


knowladge_based_chat_llm = KnowledgeBasedChatLLM()


def init_model():
    try:
        knowladge_based_chat_llm.init_model_config()
        knowladge_based_chat_llm.llm._call("你好")
        return """初始模型已成功加载"""
    except Exception as e:

        return """模型未成功加载，请检查后重新尝试"""


@serving
def reinit_model(large_language_model: str, embedding_model: str):
    try:
        knowladge_based_chat_llm.init_model_config(
            large_language_model=large_language_model,
            embedding_model=embedding_model)
        model_status = """模型已成功重新加载"""
    except Exception as e:
        model_status = """模型未成功加载，请检查后重新尝试"""
    return model_status


@serving
def vector_store(file_path: str or List[str]):

    vector_store_state, loaded_files = knowladge_based_chat_llm.init_knowledge_vector_store(
        file_path)
    return vector_store_state


@serving
def predict(input: str, 
            use_web: bool, top_k: int, history_len: int, temperature: float,
            top_p: float, history: list):
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
    print(resp['result'])
    return resp['result']

if __name__ == "__main__":
    reinit_model(large_language_model='ChatGLM-6B-int8',
                 embedding_model='text2vec-base')

    vector_store(file_path='./README.md')

    predict('chatglm-6b的局限性在哪里？',
            use_web=False,
            top_k=6,
            history_len=3,
            temperature=0.01,
            top_p=0.1,
            history=[])
