import os

import gradio as gr
import nltk
from langchain.chains import RetrievalQA
from langchain.document_loaders import UnstructuredFileLoader
from langchain.prompts import PromptTemplate
from langchain.prompts.prompt import PromptTemplate
from langchain.vectorstores import FAISS
from paddle_embedding import PaddleNLPEmbeddings

from chatllm import ChatLLM

nltk.data.path.append('../nltk_data')
llm_model_dict = {'ChatGLM-6B': 'THUDM/chatglm-6b'}

embedding_model_dict = {
    "rocketqa-zh-base-query": "rocketqa-zh-base-query-encoder",
    "rocketqa-zh-dureader": "rocketqa-zh-dureader-query-encoder",
    "rocketqa-zh-dureader-query": "rocketqa-zh-dureader-query-encoder",
    "rocketqa-zh-medium-query": "rocketqa-zh-medium-query-encoder",
    "rocketqa-zh-medium-para": "rocketqa-zh-medium-para-encoder"
}


def init_knowledge_vector_store(embedding_model, filepath):

    embeddings = PaddleNLPEmbeddings(
        model=embedding_model_dict[embedding_model])

    loader = UnstructuredFileLoader(filepath, mode="elements")
    docs = loader.load()

    vector_store = FAISS.from_documents(docs, embeddings)

    return vector_store


def get_knowledge_based_answer(query,
                               large_language_model,
                               vector_store,
                               VECTOR_SEARCH_TOP_K,
                               chat_history=[]):

    prompt_template = """基于以下已知信息，请简洁并专业地回答用户的问题。
        如果无法从中得到答案，请说 "根据已知信息无法回答该问题" 或 "没有提供足够的相关信息"。不允许在答案中添加编造成分。另外，答案请使用中文。

        已知内容:
        {context}

        问题:
        {question}"""
    prompt = PromptTemplate(template=prompt_template,
                            input_variables=["context", "question"])
    chatLLM = ChatLLM()

    knowledge_chain = RetrievalQA.from_llm(
        llm=chatLLM,
        retriever=vector_store.as_retriever(
            search_kwargs={"k": VECTOR_SEARCH_TOP_K}),
        prompt=prompt)
    knowledge_chain.combine_documents_chain.document_prompt = PromptTemplate(
        input_variables=["page_content"], template="{page_content}")

    knowledge_chain.return_source_documents = True

    result = knowledge_chain({"query": query})

    return result['result']


def clear_session():
    return '', None


def predict(input: str,
            large_language_model: str,
            embedding_model: str,
            file_obj,
            VECTOR_SEARCH_TOP_K: int,
            history=None):
    if history == None:
        history = []

    vector_store = init_knowledge_vector_store(embedding_model, file_obj.name)

    resp = get_knowledge_based_answer(
        query=input,
        large_language_model=large_language_model,
        vector_store=vector_store,
        VECTOR_SEARCH_TOP_K=VECTOR_SEARCH_TOP_K,
        chat_history=history,
    )

    history.append((input, resp['result']))
    return '', history, history


if __name__ == "__main__":

    vector_store = init_knowledge_vector_store(
        'rocketqa-zh-dureader', '/home/aistudio/docs/README-ChatGLM.md')

# for i, (old_query, response) in enumerate(history):
#                 prompt += "[Round {}]\n问：{}\n答：{}\n".format(i, old_query, response)
#             prompt += "[Round {}]\n问：{}\n答：".format(len(history), query)

    while True:
        query = input('提问: ')
        resp = get_knowledge_based_answer(query=query,
                                          large_language_model='ChatGLM-6B',
                                          vector_store=vector_store,
                                          VECTOR_SEARCH_TOP_K=10,
                                          chat_history=[])

        print("回答: " + resp)

    

