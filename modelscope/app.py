import os

import nltk

nltk.download('averaged_perceptron_tagger')
import gradio as gr
import sentence_transformers
import torch
from langchain.chains import ChatVectorDBChain
from langchain.document_loaders import UnstructuredFileLoader
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.prompts.chat import (ChatPromptTemplate,
                                    HumanMessagePromptTemplate,
                                    SystemMessagePromptTemplate)
from langchain.prompts.prompt import PromptTemplate
from langchain.vectorstores import FAISS

from chatglm_llm import ChatGLM

DEVICE = "cuda" if torch.cuda.is_available(
) else "mps" if torch.backends.mps.is_available() else "cpu"

embedding_model_dict = {
    "ernie-tiny": "nghuyong/ernie-3.0-nano-zh",
    "ernie-base": "nghuyong/ernie-3.0-base-zh",
    "text2vec": "GanymedeNil/text2vec-large-chinese"
}


def init_knowledge_vector_store(embedding_model, filepath):
    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model_dict[embedding_model], )
    loader = UnstructuredFileLoader(filepath, mode="elements")
    docs = loader.load()
    vector_store = FAISS.from_documents(docs, embeddings)
    return vector_store


def get_knowledge_based_answer(query, vector_store, chat_history=[]):
    print('get_knowledge_based_answer')
    system_template = """基于以下内容，简洁和专业的来回答用户的问题。
    如果无法从中得到答案，请说 "不知道" 或 "没有足够的相关信息"，不要试图编造答案。答案请使用中文。
    ----------------
    {context}
    ----------------
    """
    messages = [
        SystemMessagePromptTemplate.from_template(system_template),
        HumanMessagePromptTemplate.from_template("{question}"),
    ]
    prompt = ChatPromptTemplate.from_messages(messages)

    condese_propmt_template = """任务: 给一段对话和一个后续问题，将后续问题改写成一个独立的问题。确保问题是完整的，没有模糊的指代。
    ----------------
    聊天记录：
    {chat_history}
    ----------------
    后续问题：{question}
    ----------------
    改写后的独立、完整的问题："""
    new_question_prompt = PromptTemplate.from_template(condese_propmt_template)
    chatLLM = ChatGLM()
    knowledge_chain = ChatVectorDBChain.from_llm(
        llm=chatLLM,
        vectorstore=vector_store,
        qa_prompt=prompt,
        condense_question_prompt=new_question_prompt,
    )

    knowledge_chain.return_source_documents = True
    knowledge_chain.top_k_docs_for_context = 10

    result = knowledge_chain({"question": query, "chat_history": []})
    print(result)
    return result


def clear_session():
    return '', None


def predict(input, embedding_model, file_obj, history=None):

    if history == None:
        history = []
    print(file_obj.name)
    vector_store = init_knowledge_vector_store(embedding_model, file_obj.name)

    resp = get_knowledge_based_answer(query=input, vector_store=vector_store)
    history.append((input, resp['answer']))
    return '', history, history


if __name__ == "__main__":
    block = gr.Blocks()
    with block as demo:
        gr.Markdown("""<h1><center>LangChain-ChatLLM-Webui</center></h1>
        <center><font size=3>
        本项目基于LangChain和大型语言模型系列模型, 提供基于本地知识的自动问答应用. <br>
        目前项目提供基于<a href='https://github.com/THUDM/ChatGLM-6B' target="_blank">ChatGLM-6B </a>的LLM和包括GanymedeNil/text2vec-large-chinese、nghuyong/ernie-3.0-base-zh、nghuyong/ernie-3.0-nano-zh在内的多个Embedding模型, 支持上传 txt、docx、md 等文本格式文件. <br>
        后续将提供更加多样化的LLM、Embedding和参数选项供用户尝试, 欢迎关注<a href='https://github.com/thomas-yanxin/LangChain-ChatGLM-Webui' target="_blank">Github地址</a>.
        </center></font>
        """)
        with gr.Row():
            with gr.Column(scale=4):
                chatbot = gr.Chatbot(label='ChatLLM').style(height=300)
                message = gr.Textbox(label='请输入问题')
                state = gr.State()

            with gr.Column(scale=1):
                embedding_model = gr.Dropdown(
                    ["ernie-tiny", "ernie-base", "text2vec"],
                    label="Embedding model",
                    value="ernie-tiny")
                file = gr.File(label='请上传知识库文件',
                               file_types=['.txt', '.md', '.docx'])
                clear_history = gr.Button("🧹 清除历史对话")
                send = gr.Button("🚀 发送")

                send.click(predict,
                           inputs=[message, embedding_model, file, state],
                           outputs=[message, chatbot, state])
                clear_history.click(fn=clear_session,
                                    inputs=[],
                                    outputs=[chatbot, state],
                                    queue=False)

        message.submit(predict,
                       inputs=[message, embedding_model, file, state],
                       outputs=[message, chatbot, state])
    demo.queue().launch(share=False)
