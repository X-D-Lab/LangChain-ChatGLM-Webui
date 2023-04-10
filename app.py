import gradio as gr
from langchain.chains import ChatVectorDBChain
from langchain.document_loaders import UnstructuredFileLoader
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.prompts.chat import (ChatPromptTemplate,
                                    HumanMessagePromptTemplate,
                                    SystemMessagePromptTemplate)
from langchain.prompts.prompt import PromptTemplate
from langchain.vectorstores import FAISS

from chatglm_llm import ChatGLM


def init_knowledge_vector_store(filepath):
    embeddings = HuggingFaceEmbeddings(
        model_name="GanymedeNil/text2vec-large-chinese", )
    loader = UnstructuredFileLoader(filepath, mode="elements")
    docs = loader.load()

    vector_store = FAISS.from_documents(docs, embeddings)
    return vector_store


def get_knowledge_based_answer(query, vector_store, chat_history=[]):
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
    chatglm = ChatGLM()
    chatglm.history = chat_history
    knowledge_chain = ChatVectorDBChain.from_llm(
        llm=chatglm,
        vectorstore=vector_store,
        qa_prompt=prompt,
        condense_question_prompt=new_question_prompt,
    )

    knowledge_chain.return_source_documents = True
    knowledge_chain.top_k_docs_for_context = 10

    result = knowledge_chain({"question": query, "chat_history": []})
    return result


def clear_session():
    return '',  None


def predict(input, file_obj, history=None):
    if history == None:
        history = []
    print(file_obj.name)
    vector_store = init_knowledge_vector_store(file_obj.name)

    resp = get_knowledge_based_answer(query=input,
                                               vector_store=vector_store)
    history.append((input, resp['answer']))
    return '', history, history



if __name__ == "__main__":
    block = gr.Blocks()
    with block as demo:
        gr.Markdown("""<h1><center>LangChain-ChatGLM-Webui</center></h1>
        <center><font size=3><a href='https://modelscope.cn/models/ZhipuAI/ChatGLM-6B/summary' target="_blank">ChatGLM-6B </a>是一个开源的、支持中英双语的对话语言模型，基于 General Language Model (GLM) 架构，具有 62 亿参数。</center></font>
        """)
        chatbot = gr.Chatbot(label='ChatGLM-6B')
        message = gr.Textbox()
        state = gr.State()
        file = gr.File()
        message.submit(predict,
                       inputs=[message, file, state],
                       outputs=[message, chatbot, state])
        with gr.Row():
            clear_history = gr.Button("🧹 清除历史对话")
            send = gr.Button("🚀 发送")

            send.click(predict,
                       inputs=[message, file, state],
                       outputs=[message, chatbot, state])
            clear_history.click(fn=clear_session,
                                inputs=[],
                                outputs=[chatbot, state],
                                queue=False)

    demo.queue().launch(height=800, share=True)