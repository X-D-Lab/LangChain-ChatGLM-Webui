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

embedding_model_dict = {
    "ernie-tiny": "nghuyong/ernie-3.0-nano-zh",
    "ernie-base": "nghuyong/ernie-3.0-base-zh",
    "text2vec": "GanymedeNil/text2vec-large-chinese"
}

llm_model_dict = {
    "chatglm-6b": "THUDM/chatglm-6b",
    "chatglm-6b-int4": "THUDM/chatglm-6b-int4"
}


def init_knowledge_vector_store(embedding_model, filepath):
    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model_dict[embedding_model], )
    loader = UnstructuredFileLoader(filepath, mode="elements")
    docs = loader.load()

    vector_store = FAISS.from_documents(docs, embeddings)
    return vector_store


def get_knowledge_based_answer(llm, query, vector_store, chat_history=[]):
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
    chatglm.load_model(model_name_or_path=llm_model_dict[llm])
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
    return '', None


def predict(input, llm, embedding_model, file_obj, history=None):
    if history == None:
        history = []
    print(file_obj.name)
    vector_store = init_knowledge_vector_store(embedding_model, file_obj.name)

    resp = get_knowledge_based_answer(llm,
                                      query=input,
                                      vector_store=vector_store)
    history.append((input, resp['answer']))
    return '', history, history


if __name__ == "__main__":
    block = gr.Blocks()
    with block as demo:
        gr.Markdown("""<h1><center>LangChain-ChatGLM-Webui</center></h1>
        <center><font size=3><a href='https://github.com/THUDM/ChatGLM-6B' target="_blank">ChatGLM-6B </a>是一个开源的、支持中英双语的对话语言模型，基于 General Language Model (GLM) 架构，具有 62 亿参数. <br>
        本项目利用LangChain和ChatGLM-6B系列模型制作Webui, 提供基于本地知识的大模型应用. <br>
        目前支持上传 txt、docx、md 等文本格式文件.
        </center></font>
        """)
        with gr.Row():
            with gr.Column(scale=4):
                chatbot = gr.Chatbot(label='ChatGLM-6B')
                message = gr.Textbox(label = '请输入问题')

            with gr.Column(scale=1):
                llm = gr.Dropdown(["chatglm-6b", "chatglm-6b-int4"],
                                label="ChatGLM-6B", value="chatglm-6b-int4")
                embedding_model = gr.Dropdown(["ernie-tiny", "ernie-base", "text2vec"],
                                            label="Embedding model", value = "ernie-tiny")
                file = gr.File(label = '请上传知识库文件')
        
        state = gr.State()
        
        message.submit(predict,
                    inputs=[message, llm, embedding_model, file, state],
                    outputs=[message, chatbot, state])
        with gr.Row():
            clear_history = gr.Button("🧹 清除历史对话")
            send = gr.Button("🚀 发送")

            send.click(predict,
                       inputs=[message, llm, embedding_model, file, state],
                       outputs=[message, chatbot, state])
            clear_history.click(fn=clear_session,
                                inputs=[],
                                outputs=[chatbot, state],
                                queue=False)

    demo.queue().launch(share=True)