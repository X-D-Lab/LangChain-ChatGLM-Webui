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
llm_model_dict = {
    'ChatGLM-6B': 'THUDM/chatglm-6b'
}

embedding_model_dict = {
    "rocketqa-zh-base-query": "rocketqa-zh-base-query-encoder",
    "rocketqa-zh-dureader": "rocketqa-zh-dureader-query-encoder",
    "rocketqa-zh-dureader-query": "rocketqa-zh-dureader-query-encoder",
    "rocketqa-zh-medium-query": "rocketqa-zh-medium-query-encoder",
    "rocketqa-zh-medium-para": "rocketqa-zh-medium-para-encoder"
    
}


def init_knowledge_vector_store(embedding_model, filepath):

    embeddings = PaddleNLPEmbeddings(
        model = embedding_model_dict[embedding_model])
    

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
    print('result')
    print(result)
    return result


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
    print(file_obj.name)
    vector_store = init_knowledge_vector_store(embedding_model, file_obj.name)

    resp = get_knowledge_based_answer(
        query=input,
        large_language_model=large_language_model,
        vector_store=vector_store,
        VECTOR_SEARCH_TOP_K=VECTOR_SEARCH_TOP_K,
        chat_history=history,
    )
    print(resp['result'])
    history.append((input, resp['result']))
    return '', history, history


if __name__ == "__main__":
    block = gr.Blocks()
    with block as demo:
        gr.Markdown("""<h1><center>LangChain-ChatLLM-Webui</center></h1>
        <center><font size=3>
        本项目基于LangChain和大型语言模型系列模型, 提供基于本地知识的自动问答应用. <br>
        目前项目提供基于<a href='https://github.com/THUDM/ChatGLM-6B' target="_blank">ChatGLM-6B </a>的LLM和包括rocketqa-zh系列的多个Embedding模型, 支持上传 txt、docx、md等文本格式文件. <br>
        后续将提供更加多样化的LLM、Embedding和参数选项供用户尝试, 欢迎关注<a href='https://github.com/thomas-yanxin/LangChain-ChatGLM-Webui' target="_blank">Github地址</a>. <br>
        </center></font>
        """)
        with gr.Row():
            with gr.Column(scale=1):
                model_choose = gr.Accordion("模型选择")
                with model_choose:
                    large_language_model = gr.Dropdown(
                        list(llm_model_dict.keys()),
                        label="large language model",
                        value="ChatGLM-6B")
                    embedding_model = gr.Dropdown(list(
                        embedding_model_dict.keys()),
                                                  label="Embedding model",
                                                  value="rocketqa-zh-dureader-query")

                VECTOR_SEARCH_TOP_K = gr.Slider(
                    1,
                    10,
                    value=6,
                    step=1,
                    label="vector search top k",
                    interactive=True)
                file = gr.File(label='请上传知识库文件, 目前支持txt、docx、md格式',
                               file_types=['.txt', '.md', '.docx'])

            with gr.Column(scale=4):
                chatbot = gr.Chatbot(label='ChatLLM')
                message = gr.Textbox(label='请输入问题')
                state = gr.State()

                with gr.Row():
                    clear_history = gr.Button("🧹 清除历史对话")
                    send = gr.Button("🚀 发送")

                    send.click(predict,
                               inputs=[
                                   message, large_language_model,
                                   embedding_model, file, VECTOR_SEARCH_TOP_K,
                                   state
                               ],
                               outputs=[message, chatbot, state])
                    clear_history.click(fn=clear_session,
                                        inputs=[],
                                        outputs=[chatbot, state],
                                        queue=False)

                    message.submit(predict,
                                   inputs=[
                                       message, large_language_model,
                                       embedding_model, file,
                                       VECTOR_SEARCH_TOP_K, state
                                   ],
                                   outputs=[message, chatbot, state])
        gr.Markdown("""提醒：<br>
        1. 使用时请先上传自己的知识文件，并且文件中不含某些特殊字符，否则将返回error. <br>
        2. 有任何使用问题，请通过[问题交流区](https://huggingface.co/spaces/thomas-yanxin/LangChain-ChatLLM/discussions)或[Github Issue区](https://github.com/thomas-yanxin/LangChain-ChatGLM-Webui/issues)进行反馈. <br>
        """)
    demo.queue().launch(server_name='0.0.0.0', share=True)
