import os

import gradio as gr
import nltk
import torch
from chatglm_llm import ChatGLM
from duckduckgo_search import ddg
from duckduckgo_search.utils import SESSION
from langchain.chains import RetrievalQA
from langchain.document_loaders import UnstructuredFileLoader
from langchain.prompts import PromptTemplate
from langchain.vectorstores import FAISS
from modelscope_hub import ModelScopeEmbeddings

nltk.data.path.append('../nltk_data')

DEVICE = "cuda" if torch.cuda.is_available(
) else "mps" if torch.backends.mps.is_available() else "cpu"

embedding_model_dict = {
    "corom-tiny": "damo/nlp_corom_sentence-embedding_chinese-tiny",
    "corom-tiny-ecom": "damo/nlp_corom_sentence-embedding_chinese-tiny-ecom",
    "corom-base-ecom": "damo/nlp_corom_sentence-embedding_chinese-base-ecom",
    "corom-base": "damo/nlp_corom_sentence-embedding_chinese-base",
}

llm_dict = {
    'ChatGLM-6B': {
        'model_name': 'ZhipuAI/ChatGLM-6B',
        'model_revision': 'v1.0.15',
    },
    'ChatGLM-6B-int8': {
        'model_name': 'thomas/ChatGLM-6B-Int8',
        'model_revision': 'v1.0.3',
    },
    'ChatGLM-6B-int4': {
        'model_name': 'ZhipuAI/ChatGLM-6B-Int4',
        'model_revision': 'v1.0.3',
    }
}


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


def init_knowledge_vector_store(embedding_model, filepath):

    embeddings = ModelScopeEmbeddings(
        model_name=embedding_model_dict[embedding_model], )

    loader = UnstructuredFileLoader(filepath, mode="elements")
    docs = loader.load()

    vector_store = FAISS.from_documents(docs, embeddings)
    return vector_store


def get_knowledge_based_answer(
    query,
    large_language_model,
    vector_store,
    VECTOR_SEARCH_TOP_K,
    web_content,
    chat_history=[],
    history_len=3,
    temperature=0.01,
    top_p=0.9,
):
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

    chatLLM = ChatGLM()
    chatLLM.model_name = llm_dict[large_language_model]['model_name']
    chatLLM.model_revision = llm_dict[large_language_model]['model_revision']

    chatLLM.history = chat_history[-history_len:] if history_len > 0 else []
    chatLLM.temperature = temperature
    chatLLM.top_p = top_p

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


def predict(input,
            large_language_model,
            embedding_model,
            file_obj,
            VECTOR_SEARCH_TOP_K,
            history_len,
            temperature,
            top_p,
            use_web,
            history=None):
    if history == None:
        history = []
    print(file_obj.name)
    if use_web == 'True':
        web_content = search_web(query=input)
    else:
        web_content = ''
    vector_store = init_knowledge_vector_store(embedding_model, file_obj.name)

    resp = get_knowledge_based_answer(
        query=input,
        large_language_model=large_language_model,
        vector_store=vector_store,
        VECTOR_SEARCH_TOP_K=VECTOR_SEARCH_TOP_K,
        web_content=web_content,
        chat_history=history,
        history_len=history_len,
        temperature=temperature,
        top_p=top_p,
    )
    print(resp)
    history.append((input, resp))
    return '', history, history


if __name__ == "__main__":
    block = gr.Blocks()
    with block as demo:
        gr.Markdown("""<h1><center>LangChain-ChatLLM-Webui</center></h1>
        <center><font size=3>
        本项目基于LangChain和大型语言模型系列模型, 提供基于本地知识的自动问答应用. <br>
        目前项目提供基于<a href='https://github.com/THUDM/ChatGLM-6B' target="_blank">ChatGLM-6B </a>的LLM和包括nlp_corom_sentence-embedding系列的多个Embedding模型, 支持上传 txt、docx、md 等文本格式文件. <br>
        后续将提供更加多样化的LLM、Embedding和参数选项供用户尝试, 欢迎关注<a href='https://github.com/thomas-yanxin/LangChain-ChatGLM-Webui' target="_blank">Github地址</a>.
        </center></font>
        """)
        with gr.Row():
            with gr.Column(scale=1):
                model_choose = gr.Accordion("模型选择")
                with model_choose:
                    large_language_model = gr.Dropdown(
                        ["ChatGLM-6B", "ChatGLM-6B-int4", 'ChatGLM-6B-int8'],
                        label="large language model",
                        value="ChatGLM-6B-int8")

                    embedding_model = gr.Dropdown(list(
                        embedding_model_dict.keys()),
                        label="Embedding model",
                        value="corom-tiny")

                file = gr.File(label='请上传知识库文件',
                               file_types=['.txt', '.md', '.docx'])
                use_web = gr.Radio(["True", "False"],
                                   label="Web Search",
                                   value="False")
                model_argument = gr.Accordion("模型参数配置")

                with model_argument:

                    VECTOR_SEARCH_TOP_K = gr.Slider(
                        1,
                        10,
                        value=6,
                        step=1,
                        label="vector search top k",
                        interactive=True)

                    HISTORY_LEN = gr.Slider(0,
                                            3,
                                            value=0,
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

            with gr.Column(scale=4):
                chatbot = gr.Chatbot(label='ChatLLM').style(height=400)
                message = gr.Textbox(label='请输入问题')
                state = gr.State()

                with gr.Row():
                    clear_history = gr.Button("🧹 清除历史对话")
                    send = gr.Button("🚀 发送")

                    send.click(predict,
                               inputs=[
                                   message, large_language_model,
                                   embedding_model, file, VECTOR_SEARCH_TOP_K,
                                   HISTORY_LEN, temperature, top_p, use_web,
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
                                       VECTOR_SEARCH_TOP_K, HISTORY_LEN,
                                       temperature, top_p, use_web, state
                                   ],
                                   outputs=[message, chatbot, state])
        gr.Markdown("""提醒：<br>
        1. 更改LLM模型前请先刷新页面，否则将返回error（后续将完善此部分）. <br>
        2. 使用时请先上传自己的知识文件，并且文件中不含某些特殊字符，否则将返回error. <br>
        3. 请勿上传或输入敏感内容，否则输出内容将被平台拦截返回error.<br>
        4. 有任何使用问题，请通过[问题交流区](https://modelscope.cn/studios/thomas/ChatYuan-test/comment)或[Github Issue区](https://github.com/thomas-yanxin/LangChain-ChatGLM-Webui/issues)进行反馈. <br>
        """)
    demo.queue().launch(share=False)