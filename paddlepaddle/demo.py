import gradio as gr
import mdtex2html

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

"""Override Chatbot.postprocess"""

def postprocess(self, y):
    if y is None:
        return []
    for i, (message, response) in enumerate(y):
        y[i] = (
            None if message is None else mdtex2html.convert((message)),
            None if response is None else mdtex2html.convert(response),
        )
    return y

gr.Chatbot.postprocess = postprocess


def parse_text(text):
    """copy from https://github.com/GaiZhenbiao/ChuanhuChatGPT/"""
    lines = text.split("\n")
    lines = [line for line in lines if line != ""]
    count = 0
    for i, line in enumerate(lines):
        if "```" in line:
            count += 1
            items = line.split('`')
            if count % 2 == 1:
                lines[i] = f'<pre><code class="language-{items[-1]}">'
            else:
                lines[i] = f'<br></code></pre>'
        else:
            if i > 0:
                if count % 2 == 1:
                    line = line.replace("`", "\`")
                    line = line.replace("<", "&lt;")
                    line = line.replace(">", "&gt;")
                    line = line.replace(" ", "&nbsp;")
                    line = line.replace("*", "&ast;")
                    line = line.replace("_", "&lowbar;")
                    line = line.replace("-", "&#45;")
                    line = line.replace(".", "&#46;")
                    line = line.replace("!", "&#33;")
                    line = line.replace("(", "&#40;")
                    line = line.replace(")", "&#41;")
                    line = line.replace("$", "&#36;")
                lines[i] = "<br>"+line
    text = "".join(lines)
    return text


def predict(input, image_path, chatbot, history):
    if image_path is None:
        return [(input, "图片不能为空。请重新上传图片并重试。")], []
    chatbot.append((parse_text(input), ""))
    inputs =  {"text_input": input, "image_path": image_path,'history': []}
    result = pipe(inputs)
    print(result)
    chatbot[-1] = (parse_text(input), parse_text(result['response']))
    print(chatbot)

    return chatbot, history


def predict_new_image(image_path, chatbot):
    input, history = "描述这张图片。", []
    chatbot.append((parse_text(input), ""))
    print(chatbot)
    inputs =  {"text_input": input, "image_path": image_path,'history': []}
    result = pipe(inputs)
    print(result)
    chatbot[-1] = (parse_text(input), parse_text(result['response']))
    print(chatbot)

    return chatbot, history


def reset_user_input():
    return gr.update(value='')


def reset_state():
    return None, [], []


DESCRIPTION = '''<h1 align="center"><a href="https://github.com/THUDM/VisualGLM-6B">VisualGLM</a></h1>'''
MAINTENANCE_NOTICE = 'Hint 1: If the app report "Something went wrong, connection error out", please turn off your proxy and retry.\nHint 2: If you upload a large size of image like 10MB, it may take some time to upload and process. Please be patient and wait.'
NOTES = 'This app is adapted from <a href="https://github.com/THUDM/VisualGLM-6B">https://github.com/THUDM/VisualGLM-6B</a>. It would be recommended to check out the repo if you want to see the detail of our model and training process.'

if __name__ == '__main__':

    pipe = pipeline(task=Tasks.chat, model='ZhipuAI/visualglm-6b', model_revision='v1.0.3', device='cuda')

    with gr.Blocks(css='style.css') as demo:
        gr.HTML(DESCRIPTION)
        
        with gr.Row():
            with gr.Column(scale=2):
                image_path = gr.Image(type="filepath", label="Image Prompt", value=None).style(height=504)
            with gr.Column(scale=4):
                chatbot = gr.Chatbot().style(height=480)
        with gr.Row():
            with gr.Column(scale=4):
                with gr.Box():
                    with gr.Row():
                        with gr.Column(scale=2):
                            user_input = gr.Textbox(show_label=False, placeholder="Input...", lines=4).style(
                                container=False)
                        with gr.Column(scale=1, min_width=64):
                            submitBtn = gr.Button("Submit", variant="primary")
                            emptyBtn = gr.Button("Clear History")
                    gr.Markdown(MAINTENANCE_NOTICE + '\n' + NOTES)
        history = gr.State([])
        

        submitBtn.click(predict, [user_input, image_path, chatbot, history], [chatbot, history],
                        show_progress=True)
        image_path.upload(predict_new_image, [image_path, chatbot], [chatbot, history],
                        show_progress=True)
        image_path.clear(reset_state, outputs=[image_path, chatbot, history], show_progress=True)
        submitBtn.click(reset_user_input, [], [user_input])
        emptyBtn.click(reset_state, outputs=[image_path, chatbot, history], show_progress=True)

        print(gr.__version__)

        demo.queue().launch(share=False, inbrowser=True, server_name='0.0.0.0', server_port=8080)

