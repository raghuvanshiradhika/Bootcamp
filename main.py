from openai_chat import OpenAIChat
import gradio as gr

openai_chat = OpenAIChat()


class HomeLine:
    def __init__(self):
        self.openai_chat = OpenAIChat()

    def submit_prompt(self, question):
        print("Please enter")
        return self.openai_chat.submit_prompt(question)


def chat_interface(question):
    home_instance = HomeLine()
    response = home_instance.submit_prompt(question)
    return response


gr.Interface(
    fn=chat_interface,
    inputs="text",
    outputs="text",
    title="Home Line",
    description="Ask any question to Home Line and get a response."
).launch(share=True)
