import modal
import os
from dotenv import load_dotenv
load_dotenv(".env")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
MODEL_ID       = "gemini-2.5-flash"
PYTHON_VERSION = "3.11"
TIMEOUT        = 600

image = (
    modal.Image.debian_slim(python_version=PYTHON_VERSION)
    .pip_install(
        "google-generativeai",
        "python-dotenv",
    )
    .add_local_dir("src", remote_path="/root/project/src")
)

app = modal.App("3yno-chatbot", image=image)


@app.cls(
    timeout = TIMEOUT,
    secrets = [modal.Secret.from_dict({"GEMINI_API_KEY": GEMINI_API_KEY})],
)
class ChatbotModal:

    @modal.enter()
    def load_model(self):
        import sys
        sys.path.insert(0, "/root/project")

        from src.chatbot import Chatbot3YNO
        self.bot = Chatbot3YNO()
        self.bot.load_model(
            gemini_api_key = os.environ["GEMINI_API_KEY"],
            model_id       = MODEL_ID,
        )

    @modal.method()
    def chat(self, message: str, reset: bool = False) -> str:
        import sys
        sys.path.insert(0, "/root/project")

        if reset:
            self.bot.reset()
        return self.bot.send_message(message)


@app.local_entrypoint()
def main(message: str = "What is dyslexia?", reset: bool = False):
    reply = ChatbotModal().chat.remote(message=message, reset=reset)
    print(f"\n3YNO: {reply}\n")
