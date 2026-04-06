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
        "fastapi[standard]",
        "pydantic",
        "python-dotenv",
    )
    .add_local_dir("src", remote_path="/root/project/src")
)

app = modal.App("3yno-chatbot-api", image=image)


@app.cls(
    timeout          = TIMEOUT,
    scaledown_window = 300,
    secrets          = [modal.Secret.from_dict({"GEMINI_API_KEY": GEMINI_API_KEY})],
)
class ChatbotAPI:

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
        print("3YNO Chatbot API ready.", flush=True)

    @modal.fastapi_endpoint(method="POST", docs=True)
    def chat(self, request: dict):
        """
        Send a message to 3YNO chatbot.

        Request body:
          {
            "message": "What is dyslexia?",
            "reset": false        ← optional, resets conversation history
          }

        Response:
          {
            "reply": "...",
            "history": [...]
          }
        """
        import sys
        sys.path.insert(0, "/root/project")

        message = request.get("message", "").strip()
        reset   = request.get("reset", False)

        if not message:
            return {"error": "message is required"}

        if reset:
            self.bot.reset()

        reply = self.bot.send_message(message)

        return {
            "reply":   reply,
            "history": self.bot.get_history(),
        }

    @modal.fastapi_endpoint(method="POST", docs=True)
    def reset_conversation(self, request: dict = {}):
        """Reset the conversation history."""
        import sys
        sys.path.insert(0, "/root/project")

        self.bot.reset()
        return {"status": "conversation reset"}
