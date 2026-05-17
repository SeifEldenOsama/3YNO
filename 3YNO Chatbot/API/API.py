import modal
import os
import random
from dotenv import load_dotenv

load_dotenv(".env")

_raw_keys = os.getenv("GEMINI_API_KEYS", "") or os.getenv("GEMINI_API_KEY", "")
GEMINI_API_KEYS = [k.strip() for k in _raw_keys.split(",") if k.strip()]

MODEL_ID = "gemini-2.5-flash"
PYTHON_VERSION = "3.11"
TIMEOUT = 600

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
    timeout=TIMEOUT,
    scaledown_window=300,
    secrets=[modal.Secret.from_dict({"GEMINI_API_KEYS": ",".join(GEMINI_API_KEYS)})],
)
class ChatbotAPI:

    @modal.enter()
    def load_model(self):
        import sys
        sys.path.insert(0, "/root/project")

        from src.chatbot import Chatbot3YNO

        raw = os.environ.get("GEMINI_API_KEYS", "")
        self.api_keys = [k.strip() for k in raw.split(",") if k.strip()]
        if not self.api_keys:
            raise RuntimeError("No Gemini API keys found. Set GEMINI_API_KEYS in .env")

        print(f"3YNO Chatbot API ready. {len(self.api_keys)} key(s) available.", flush=True)
        self.Chatbot3YNO = Chatbot3YNO
        self.model_id = MODEL_ID

        self._init_bot(self.api_keys[0])

    def _init_bot(self, api_key: str):
        self.bot = self.Chatbot3YNO()
        self.bot.load_model(gemini_api_key=api_key, model_id=self.model_id)
        self._active_key = api_key

    def _send_with_fallback(self, message: str) -> str:
        keys = self.api_keys.copy()
        random.shuffle(keys)

        if self._active_key in keys:
            keys.remove(self._active_key)
        keys.insert(0, self._active_key)

        last_error = None
        for key in keys:
            try:
                if key != self._active_key:
                    print(f"Switching to key ...{key[-6:]}", flush=True)
                    self._init_bot(key)
                print(f"Trying key ...{key[-6:]}", flush=True)
                reply = self.bot.send_message(message)
                print(f"Success with key ...{key[-6:]}", flush=True)
                return reply
            except Exception as e:
                last_error = e
                print(f"Key ...{key[-6:]} failed: {e}. Trying next key immediately...", flush=True)

        raise RuntimeError(f"All {len(keys)} API keys failed. Last error: {last_error}")

    @modal.fastapi_endpoint(method="POST", docs=True)
    def chat(self, request: dict):
        import sys
        sys.path.insert(0, "/root/project")

        message = request.get("message", "").strip()
        history = request.get("history", [])

        if not message:
            return {"error": "message is required"}

        self.bot.reset()

        if history and hasattr(self.bot, "set_history"):
            self.bot.set_history(history)

        reply = self._send_with_fallback(message)

        return {
            "reply": reply,
            "history": self.bot.get_history(),
        }