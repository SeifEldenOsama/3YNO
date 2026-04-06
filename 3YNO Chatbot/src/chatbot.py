from __future__ import annotations
from src.system_prompt import SYSTEM_PROMPT


class Chatbot3YNO:
    """
    3YNO Chatbot using Google Gemini API.
    Maintains full conversation history for multi-turn chat.
    """

    def __init__(self):
        self.client  = None
        self.model   = None
        self.history = []   # list of {"role": ..., "parts": [...]}

    def load_model(self, gemini_api_key: str, model_id: str = "gemini-2.5-flash"):
        import google.generativeai as genai

        if not gemini_api_key:
            raise RuntimeError("GEMINI_API_KEY not set.")

        genai.configure(api_key=gemini_api_key)

        self.model = genai.GenerativeModel(
            model_name    = model_id,
            system_instruction = SYSTEM_PROMPT,
            generation_config  = genai.GenerationConfig(
                temperature      = 0.7,
                top_p            = 0.9,
                max_output_tokens= 2048,
            ),
        )

        self.chat    = self.model.start_chat(history=[])
        self.history = []
        print(f"3YNO Chatbot ready. Model: {model_id}")

    def send_message(self, user_message: str) -> str:
        """
        Send a message and get a response.
        Maintains full conversation history automatically.
        """
        if not user_message.strip():
            return "Please enter a message."

        response = self.chat.send_message(user_message)
        reply    = response.text.strip()

        # Track history for API responses
        self.history.append({"role": "user",      "content": user_message})
        self.history.append({"role": "assistant",  "content": reply})

        return reply

    def reset(self):
        """Reset conversation history and start fresh."""
        self.chat    = self.model.start_chat(history=[])
        self.history = []
        print("Conversation reset.")

    def get_history(self) -> list:
        return self.history
