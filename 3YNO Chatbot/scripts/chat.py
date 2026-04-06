import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from dotenv import load_dotenv
load_dotenv()

from src.chatbot import Chatbot3YNO

MODEL_ID = "gemini-2.5-flash"


def main():
    gemini_api_key = os.environ.get("GEMINI_API_KEY", "")
    if not gemini_api_key:
        print("Error: GEMINI_API_KEY not set in .env")
        return

    bot = Chatbot3YNO()
    bot.load_model(gemini_api_key=gemini_api_key, model_id=MODEL_ID)

    print("\n" + "="*60)
    print("  3YNO — Dyslexia & Visual Learning Chatbot")
    print("  Type 'quit' to exit | 'reset' to clear history")
    print("="*60 + "\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() == "quit":
            print("Goodbye!")
            break

        if user_input.lower() == "reset":
            bot.reset()
            print("3YNO: Conversation reset. How can I help you?\n")
            continue

        reply = bot.send_message(user_input)
        print(f"\n3YNO: {reply}\n")


if __name__ == "__main__":
    main()
