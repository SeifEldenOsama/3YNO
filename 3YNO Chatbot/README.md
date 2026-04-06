# 3YNO Chatbot

A conversational AI assistant specialised in dyslexia support and visual learning. Powered by **Google Gemini 2.5 Flash**, deployed on **Modal** (serverless cloud) with a FastAPI endpoint. The chatbot supports parents, teachers, therapists, and caregivers with expert guidance on dyslexia signs, learning strategies, and resources — while never diagnosing.

---

## Project Structure

```
3YNO Chatbot/
├── .env                 ← credentials (never commit)
├── .env.example
├── .gitignore
├── requirements.txt
├── Makefile
│
├── src/
│   ├── chatbot.py       ← Chatbot3YNO class (Gemini multi-turn chat)
│   └── system_prompt.py ← Full system prompt defining 3YNO's identity & knowledge
│
├── cloud/
│   └── run.py           ← Modal cloud entrypoint (single message)
│
├── API/
│   └── API.py           ← Modal-hosted FastAPI endpoint (persistent chatbot)
│
└── scripts/
    └── chat.py          ← Local interactive CLI
```

---

## Setup

```bash
pip install -r requirements.txt
```

```bash
cp .env.example .env
```

Fill in `.env`:
```env
GEMINI_API_KEY=your_google_gemini_api_key_here
```

> ⚠️ **Never commit `.env` to git** — it's already in `.gitignore`

---

## Run Locally (Interactive CLI)

Starts an interactive terminal session with the chatbot:

```bash
# Using Makefile
make chat

# Or directly
python scripts/chat.py
```

Type `reset` to clear conversation history, or `quit` to exit.

---

## Run on Modal (Single Message)

Sends a single message to the chatbot via Modal serverless:

```bash
# Using Makefile
make modal-chat MSG="What is dyslexia?"

# Or directly
modal run cloud/run.py --message "What is dyslexia?"
```

Optional flags:
```
--message   The message to send (default: "What is dyslexia?")
--reset     Pass --reset to clear conversation history before sending
```

---

## Deploy API (Modal-hosted FastAPI)

Deploy a persistent HTTP endpoint:

```bash
# Using Makefile
make deploy

# Or directly
modal deploy API/API.py
```

### Chat endpoint

```bash
POST https://<your-modal-url>/chat
Content-Type: application/json

{
  "message": "What are signs of dyslexia in a 7-year-old?",
  "reset": false
}
```

Response:
```json
{
  "reply": "...",
  "history": [...]
}
```

### Reset endpoint

```bash
POST https://<your-modal-url>/reset_conversation
```

Response:
```json
{ "status": "conversation reset" }
```

---

## Configuration

| Setting | Value | Where |
|---|---|---|
| Model | `gemini-2.5-flash` | `API/API.py`, `cloud/run.py`, `scripts/chat.py` |
| Temperature | `0.7` | `src/chatbot.py` |
| Top-p | `0.9` | `src/chatbot.py` |
| Max output tokens | `2048` | `src/chatbot.py` |
| Modal timeout | `600 s` | `API/API.py`, `cloud/run.py` |
| Scale-down window | `300 s` | `API/API.py` |

---

## Model

| | |
|---|---|
| Model | `gemini-2.5-flash` |
| Provider | Google Generative AI |
| Approach | Prompt engineering (system prompt) |
| Multi-turn | Yes — full conversation history maintained per session |
| GPU required | No |
| Cloud runtime | [Modal](https://modal.com) |

---

## What the Chatbot Knows

The system prompt equips 3YNO with deep knowledge across:

- **Dyslexia**: types, signs by age group, neurological basis, myths vs facts
- **Assessment**: professional evaluation process, tools, red flags for referral
- **Teaching strategies**: multi-sensory learning, structured literacy, classroom accommodations
- **At-home support**: reading routines, confidence building, assistive tools
- **Technology**: dyslexia-friendly apps, text-to-speech tools, font recommendations
- **Emotional support**: coping strategies for children and caregivers
- **Legal & school rights**: IEP, 504 plans, educational accommodations

3YNO never diagnoses — it informs, guides, and empowers.

---

## License

Apache 2.0
