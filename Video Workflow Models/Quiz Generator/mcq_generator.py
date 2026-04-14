# mcq_generator.py
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
import json, re, os

load_dotenv()

# عدد الأسئلة ثابت
NUM_QUESTIONS = 5

def get_llm():
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.6,
        groq_api_key=os.getenv("GROQ_API_KEY"),
    )
    return llm

# ❗ هنا أهم تعديل
MCQ_PROMPT = f"""
You are an assistant that generates **{NUM_QUESTIONS} multiple-choice questions (MCQs)**.

The questions must be based ONLY on the provided content.

Return output ONLY as JSON list of objects with keys:
question, choices (list of 4 strings), answer_index (0-3), explanation, difficulty_score (1-10).

Input:
- content: {{content}}

Rules:
- Generate EXACTLY {NUM_QUESTIONS} questions.
- Each question must have exactly 4 choices.
- Do NOT repeat questions.
- Questions must cover different parts of the content.
- Automatically vary difficulty (easy, medium, hard).
- Keep questions short (<= 30 words).
- Make choices realistic and similar.

Generate now.
"""

def generate_mcq(content, past_questions=None):
    past_questions = past_questions or []

    llm = get_llm()
    prompt = ChatPromptTemplate.from_template(MCQ_PROMPT)
    parser = StrOutputParser()

    chain = prompt | llm | parser

    resp = chain.invoke({
        "content": content
    })

    try:
        data = json.loads(resp)
    except Exception:
        j = re.search(r'\[.*\]', resp, re.DOTALL)
        if j:
            data = json.loads(j.group())
        else:
            raise ValueError("Failed to parse JSON from LLM response")

    return data