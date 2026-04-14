# main_modal.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from modal import Image, App, Secret, asgi_app
from mcq_generator import generate_mcq
from typing import List

# ===========================
# إعداد Modal App
# ===========================
app = App("mcq-generator-api")

image = Image.debian_slim()\
    .pip_install(
        "fastapi",
        "uvicorn",
        "langchain-groq",
        "python-dotenv",
        "langchain-core",
        "langchain",
        "langchain-openai"
    )\
    .add_local_dir(".", remote_path="/root")

web_app = FastAPI(title="MCQ Quiz API")

# ===========================
# ✅ تعديل هنا
# ===========================
class QuizRequest(BaseModel):
    content: str   # بدل subject + difficulty + topic

class AnswerItem(BaseModel):
    question_index: int
    selected_index: int

class QuizSubmit(BaseModel):
    answers: List[AnswerItem]
    questions: List[dict]

# ===========================
# Generate Quiz
# ===========================
@web_app.post("/generate-quiz/")
def generate_quiz_endpoint(request: QuizRequest):
    try:
        mcqs = generate_mcq(
            content=request.content   # ✅ التعديل هنا
        )

        mcqs_for_user = []
        for q in mcqs:
            mcqs_for_user.append({
                "question": q["question"],
                "choices": q["choices"]
            })

        return {
            "questions": mcqs_for_user,
            "full_data": mcqs
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ===========================
# Submit Quiz
# ===========================
@web_app.post("/submit-quiz/")
def submit_quiz(submit: QuizSubmit):
    correct_count = 0
    feedback = []

    for ans in submit.answers:
        idx = ans.question_index
        user_choice = ans.selected_index
        question_data = submit.questions[idx]

        correct_index = question_data.get("answer_index", -1)
        is_correct = (user_choice == correct_index)

        if is_correct:
            correct_count += 1

        feedback.append({
            "question": question_data["question"],
            "your_answer": question_data["choices"][user_choice],
            "correct_answer": question_data["choices"][correct_index],
            "correct_index": correct_index,
            "is_correct": is_correct,
            "explanation": question_data.get("explanation", "")
        })

    return {
        "score": correct_count,
        "total": len(submit.answers),
        "feedback": feedback
    }

# ===========================
# Modal تشغيل
# ===========================
@app.function(image=image, secrets=[Secret.from_name("groq-api-key")])
@asgi_app()
def serve_api():
    return web_app

# ===========================
# تشغيل محلي
# ===========================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(web_app, host="0.0.0.0", port=8000)