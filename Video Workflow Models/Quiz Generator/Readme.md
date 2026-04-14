# MCQ Quiz Generator Model

This model generates multiple-choice questions (MCQs) from a given text summary using a Large Language Model (LLM). It is designed to be integrated into an educational system where users receive a summary and are then tested through automatically generated quizzes.

---

##  Features

- Generate 5 MCQs automatically from any input text  
- Each question includes 4 choices  
- Hidden correct answers for evaluation  
- Quiz evaluation with:
  - Score calculation  
  - Correct/incorrect feedback  
  - Explanations for each answer  
- Fully deployed using Modal  
- Built with FastAPI  

---

##  How It Works

1. The system receives a summary text (from another model or input).  
2. The API generates 5 MCQs based on the content.  
3. The frontend displays only the questions and choices.  
4. The user submits their answers.  
5. The API evaluates the answers and returns the final score with feedback.  

---

##  API Endpoints

### 1. Generate Quiz

**POST** `/generate-quiz/`

**Request Body:**
```json
{
  "content": "Your summary text here"
}

Response:

{
  "questions": [],
  "full_data": []
}
questions: Contains only questions and choices (for the user)
full_data: Contains answers and explanations (used for evaluation)
2. Submit Quiz

POST /submit-quiz/

Request Body:

{
  "answers": [
    {
      "question_index": 0,
      "selected_index": 1
    }
  ],
  "questions": []
}

Response:

{
  "score": 4,
  "total": 5,
  "feedback": []
}
```
---

##  Tech Stack
Python
FastAPI
LangChain
Groq API (LLM)

##  Modal (Deployment)
  Setup (Local) :
git clone <repo-link>
cd <project-folder>
pip install -r requirements.txt

Create .env file:

GROQ_API_KEY=your_api_key_here

Run locally:

python main_modal.py
 Deployment

The API is deployed using Modal.

modal deploy main_modal.py
 ##  Notes
Correct answers are hidden from users
Evaluation is handled via /submit-quiz/
API is stateless (no user data stored)
 ##  Integration

This API can be used with:

Summary generation models
Frontend / mobile apps
