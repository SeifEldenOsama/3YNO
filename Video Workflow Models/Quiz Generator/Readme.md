MCQ Quiz Generator Model
    This Model generates multiple-choice questions (MCQs) from a given text summary using a Large Language Model (LLM).
    It is designed to be integrated into an educational system where users receive a summary and are then tested through automatically generated quizzes.

>> Features
    Generate 5 MCQs automatically from any input text
    Each question includes 4 choices
    Hidden correct answers for evaluation
    Quiz evaluation with:
    Score calculation
    Correct/incorrect feedback
    Explanations for each answer
    Fully deployed using Modal
    Built with FastAPI


>> How It Works
    The system receives a summary text (from another model or input).
    The API generates 5 MCQs based on the content.
    The frontend displays only the questions and choices.
    The user submits their answers.
    The API evaluates the answers and returns the final score with feedback.

 >> API Endpoints
 1. Generate Quiz
  
   POST /generate-quiz/
  
   Request Body:
   {
     "content": "Your summary text here"
   }
   Response:
   {
     "questions": [...],
     "full_data": [...]
   }
   questions: Contains only questions and choices (for the user)
   full_data: Contains answers and explanations (used for evaluation)
 2. Submit Quiz
  
   POST /submit-quiz/
  
   Request Body:
   {
     "answers": [
     {"question_index": 0, "selected_index": 1}
    ],
     "questions": [...]
   }
   Response:
   {
     "score": 4,
     "total": 5,
     "feedback": [...]
  }

>> Tech Stack
    Python
    FastAPI
    LangChain
    Groq API (LLM)
    Modal (Deployment)

>> Setup (Local)
        Clone the repository:
        git clone <repo-link>
        cd <repo-name>
  Install dependencies:
        pip install -r requirements.txt
        Create a .env file:
        GROQ_API_KEY=your_api_key_here
  
  Run locally:
        python main_modal.py

>> Deployment

  The API is deployed using Modal.
  
  To deploy:
  
  modal deploy main_modal.py


>> Notes
    The correct answers are not shown to the user
    Evaluation is handled securely through the /submit-quiz/ endpoint
    The API is stateless and does not store user data
   

>> Integration
       This API is intended to be integrated with:
       A summary generation model
       A frontend/mobile application for user interaction
