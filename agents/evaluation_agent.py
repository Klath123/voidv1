# agents/evaluation_agent.py
from crewai import Agent
from tools.evaluation_tool import EvaluationTool
from langchain_google_genai import ChatGoogleGenerativeAI
import os

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro-latest",
    verbose=True,
    temperature=0.1,
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

def create_evaluation_agent():
    return Agent(
        role='Answer Evaluation Specialist',
        goal='Accurately evaluate student answers against reference answers',
        backstory="""You are an expert in automated grading systems.
        You can handle multiple question types (MCQ, fill-in-the-blank, one-word)
        and provide fair, consistent evaluations with confidence scores.""",
        tools=[EvaluationTool()],
        llm=llm,  # IMPORTANT: Add this
        verbose=True,
        allow_delegation=False
    )