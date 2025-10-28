# agents/validation_agent.py
from crewai import Agent
from langchain_google_genai import ChatGoogleGenerativeAI
import os

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro-latest",
    verbose=True,
    temperature=0.1,
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

def create_validation_agent():
    return Agent(
        role='Quality Assurance Specialist',
        goal='Validate results and flag uncertain cases for manual review',
        backstory="""You are a quality assurance expert who reviews the entire evaluation pipeline.
        You identify low-confidence results, potential errors, and cases that need human review.""",
        llm=llm,  # IMPORTANT: Add this
        verbose=True,
        allow_delegation=False
    )