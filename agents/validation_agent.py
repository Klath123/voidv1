# agents/validation_agent.py
from crewai import Agent
from crewai import LLM
import os

llm = LLM(
    model="gemini/gemini-2.5-pro",  # Note the "gemini/" prefix for LiteLLM
    temperature=0.1,
    api_key=os.getenv("GOOGLE_API_KEY")
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