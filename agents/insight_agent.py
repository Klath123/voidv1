# agents/insight_agent.py
from crewai import Agent
from crewai import LLM
import os

# Import the new tools
from tools.insight_tool import FileReaderTool, FileWriterTool

# Reuse the same LLM configuration
llm = LLM(
    model="gemini/gemini-2.5-flash",
    temperature=0.1,
    api_key=os.getenv("GOOGLE_API_KEY")
)

def create_insight_agent():
    return Agent(
        role='Academic Performance Analyst',
        goal='Analyze a student\'s evaluation report to provide actionable, qualitative feedback.',
        backstory="""You are an experienced and empathetic educator. You excel at
        looking at raw evaluation data (like scores and wrong answers) and
        transforming it into constructive, encouraging feedback that helps
        the student understand their strengths and pinpoint exactly
        where they need to improve.""",
        tools=[
            FileReaderTool(),
            FileWriterTool()
        ],
        llm=llm,
        verbose=True,
        allow_delegation=False
    )