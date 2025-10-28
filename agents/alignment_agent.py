# agents/alignment_agent.py
from crewai import Agent
from tools.alignment_tool import AlignmentTool
from langchain_google_genai import ChatGoogleGenerativeAI
import os

# Create LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro-latest",
    verbose=True,
    temperature=0.1,
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

def create_alignment_agent():
    return Agent(
        role='Image Alignment Specialist',
        goal='Accurately align scanned answer sheets with the template',
        backstory="""You are an expert in computer vision and image processing.
        Your specialty is detecting and correcting geometric distortions in scanned documents.
        You ensure that every student answer sheet is perfectly aligned with the template.""",
        tools=[AlignmentTool()],
        llm=llm,  # IMPORTANT: Add this
        verbose=True,
        allow_delegation=False
    )