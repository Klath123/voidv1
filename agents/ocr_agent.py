# agents/ocr_agent.py
from crewai import Agent
from tools.azure_ocr_tool import AzureOCRTool
from langchain_google_genai import ChatGoogleGenerativeAI
import os

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro-latest",
    verbose=True,
    temperature=0.1,
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

def create_ocr_agent():
    return Agent(
        role='OCR Extraction Specialist',
        goal='Extract text accurately from aligned answer sheets',
        backstory="""You are an expert in optical character recognition and handwriting analysis.
        You use Azure's state-of-the-art OCR technology to extract text with high accuracy,
        even from challenging handwritten content.""",
        tools=[AzureOCRTool()],
        llm=llm,  # IMPORTANT: Add this
        verbose=True,
        allow_delegation=False
    )