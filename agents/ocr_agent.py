from crewai import Agent
from tools.azure_ocr_tool import AzureOCRTool

def create_ocr_agent():
    return Agent(
        role='OCR Extraction Specialist',
        goal='Extract text accurately from aligned answer sheets',
        backstory="""You are an expert in optical character recognition and handwriting analysis.
        You use Azure's state-of-the-art OCR technology to extract text with high accuracy,
        even from challenging handwritten content.""",
        tools=[AzureOCRTool()],
        verbose=True,
        allow_delegation=False
    )