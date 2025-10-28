from crewai import Agent
from tools.alignment_tool import AlignmentTool

def create_alignment_agent():
    return Agent(
        role='Image Alignment Specialist',
        goal='Accurately align scanned answer sheets with the template',
        backstory="""You are an expert in computer vision and image processing.
        Your specialty is detecting and correcting geometric distortions in scanned documents.
        You ensure that every student answer sheet is perfectly aligned with the template.""",
        tools=[AlignmentTool()],
        verbose=True,
        allow_delegation=False
    )