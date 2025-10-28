from crewai import Agent
from tools.evaluation_tool import EvaluationTool

def create_evaluation_agent():
    return Agent(
        role='Answer Evaluation Specialist',
        goal='Accurately evaluate student answers against reference answers',
        backstory="""You are an expert in automated grading systems.
        You can handle multiple question types (MCQ, fill-in-the-blank, one-word)
        and provide fair, consistent evaluations with confidence scores.""",
        tools=[EvaluationTool()],
        verbose=True,
        allow_delegation=False
    )