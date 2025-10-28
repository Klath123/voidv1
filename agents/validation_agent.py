from crewai import Agent

def create_validation_agent():
    return Agent(
        role='Quality Assurance Specialist',
        goal='Validate results and flag uncertain cases for manual review',
        backstory="""You are a quality assurance expert who reviews the entire evaluation pipeline.
        You identify low-confidence results, potential errors, and cases that need human review.""",
        verbose=True,
        allow_delegation=False
    )