from crewai import Task

def create_evaluation_task(agent, student_answers, reference_answers):
    return Task(
        description=f"""
        Evaluate student answers against reference:
        - Student Answers: {len(student_answers)} questions
        - Reference Answers provided
        
        For each answer:
        1. Match student answer to reference
        2. Apply question-type specific evaluation
        3. Calculate marks
        4. Provide confidence score
        
        Handle spelling variations and synonyms for text answers.
        """,
        agent=agent,
        expected_output="""JSON with:
        - total_marks
        - obtained_marks
        - percentage
        - question_results: list with per-question evaluation"""
    )