from crewai import Task

def create_alignment_task(agent, template_path, student_sheet_path):
    return Task(
        description=f"""
        Align the scanned student answer sheet with the template:
        - Template: {template_path}
        - Student Sheet: {student_sheet_path}
        
        Steps:
        1. Detect and correct skewness
        2. Apply feature-based alignment
        3. Correct perspective distortions
        4. Calculate confidence score
        5. Save aligned image
        
        Provide transformation parameters and confidence score.
        """,
        agent=agent,
        expected_output="""JSON with:
        - aligned_image_path
        - transform_matrix
        - confidence_score
        - transformations_applied"""
    )