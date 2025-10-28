from crewai import Task

def create_ocr_task(agent, aligned_image_path, question_regions):
    return Task(
        description=f"""
        Extract text from the aligned answer sheet:
        - Image: {aligned_image_path}
        - Question Regions: {question_regions}
        
        For each question region:
        1. Extract text using Azure OCR
        2. Identify question number
        3. Extract student answer
        4. Provide confidence scores
        
        Return structured data with answers mapped to questions.
        """,
        agent=agent,
        expected_output="""JSON with:
        - extracted_answers: list of {q_num, answer, confidence}
        - full_text
        - ocr_quality_score"""
    )