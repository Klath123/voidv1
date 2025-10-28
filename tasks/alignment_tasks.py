from crewai import Task

def create_alignment_task(agent, template_path, student_sheet_path):
    return Task(
        description=f"""
        Use the Image Alignment Tool to align the scanned student answer sheet with the template.
        
        IMPORTANT: You MUST call the Image Alignment Tool with these exact parameters:
        - template_path: {template_path}
        - student_sheet_path: {student_sheet_path}
        
        The tool will:
        1. Detect and correct skewness
        2. Apply feature-based alignment using ORB features
        3. Correct perspective distortions using homography
        4. Calculate confidence score
        5. Save the aligned image to the 'outputs' directory
        
        After the tool completes, report the results including:
        - Path where aligned image was saved
        - Transformation matrix
        - Confidence score
        - Whether alignment was successful
        """,
        agent=agent,
        expected_output="""A JSON object containing:
        {{
            "success": true/false,
            "aligned_image_path": "path/to/aligned/image.jpg",
            "transform_matrix": [[matrix values]],
            "confidence_score": 0.XX,
            "transformations_applied": {{"homography_alignment": true}}
        }}""",
        output_file="alignment_result.json"  # Optional: saves result to file
    )