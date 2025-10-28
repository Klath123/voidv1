from crewai import Task

# In ocr_tasks.py
import os

def create_ocr_task(agent, question_regions, student_sheet_path):
    
    # --- ADD THIS ---
    # Dynamically create a unique output path for the JSON
    base_name = os.path.basename(student_sheet_path)
    file_name, _ = os.path.splitext(base_name)
    output_json_path = f"outputs/{file_name}_ocr_results.json"
    # --- END ADD ---

    return Task(
        description=f"""
        Your goal is to perform OCR on an aligned answer sheet.
        
        1.  *FIND THE IMAGE:* Look in your context. The 'alignment_task'
            provided its output. Find the 'aligned_image_path' value
            from that output. This is the image you must process.
            
        2.  *DEFINE OUTPUT PATH:* You must save your results to this
            exact file path: '{output_json_path}'
            
        3.  *PROCESS IMAGE:* Use your 'AzureOCRTool' with the
            'aligned_image_path' you found and the 'output_json_path'
            I provided.
            
        4.  *USE REGIONS (Optional):* The question regions are:
            {question_regions}
        """,
        agent=agent,
        expected_output=f"""A JSON object with the full OCR results, which has
        also been saved to '{output_json_path}'."""
    )