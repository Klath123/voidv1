# tasks/alignment_tasks.py
from crewai import Task
import os

def _get_output_path(input_path, suffix):
    """Helper to create a unique output path in the 'outputs' folder."""
    base_name = os.path.basename(input_path)
    file_name, _ = os.path.splitext(base_name)
    # e.g., outputs/aligned_student_sheet_001_student.jpg
    return f"outputs/aligned_{file_name}_{suffix}.jpg"

def create_alignment_task(agent, template_path, sheet_path, sheet_type='student'):
    """
    Creates a task to align either a student or teacher sheet.
    
    Args:
        agent: The alignment agent.
        template_path: Path to the blank template image.
        sheet_path: Path to the sheet (student or teacher) to be aligned.
        sheet_type: A string ('student' or 'teacher') for naming the output.
    """
    output_image_path = _get_output_path(sheet_path, sheet_type)
    
    return Task(
        description=f"""
        Align the {sheet_type} sheet against the template.
        
        1.  **Template Path:** '{template_path}'
        2.  **Sheet Path:** '{sheet_path}'
        3.  **Output Path:** Save the aligned image to '{output_image_path}'
        
        Use your 'ImageAlignmentTool' to perform the alignment.
        Your tool must return the 'aligned_image_path' in its output.
        """,
        agent=agent,
        expected_output=f"A JSON object with alignment details, including the saved path: 'aligned_image_path': '{output_image_path}'"
    )