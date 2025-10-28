# tasks/ocr_tasks.py
from crewai import Task
import os

# --- TASK 1: FOR THE TEACHER'S KEY ---

def create_key_generation_task(agent, teacher_sheet_path, key_output_path):
    """
    Creates the task for generating the master answer key from the teacher's sheet.
    """
    return Task(
        description=f"""
        Generate the master answer key.
        1.  **Input Image:** Use the teacher's answer sheet located at:
            '{teacher_sheet_path}'
        2.  **Output JSON:** Save the extracted answers to this *exact* file path:
            '{key_output_path}'
        
        Use your 'Azure OCR Tool' to process the image and save the formatted JSON.
        """,
        agent=agent,
        expected_output=f"The master answer key, saved as a JSON file to '{key_output_path}'."
    )

# --- TASK 2: FOR THE STUDENT'S SHEET ---

def create_student_extraction_task(agent, student_sheet_path, student_output_path):
    """
    Creates the task for extracting answers from the student's sheet.
    """
    return Task(
        description=f"""
        Extract the student's answers.
        1.  **Input Image:** Use the student's answer sheet located at:
            '{student_sheet_path}'
        2.  **Output JSON:** Save the extracted answers to this *exact* file path:
            '{student_output_path}'
        
    Use your 'Azure OCR Tool' to process the image and save the formatted JSON.
    """,
        agent=agent,
        expected_output=f"The student's answers, saved as a JSON file to '{student_output_path}'."
    )