# tasks/evaluation_tasks.py
from crewai import Task
import os

def create_evaluation_task(agent, answer_key_path, student_answers_path, report_output_path):
    """
    Creates the evaluation task.
    It is given all three file paths it needs to call its tool.
    """
    return Task(
        description=f"""
        Your goal is to evaluate the student's answer sheet.
        
        1.  **Find Answer Key:** The master answer key JSON is at:
            '{answer_key_path}'
            
        2.  **Find Student Answers:** The student's answer JSON is at:
            '{student_answers_path}'
            
        3.  **Define Output Path:** You must save your evaluation report to:
            '{report_output_path}'
            
        4.  **EVALUATE:** Use your 'AnswerEvaluationTool'. It requires three arguments:
            - `answer_key_path`: The file path from Step 1.
            - `student_answers_path`: The file path from Step 2.
            - `report_output_path`: The file path from Step 3.
        """,
        agent=agent,
        expected_output=f"""A JSON object with the full evaluation report, which has
        also been saved to '{report_output_path}'."""
    )