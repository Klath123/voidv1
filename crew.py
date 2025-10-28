import os
from crewai import Crew, Process, Task 
from agents.alignment_agent import create_alignment_agent
from agents.ocr_agent import create_ocr_agent
from agents.evaluation_agent import create_evaluation_agent
from agents.validation_agent import create_validation_agent

# --- IMPORT ALL THE TASK FUNCTIONS ---
from tasks.alignment_tasks import create_alignment_task

# --- THIS IS THE KEY IMPORT CHANGE ---
# Import the two functions from your ocr_tasks.py file
from tasks.ocr_tasks import (
    create_key_generation_task, 
    create_student_extraction_task
)
# --- END CHANGE ---

from tasks.evaluation_tasks import create_evaluation_task

class SASESCrew:
    def __init__(self):
        # Create agents (this part is unchanged)
        self.alignment_agent = create_alignment_agent()
        self.ocr_agent = create_ocr_agent()
        self.evaluation_agent = create_evaluation_agent()
        self.validation_agent = create_validation_agent()
    
    # --- THIS METHOD SIGNATURE IS UPDATED ---
    # It no longer needs 'reference_answers'
    # It now REQUIRES 'teacher_sheet_path'
    def process_answer_sheet(self, 
                             template_path: str,
                             teacher_sheet_path: str,
                             student_sheet_path: str):
        """
        Process a single answer sheet through the complete, image-based pipeline.
        """
        
        # --- Define output file paths based on inputs ---
        teacher_key_json_path = f"outputs/{os.path.splitext(os.path.basename(teacher_sheet_path))[0]}_key.json"
        student_answers_json_path = f"outputs/{os.path.splitext(os.path.basename(student_sheet_path))[0]}_answers.json"
        report_output_path = f"outputs/{os.path.splitext(os.path.basename(student_sheet_path))[0]}_report.json"
        
        # --- Alignment Phase (Tasks 1 & 2) ---
        teacher_alignment_task = create_alignment_task(
            self.alignment_agent,
            template_path,
            teacher_sheet_path,
            sheet_type='teacher'
        )
        
        student_alignment_task = create_alignment_task(
            self.alignment_agent,
            template_path,
            student_sheet_path,
            sheet_type='student'
        )
        
        # --- OCR Phase (Tasks 3 & 4) ---
        key_generation_task = create_key_generation_task(
            self.ocr_agent,
            teacher_sheet_path,
            teacher_key_json_path
        )
        # Set dependency on teacher alignment
        key_generation_task.context = [teacher_alignment_task]
        
        student_extraction_task = create_student_extraction_task(
            self.ocr_agent,
            student_sheet_path,
            student_answers_json_path
        )
        # Set dependency on student alignment
        student_extraction_task.context = [student_alignment_task]
        
        # --- Evaluation Phase (Task 5) ---
        evaluation_task = create_evaluation_task(
            self.evaluation_agent,
            teacher_key_json_path,    # Path to the key
            student_answers_json_path, # Path to the student answers
            report_output_path         # Path for the final report
        )
        # Set dependency on *both* OCR tasks
        evaluation_task.context = [key_generation_task, student_extraction_task]
        
        # --- Validation Phase (Task 6) ---
        validation_task = Task(
            description="""
            Review the complete evaluation pipeline.
            Check alignment confidence from both alignment tasks.
            Review OCR quality from both OCR tasks.
            Validate evaluation results from the evaluation task.
            Flag cases needing manual review and provide a final quality report.
            """,
            agent=self.validation_agent,
            expected_output="Final quality report as a JSON object, with a 'manual_review_needed' flag.",
            context=[
                teacher_alignment_task,
                student_alignment_task,
                key_generation_task,
                student_extraction_task,
                evaluation_task
            ]
        )
        
        # 2. Create crew
        crew = Crew(
            agents=[
                self.alignment_agent,
                self.ocr_agent,
                self.evaluation_agent,
                self.validation_agent
            ],
            tasks=[
                # Tasks will run in correct order based on dependencies
                teacher_alignment_task,
                student_alignment_task,
                key_generation_task,
                student_extraction_task,
                evaluation_task,
                validation_task
            ],
            process=Process.sequential,
            verbose=True
        )
        
        # 3. Execute
        # No inputs dict is needed, paths are passed via task descriptions
        result = crew.kickoff()
        
        return result