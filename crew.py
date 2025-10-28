import os
from crewai import Crew, Process, Task 
from agents.alignment_agent import create_alignment_agent
from agents.ocr_agent import create_ocr_agent
from agents.evaluation_agent import create_evaluation_agent
from agents.insight_agent import create_insight_agent

# --- 1. IMPORT THE NEW AGENT ---
from agents.insight_agent import create_insight_agent

# --- IMPORT ALL THE TASK FUNCTIONS ---
from tasks.alignment_tasks import create_alignment_task
from tasks.ocr_tasks import (
    create_key_generation_task, 
    create_student_extraction_task
)
from tasks.evaluation_tasks import create_evaluation_task

# --- 2. IMPORT THE NEW TASK ---
from tasks.insight_tasks import create_insight_task


class SASESCrew:
    def __init__(self):
        # Create agents
        self.alignment_agent = create_alignment_agent()
        self.ocr_agent = create_ocr_agent()
        self.evaluation_agent = create_evaluation_agent()
        self.validation_agent = create_insight_agent()
        
        # --- 3. ADD THE NEW AGENT ---
        self.insight_agent = create_insight_agent()
    
    def process_answer_sheet(self, 
                             template_path: str,
                             teacher_sheet_path: str,
                             student_sheet_path: str):
        
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
        key_generation_task.context = [teacher_alignment_task]
        
        student_extraction_task = create_student_extraction_task(
            self.ocr_agent,
            student_sheet_path,
            student_answers_json_path
        )
        student_extraction_task.context = [student_alignment_task]
        
        # --- Evaluation Phase (Task 5) ---
        evaluation_task = create_evaluation_task(
            self.evaluation_agent,
            teacher_key_json_path,
            student_answers_json_path,
            report_output_path
        )
        evaluation_task.context = [key_generation_task, student_extraction_task]
        
        # --- 4. ADD THE NEW INSIGHT TASK (Now Task 6) ---
        insight_task = create_insight_task(
            self.insight_agent,
            report_output_path  # It takes the report path as input
        )
        # Set dependency on the evaluation task
        insight_task.context = [evaluation_task]

        # --- Validation Phase (Now Task 7) ---
        validation_task = Task(
            description="""
            Review the complete evaluation and insight pipeline.
            Check alignment confidence.
            Review OCR quality.
            Validate evaluation results.
            Verify that the final insights are generated and saved.
            Flag cases needing manual review.
            """,
            agent=self.validation_agent,
            expected_output="Final quality report as a JSON object, with a 'manual_review_needed' flag.",
            context=[
                teacher_alignment_task,
                student_alignment_task,
                key_generation_task,
                student_extraction_task,
                evaluation_task,
                insight_task  # --- 5. ADD INSIGHT TASK TO FINAL CONTEXT ---
            ]
        )
        
        # 6. ADD NEW AGENT AND TASK TO THE CREW
        crew = Crew(
            agents=[
                self.alignment_agent,
                self.ocr_agent,
                self.evaluation_agent,
                self.insight_agent,  # <-- Added new agent
                self.validation_agent
            ],
            tasks=[
                teacher_alignment_task,
                student_alignment_task,
                key_generation_task,
                student_extraction_task,
                evaluation_task,
                insight_task,  # <-- Added new task
                validation_task
            ],
            process=Process.sequential,
            verbose=True
        )
        
        result = crew.kickoff()
        
        return result