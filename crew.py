from crewai import Crew, Process, Task  # <-- Fixed: Added 'Task' import
from agents.alignment_agent import create_alignment_agent
from agents.ocr_agent import create_ocr_agent
from agents.evaluation_agent import create_evaluation_agent
from agents.validation_agent import create_validation_agent
from tasks.alignment_tasks import create_alignment_task
from tasks.ocr_tasks import create_ocr_task
from tasks.evaluation_tasks import create_evaluation_task

class SASESCrew:
    def __init__(self):
        # Create agents
        self.alignment_agent = create_alignment_agent()
        self.ocr_agent = create_ocr_agent()
        self.evaluation_agent = create_evaluation_agent()
        self.validation_agent = create_validation_agent()
    
    def process_answer_sheet(self, 
                             template_path: str,
                             student_sheet_path: str,
                             reference_answers: list,
                             question_regions: list = None):
        """
        Process a single answer sheet through the complete pipeline
        """
        
        # --- This is the new, correct data-passing logic ---
        # 1. Create the 'inputs' dictionary for kickoff.
        #    This is where all runtime data goes.
        inputs = {
            "template_path": template_path,
            "student_sheet_path": student_sheet_path,
            "reference_answers": reference_answers,
            "question_regions": question_regions or []
        }

        # --- 2. Create tasks as templates ---
        #    The 'create_..._task' functions should NOT be passed
        #    runtime data like 'template_path'. They just create the task.
        
        # Task 1: Alignment
        # The description for this task should use placeholders:
        # e.g., "Align {student_sheet_path} with {template_path}"
        alignment_task = create_alignment_task(self.alignment_agent)
        
        # Task 2: OCR (depends on alignment)
        # This task will automatically receive the output of alignment_task
        # because of the 'context' property.
        ocr_task = create_ocr_task(self.ocr_agent)
        ocr_task.context = [alignment_task]
        
        # Task 3: Evaluation (depends on OCR)
        # This task's description should use the "{reference_answers}" placeholder
        evaluation_task = create_evaluation_task(self.evaluation_agent)
        evaluation_task.context = [ocr_task]
        
        # Task 4: Validation
        validation_task = Task(
            description="""
            Review the complete evaluation pipeline.
            Check alignment confidence from the alignment task.
            Review OCR quality from the OCR task.
            Validate evaluation results from the evaluation task.
            Flag cases needing manual review and provide a final quality report.
            """,
            agent=self.validation_agent,
            expected_output="Final quality report as a JSON object, with a 'manual_review_needed' flag.",
            context=[alignment_task, ocr_task, evaluation_task]
        )
        
        # --- 3. Create crew ---
        crew = Crew(
            agents=[
                self.alignment_agent,
                self.ocr_agent,
                self.evaluation_agent,
                self.validation_agent
            ],
            tasks=[
                alignment_task,
                ocr_task,
                evaluation_task,
                validation_task
            ],
            process=Process.sequential,
            verbose=True
        )
        
        # --- 4. Execute ---
        # Pass the 'inputs' dictionary here.
        result = crew.kickoff(inputs=inputs)
        
        return result