from crewai import Crew, Process, Task 
from agents.alignment_agent import create_alignment_agent
from agents.ocr_agent import create_ocr_agent
from agents.evaluation_agent import create_evaluation_agent
from agents.validation_agent import create_validation_agent
from tasks.alignment_tasks import create_alignment_task
from tasks.ocr_tasks import create_ocr_task
from tasks.evaluation_tasks import create_evaluation_task

# NOTE: The definition of create_evaluation_task you provided is correct:
# def create_evaluation_task(agent, student_answers, reference_answers):
#     ...

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
        
        # 1. Create the 'inputs' dictionary for kickoff.
        inputs = {
            "template_path": template_path,
            "student_sheet_path": student_sheet_path,
            "reference_answers": reference_answers,
            "question_regions": question_regions or []
        }

        # 2. Create tasks as templates
        
        # Task 1: Alignment
        alignment_task = create_alignment_task(
            self.alignment_agent, 
            template_path, 
            student_sheet_path
        )
        
        # Task 2: OCR (depends on alignment)
        ocr_task = create_ocr_task(
            self.ocr_agent,
            question_regions=question_regions or [],
            student_sheet_path=student_sheet_path # <-- New argument
        )
        ocr_task.context = [alignment_task] # <-- This is still correct!
        # ...
        
        # Task 3: Evaluation (depends on OCR)
        # *FIX APPLIED HERE:* Passing the two missing required arguments.
        evaluation_task = create_evaluation_task(
            self.evaluation_agent,
            student_answers="OCR_OUTPUT_FROM_CONTEXT", # Placeholder for the OCR output
            reference_answers=reference_answers          # Uses the method's input argument
        )
        evaluation_task.context = [ocr_task] # The evaluation agent will read the student answers from here.
        
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
        
        # 3. Create crew
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
        
        # 4. Execute
        result = crew.kickoff(inputs=inputs)
        
        return result