from crewai import Crew, Process
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
        
        # Task 1: Alignment
        alignment_task = create_alignment_task(
            self.alignment_agent,
            template_path,
            student_sheet_path
        )
        
        # Task 2: OCR (depends on alignment)
        ocr_task = create_ocr_task(
            self.ocr_agent,
            "{alignment_output.aligned_image_path}",
            question_regions or []
        )
        ocr_task.context = [alignment_task]
        
        # Task 3: Evaluation (depends on OCR)
        evaluation_task = create_evaluation_task(
            self.evaluation_agent,
            "{ocr_output.extracted_answers}",
            reference_answers
        )
        evaluation_task.context = [ocr_task]
        
        # Task 4: Validation
        validation_task = Task(
            description="""
            Review the complete evaluation pipeline:
            1. Check alignment confidence
            2. Review OCR quality
            3. Validate evaluation results
            4. Flag cases needing manual review (confidence < 0.7)
            5. Provide final quality report
            """,
            agent=self.validation_agent,
            expected_output="Quality report with flags for manual review",
            context=[alignment_task, ocr_task, evaluation_task]
        )
        
        # Create crew
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
        
        # Execute
        result = crew.kickoff()
        
        return result