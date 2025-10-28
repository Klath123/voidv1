
# Add these 2 lines at the top of api.py
from dotenv import load_dotenv
load_dotenv()


from crew import SASESCrew
import json

def main():
    # Initialize crew
    crew = SASESCrew()
    
    # Example usage
    result = crew.process_answer_sheet(
        template_path="data/template.jpg",
        student_sheet_path="data/student_sheet_001.jpg",
        reference_answers=[
            {
                "q_num": 1,
                "type": "mcq",
                "correct": "B",
                "marks": 1
            },
            {
                "q_num": 2,
                "type": "fill_blank",
                "correct": "photosynthesis",
                "correct_options": ["photosynthesis", "photosynthetic"],
                "marks": 2
            }
        ]
    )
    
    print(json.dumps(result.model_dump(), indent=2))

if __name__ == "__main__":
    main()