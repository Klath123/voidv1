# main.py
import json
from dotenv import load_dotenv

# 1. Load all environment variables
# (GOOGLE_API_KEY, AZURE_VISION_ENDPOINT, AZURE_VISION_KEY)
load_dotenv() 

# 2. Import your main crew class
from crew import SASESCrew

def run_full_pipeline():
    """
    Initializes and runs the complete SASESCrew pipeline.
    """
    
    # 3. Initialize the crew
    print("Initializing SASESCrew with all agents...")
    crew = SASESCrew()
    
    # --- Define Your Input Image Paths Here ---
    
    TEMPLATE_PATH = "data/template.jpg"
    TEACHER_SHEET_PATH = "data/teacher_key_sheet.jpg"
    STUDENT_SHEET_PATH = "data/student_sheet_001.jpg"
    
    # -----------------------------------------
    
    # 4. Run the crew
    print(f"Starting full pipeline for: {STUDENT_SHEET_PATH}")
    
    result = crew.process_answer_sheet(
        template_path=TEMPLATE_PATH,
        teacher_sheet_path=TEACHER_SHEET_PATH,
        student_sheet_path=STUDENT_SHEET_PATH
    )
    
    # 5. Print the final result
    print("\n--- Full Pipeline Complete ---")
    print("Final Result:")
    
    # Use .model_dump() to get a clean JSON-serializable output
    try:
        print(json.dumps(result.model_dump(), indent=2))
    except AttributeError:
        # Fallback for older crewAI versions
        print(result)

if __name__ == "__main__":
    run_full_pipeline()