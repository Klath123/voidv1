from dotenv import load_dotenv
load_dotenv()  # Must be BEFORE any other imports that use env vars

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
# Pydantic and List/Optional are no longer needed as per main.py logic
# from pydantic import BaseModel
# from typing import List, Optional
import shutil
import os
from crew import SASESCrew  # This imports agents which need GOOGLE_API_KEY
import json

app = FastAPI(title="SASES API")

# Initialize crew
sases_crew = SASESCrew()

# --- Removed Pydantic models ---
# The models 'ReferenceAnswer' and 'EvaluationRequest' are not used in
# the main.py workflow, which relies on a teacher's answer sheet image.

@app.post("/api/v1/evaluate")
async def evaluate_answer_sheet(
    template: UploadFile = File(...),
    teacher_sheet: UploadFile = File(...),  # <-- ADDED
    student_sheet: UploadFile = File(...)
    # reference_answers: str = Form(...)  <-- REMOVED
):
    """
    Main endpoint to evaluate answer sheet by providing
    template, teacher's key, and the student's sheet.
    """
    
    # Define temp paths
    temp_dir = "temp"
    os.makedirs(temp_dir, exist_ok=True)
    
    template_path = os.path.join(temp_dir, template.filename)
    teacher_path = os.path.join(temp_dir, teacher_sheet.filename) # <-- ADDED
    student_path = os.path.join(temp_dir, student_sheet.filename)

    try:
        # Save uploaded files
        with open(template_path, "wb") as f:
            shutil.copyfileobj(template.file, f)
        
        with open(teacher_path, "wb") as f:  # <-- ADDED
            shutil.copyfileobj(teacher_sheet.file, f)
            
        with open(student_path, "wb") as f:
            shutil.copyfileobj(student_sheet.file, f)
        
        # --- Removed parsing of reference_answers ---
        
        # Process through crew (matching main.py)
        print(f"Starting API pipeline for: {student_path}")
        result = sases_crew.process_answer_sheet(
            template_path=template_path,
            teacher_sheet_path=teacher_path,  # <-- ADDED
            student_sheet_path=student_path
        )
        
        # Use .model_dump() as seen in main.py for clean JSON output
        final_output = {}
        try:
            final_output = result.model_dump()
        except AttributeError:
            final_output = result  # Fallback for older versions

        return JSONResponse({
            "success": True,
            "result": final_output  # <-- Use serialized output
        })
        
    except Exception as e:
        return JSONResponse({
            "success": False,
            "error": str(e)
        }, status_code=500)
    
    finally:
        # Clean up temp files
        if os.path.exists(template_path):
            os.remove(template_path)
        if os.path.exists(teacher_path):
            os.remove(teacher_path)
        if os.path.exists(student_path):
            os.remove(student_path)

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    # Run on all interfaces on port 8000
    uvicorn.run(app)