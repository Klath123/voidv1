from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
import shutil
import os
from crew import SASESCrew
import json

app = FastAPI(title="SASES API")

# Initialize crew
sases_crew = SASESCrew()

class ReferenceAnswer(BaseModel):
    q_num: int
    type: str  # mcq, fill_blank, one_word
    correct: str
    correct_options: Optional[List[str]] = None
    marks: float

class EvaluationRequest(BaseModel):
    reference_answers: List[ReferenceAnswer]
    question_regions: Optional[List[dict]] = None

@app.post("/api/v1/evaluate")
async def evaluate_answer_sheet(
    template: UploadFile = File(...),
    student_sheet: UploadFile = File(...),
    reference_answers: str = Form(...)
):
    """
    Main endpoint to evaluate answer sheet
    """
    try:
        # Save uploaded files
        template_path = f"temp/{template.filename}"
        student_path = f"temp/{student_sheet.filename}"
        
        os.makedirs("temp", exist_ok=True)
        
        with open(template_path, "wb") as f:
            shutil.copyfileobj(template.file, f)
        
        with open(student_path, "wb") as f:
            shutil.copyfileobj(student_sheet.file, f)
        
        # Parse reference answers
        ref_answers = json.loads(reference_answers)
        
        # Process through crew
        result = sases_crew.process_answer_sheet(
            template_path=template_path,
            student_sheet_path=student_path,
            reference_answers=ref_answers
        )
        
        return JSONResponse({
            "success": True,
            "result": result
        })
        
    except Exception as e:
        return JSONResponse({
            "success": False,
            "error": str(e)
        }, status_code=500)

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)