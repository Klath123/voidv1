import json
import os
from crewai.tools import BaseTool
from typing import Dict, Any, List

class AnswerEvaluationTool(BaseTool):
    name: str = "Answer Evaluation Tool"
    description: str = (
        "Compares a student's answer JSON file with the official answer key JSON file. "
        "It calculates statistics like accuracy and precision, then saves "
        "a detailed report to a new JSON file."
    )

    def _run(self, answer_key_path: str, student_answers_path: str, report_output_path: str) -> Dict[str, Any]:
        """
        Compares answer key file (key_path) to student answers file (student_path)
        and saves a report (report_path).
        """
        try:
            # --- 1. Load Both JSON Files ---
            print(f"\n[EvaluationTool] Loading Answer Key: '{answer_key_path}'")
            with open(answer_key_path, 'r') as f:
                key_data = json.load(f)
            
            print(f"[EvaluationTool] Loading Student Answers: '{student_answers_path}'")
            with open(student_answers_path, 'r') as f:
                student_data = json.load(f)

            # --- 2. Create Fast-Lookup Dictionaries ---
            key_mcq = {item['question_number']: item['selected_answer'] for item in key_data.get('multiple_choice', [])}
            key_fib = {item['question_prompt']: item['written_answer'] for item in key_data.get('fill_in_the_blanks', [])}
            
            student_mcq = {item['question_number']: item['selected_answer'] for item in student_data.get('multiple_choice', [])}
            student_fib = {item['question_prompt']: item['written_answer'] for item in student_data.get('fill_in_the_blanks', [])}

            # --- 3. Evaluate and Store Details ---
            total_questions = 0
            correct_answers = 0
            wrong_answers = 0
            unanswered = 0
            detailed_results: List[Dict[str, Any]] = []

            # Evaluate Multiple Choice
            for q_num, correct_ans in key_mcq.items():
                total_questions += 1
                student_ans = student_mcq.get(q_num)
                status = ""
                
                if not student_ans:
                    unanswered += 1
                    status = "unanswered"
                elif student_ans.strip().upper() == correct_ans.strip().upper():
                    correct_answers += 1
                    status = "correct"
                else:
                    wrong_answers += 1
                    status = "wrong"
                
                detailed_results.append({
                    "question": f"MCQ {q_num}",
                    "student_answer": student_ans or "N/A",
                    "correct_answer": correct_ans,
                    "status": status
                })

            # Evaluate Fill-in-the-Blanks
            for q_prompt, correct_ans in key_fib.items():
                total_questions += 1
                student_ans = student_fib.get(q_prompt)
                status = ""

                if not student_ans:
                    unanswered += 1
                    status = "unanswered"
                # Compare as case-insensitive strings
                elif student_ans.strip().lower() == correct_ans.strip().lower():
                    correct_answers += 1
                    status = "correct"
                else:
                    wrong_answers += 1
                    status = "wrong"

                detailed_results.append({
                    "question": q_prompt,
                    "student_answer": student_ans or "N/A",
                    "correct_answer": correct_ans,
                    "status": status
                })

            # --- 4. Calculate Final Metrics (Accuracy & Precision) ---
            total_answered = total_questions - unanswered
            # Accuracy: Correct answers out of all possible questions
            accuracy = (correct_answers / total_questions) * 100 if total_questions > 0 else 0
            # Precision: Correct answers out of the questions the student attempted
            precision = (correct_answers / total_answered) * 100 if total_answered > 0 else 0

            final_report = {
                "summary": {
                    "total_questions": total_questions,
                    "correct_answers": correct_answers,
                    "wrong_answers": wrong_answers,
                    "unanswered": unanswered,
                    "accuracy_percent": f"{accuracy:.2f}%",
                    "precision_of_answered_percent": f"{precision:.2f}%"
                },
                "detailed_results": detailed_results
            }

            # --- 5. Save Report to File ---
            os.makedirs(os.path.dirname(report_output_path), exist_ok=True)
            with open(report_output_path, 'w') as f:
                json.dump(final_report, f, indent=4)
            
            print(f"[EvaluationTool] Successfully saved report to {report_output_path}")
            return final_report

        except FileNotFoundError as e:
            error_msg = f"File not found: {e.filename}"
            print(f"[EvaluationTool] ERROR: {error_msg}")
            return {"error": error_msg}
        except Exception as e:
            error_msg = f"An unexpected error occurred in EvaluationTool: {str(e)}"
            print(f"[EvaluationTool] ERROR: {error_msg}")
            return {"error": error_msg}