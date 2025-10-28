# tasks/insight_tasks.py
from crewai import Task
import os
import json

def create_insight_task(agent, evaluation_report_path):
    """
    Creates the task for generating insights from the evaluation report.
    """
    # Define the output path for the new insights JSON
    insight_json_path = evaluation_report_path.replace("_report.json", "_insights.json")

    return Task(
        description=f"""
        Your goal is to generate a comprehensive, actionable academic report
        from a student's evaluation JSON.

        1.  **Read the Report:** The evaluation report is located at:
            '{evaluation_report_path}'
            Use your 'File Reader Tool' to read its content.

        2.  **Analyze the Data:** The file content will be a JSON string.
            Parse this JSON in your mind. It has two main keys: "summary"
            and "detailed_results".
            
            - The "summary" contains all the key statistics.
            - The "detailed_results" contains the question-by-question breakdown.

        3.  **Draft Insights JSON:** Based on your analysis, generate insights
            for the student. I want these insights drafted as a JSON object
            with these *exact* keys. Copy the stats directly from the
            report's "summary" section.

            - **`total_questions`**: (number)
            - **`correct_answers`**: (number)
            - **`wrong_answers`**: (number)
            - **`unanswered`**: (number)
            - **`score_percentage`**: (string, from "accuracy_percent")
            - **`overall_performance`**: (string) A 1-2 sentence summary.
            - **`strengths`**: (list of strings) What the student did well.
            - **`areas_for_improvement`**: (list of strings) Actionable advice.
            - **`motivational_feedback`**: (string) An encouraging closing remark.
            
            **Example JSON to generate:**
            {{
                "total_questions": 7,
                "correct_answers": 5,
                "wrong_answers": 1,
                "unanswered": 1,
                "score_percentage": "71.43%",
                "overall_performance": "Good effort on the test, with strong performance on multiple-choice questions but some struggles with fill-in-the-blanks.",
                "strengths": [
                    "Excellent accuracy on multiple-choice questions.",
                    "Attempted almost all questions."
                ],
                "areas_for_improvement": [
                    "Review the 'Fill in the Blanks' section, as this was the main source of incorrect answers.",
                    "Be sure to answer every question, as one was left unanswered."
                ],
                "motivational_feedback": "You're on the right track! Keep reviewing the 'fill-in-the-blanks' material and you'll ace the next one."
            }}

        4.  **Format as JSON String:** Make *sure* the content you write to the
            file is *only* the valid JSON string you just generated.

        5.  **Save the Insights:** Use your 'File Writer Tool' to save this
            new JSON string to this *exact* path:
            '{insight_json_path}'
        """,
        agent=agent,
        expected_output=f"A new, detailed JSON file with academic insights, saved to {insight_json_path}."
    )