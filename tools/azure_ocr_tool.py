import os
import json  # <-- 1. IMPORT JSON
import time
from crewai.tools import BaseTool
from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeResult
from pydantic import Field
from typing import Any

# Helper function to create the client
def _create_azure_di_client():
    endpoint = os.getenv('AZURE_VISION_ENDPOINT')
    key = os.getenv('AZURE_VISION_KEY')
    
    if not endpoint or not key:
        print("Warning: AZURE_VISION_ENDPOINT or AZURE_VISION_KEY not set. OCR tool will fail.")
        return None
        
    return DocumentIntelligenceClient(
        endpoint=endpoint, 
        credential=AzureKeyCredential(key)
    )

# In ocrtool.py
# ... (keep all your imports and the _create_azure_di_client function) ...

class AzureOCRTool(BaseTool):
    name: str = "Azure OCR Tool"
    description: str = "Performs OCR on an image, intelligently extracts only handwritten answers, and saves them to a structured JSON file."
    
    client: Any = Field(default_factory=_create_azure_di_client)

    # --- REPLACE YOUR OLD _run METHOD WITH THIS ---
    def _run(self, image_path: str, output_json_path: str) -> dict:
        """
        Perform smart OCR on an image, find only handwritten answers,
        save them to the specified JSON path, and return the JSON object.
        """
        if not self.client:
            return {
                'success': False,
                'error': "Azure Document Intelligence client is not initialized. Check environment variables."
            }

        try:
            print(f"OCR Tool: Analyzing {image_path} with 'prebuilt-layout'...")
            
            # Read image
            with open(image_path, "rb") as image_file:
                image_bytes = image_file.read()
            
            # --- THIS IS THE KEY CHANGE ---
            # 1. Use the "prebuilt-layout" model to get handwriting info
            poller = self.client.begin_analyze_document(
                model_id="prebuilt-layout",  # Use layout model
                body=image_bytes, # Correct parameter name
                content_type="application/octet-stream"
            )
            result: AnalyzeResult = poller.result()
            print("OCR Tool: Analysis complete.")

            # 2. Extract only handwritten text
            handwritten_answers = []
            if result.styles:
                for style in result.styles:
                    if style.is_handwritten:
                        for span in style.spans:
                            start = span.offset
                            end = span.offset + span.length
                            handwritten_text = result.content[start:end].strip()
                            
                            if handwritten_text:
                                handwritten_answers.append(handwritten_text)

            print(f"OCR Tool: Found handwritten text: {handwritten_answers}")

            # 3. Format the text into your desired JSON structure
            output_json = {
                "multiple_choice": [],
                "fill_in_the_blanks": []
            }
            
            # Use heuristics to sort answers
            mcq_answers = [text for text in handwritten_answers if len(text) == 1 and text.isalpha()]
            
            # Filter out common non-answer numbers (like '59' from Roll No.)
            # You may need to make this filter smarter later
            fib_answers = [text for text in handwritten_answers if text.isdigit() and text not in ["59","24","2"]]

            for i, answer in enumerate(mcq_answers):
                output_json["multiple_choice"].append({
                    "question_number": str(i + 1),
                    "selected_answer": answer.upper() # Standardize to uppercase
                })

            for i, answer in enumerate(fib_answers):
                output_json["fill_in_the_blanks"].append({
                    "question_prompt": f"Fill in the blank {i + 1}", # Generic prompt
                    "written_answer": answer
                })

            # 4. Save the final JSON file
            try:
                os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
                with open(output_json_path, 'w') as f:
                    json.dump(output_json, f, indent=2)
                
                print(f"OCR Tool: Successfully saved formatted answers to {output_json_path}")
            
            except Exception as e:
                return {
                    'success': False,
                    'error': f"Failed to save JSON file: {str(e)}"
                }

            # 5. Return the final JSON object to the agent
            return output_json
        
        except Exception as e:
            return {
                'success': False,
                'error': f"Error during OCR analysis: {str(e)}"
            }