import os
import time
from crewai.tools import BaseTool
from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeResult
from pydantic import Field  # <-- Import Field
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

class AzureOCRTool(BaseTool):
    name: str = "Azure OCR Tool"
    description: str = "Performs OCR on images using Azure Document Intelligence"
    
    # Use Field and default_factory to initialize the client
    # Pydantic will call the _create_azure_di_client function
    # when an instance of AzureOCRTool is created.
    client: Any = Field(default_factory=_create_azure_di_client)

    # --- No __init__ method needed ---
    
    def _run(self, image_path: str) -> dict:
        """
        Perform OCR on image and return structured text with bounding boxes
        """
        # Add a check in case the client failed to initialize
        if not self.client:
            return {
                'success': False,
                'error': "Azure Document Intelligence client is not initialized. Check environment variables."
            }

        try:
            # Read image
            with open(image_path, "rb") as image_file:
                # Call the new "prebuilt-read" model
                poller = self.client.begin_analyze_document(
                    "prebuilt-read",
                    analyze_request=image_file,
                    content_type="application/octet-stream"
                )
            
            result: AnalyzeResult = poller.result()
            
            ocr_results = []
            if result.pages:
                for page in result.pages:
                    for line in page.lines:
                        # Extract text and bounding box
                        ocr_results.append({
                            'text': line.content,
                            'bounding_box': [
                                {'x': point.x, 'y': point.y} for point in line.polygon
                            ]
                        })
            
            return {
                'success': True,
                'lines': ocr_results,
                'full_text': result.content 
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }