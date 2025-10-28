# tools/file_tools.py
import os
from crewai.tools import BaseTool

class FileReaderTool(BaseTool):
    name: str = "File Reader Tool"
    description: str = "Reads the full content of a specified text file."

    def _run(self, file_path: str) -> str:
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            return content
        except FileNotFoundError:
            return f"Error: File not found at {file_path}"
        except Exception as e:
            return f"Error reading file: {str(e)}"

class FileWriterTool(BaseTool):
    name: str = "File Writer Tool"
    description: str = "Writes content to a specified file. Overwrites if the file exists."

    def _run(self, file_path: str, content: str) -> str:
        try:
            # Ensure the directory exists
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            with open(file_path, 'w') as f:
                f.write(content)
            return f"Successfully wrote content to {file_path}"
        except Exception as e:
            return f"Error writing file: {str(e)}"