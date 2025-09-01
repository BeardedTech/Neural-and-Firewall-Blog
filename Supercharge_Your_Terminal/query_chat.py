# query_chat.py

# Requirement:
# pip install instructor

import subprocess
import os
from pydantic import Field
from instructor import OpenAISchema

class Function(OpenAISchema):
    """
    Pass the (question) to get related documents that are returned as output (result)
    """
    question: str = Field(..., example="from my notes how to create golden ticket?", descriptions="user query to pass as the question",)

    class Config:
        title = "chromadb_query"

    @classmethod
    def execute(cls, question: str) -> str:
        script_path = "~/.config/shell_gpt/query.py" # <-- Change this path if necessary
        command = ["python3", script_path, question] # <-- Change python3 to python if necessary
        result = subprocess.run(command, capture_output=True, text=True)

        # Debugging output to verify command execution
        print(f"Running command: {command}")
        print(f"Command output: {result.stdout}")
        print(f"Command error (if any): {result.stderr}")

        # Check if the command was successful 
        if result.returncode == 0:
            return result.stdout.strip()
        else:
            return f"Error: {result.stderr.strip()}"
# Function to pretty print documents 
def pretty_print_docs(docs):
    print("Debug: Entered pretty_print_docs")
    if not docs:
        print("No documents found.")
    else:
        split_docs = docs.split('\n----------------------------------------------------------------------------------------------------\n')
        print(f"Debug: Number of documents found: {len(split_docs)}")
        for i, doc in enumerate(split_docs):
            doc_content = doc.strip()
            # Skip the lines containing metadata 
            if any(keyword in doc_content for keyword in ["Received question", "Retrieved"]): 
                continue
            # Check if the document is not empty before printing
            if doc_content:
                print(f"Document {i+1}:\n\n{doc}\n{'-' * 100}")
# Example usage 
if __name__ == "__main__": 
    # Sample question 
    test_question = input("Please enter your question: ")
    # Call the function 
    output = Function.execute(test_question)
    # Print the output print
    print("Output from Function.execute:")
    print(output)
    # Pretty print the retrieved documents
    pretty_print_docs(output)