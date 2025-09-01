# qury.py

# Requirements:
# pip install langchain
# pip install langchain-chroma 
# pip install langchain-openai


import subprocess
import argparse
import os
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.retrievers import ContextualCompressionRetriever 
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_openai import OpenAI

# Path to the persist directory
persist_directory = 'Path/To/Chroma/Database' # <-- Change this path
collection_name = "your_collection" # <-- Change this to your collection's name

# Initialize the embedding model
embedding = OpenAIEmbeddings()

# Initialize the Chroma vector store with the embedding function and persist_directory
vectordb = Chroma(
    collection_name=collection_name,
    persist_directory=persist_directory,
    embedding_function=embedding
)

# Initialize the language model
llm = OpenAI(temperature=0, model="gpt-3.5-turbo-instruct")

# Initialize the compressor
compressor = LLMChainExtractor.from_llm(llm)

#init base retriever
base_retriever=vectordb.as_retriever(search_kwargs={"k": 5})

# Function to pretty print documents
def pretty_print_docs(docs):
    for i, d in enumerate(docs):
        # Print the document index, metadata, and content
        print(f"Document {i+1}:")
        #print("Metadata:")
        for key, value in d.metadata.items():
            print(f"  {key}: {value}")
        #print("\nContent:")
        print(d.page_content)
        print("\n" + "-" * 100 + "\n")

# Main execution block
if __name__ == "__main__":
    # Set up argument parsing 
    parser = argparse.ArgumentParser(description="Query the vector store")
    parser.add_argument("question", type=str, help="The question to query the vector store with")
    args = parser.parse_args()
    # Get the question from command-line arguments 
    question = args.question
    # Debugging output to verify the received question 
    print(f"Received question: {question}") 
    # Retrieve relevant documents 
    retrieved_docs = base_retriever.invoke(question)
    # Debugging output 
    print(f"Retrieved {len(retrieved_docs)} documents.")
    # Pretty print the retrieved documents
    pretty_print_docs(retrieved_docs)