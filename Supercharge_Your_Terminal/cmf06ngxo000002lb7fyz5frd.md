---
title: "Supercharge Your Terminal: ShellGPT + ChromaDB + LangChain for Context-Aware Automation"
seoTitle: "Supercharge Your Terminal with ShellGPT, ChromaDB & LangChain"
seoDescription: "Learn how to integrate ShellGPT with ChromaDB and LangChain to run smarter commands, query your personal notes, and automate workflows, all from the CLI"
datePublished: Sun Aug 31 2025 21:08:37 GMT+0000 (Coordinated Universal Time)
cuid: cmf06ngxo000002lb7fyz5frd
slug: supercharge-your-terminal-shellgpt-chromadb-langchain-for-context-aware-automation
cover: https://cdn.hashnode.com/res/hashnode/image/upload/v1756638161613/ddcee52f-3bdc-452a-95f1-c8618ee0baea.png
tags: langchain, chromadb, rag, shellgpt, ai-cli

---

# A Smarter Way to Work in the CLI

The command line has always been a place for power users — fast, flexible, and unforgiving. But what if your terminal could do more than just run commands? What if it could *understand* your intent, execute the right actions, and even pull answers from your own notes before acting?

In this guide, we’ll start with ShellGPT — an AI-powered CLI companion that can chat, generate commands, and execute them directly in your system.

And we won’t stop there. ShellGPT already comes with a built-in function called execute\_shell\_command, which allows it to run generated commands. We’ll add a brand‑new custom function call (tool) to its toolset — one that can query your knowledge store and return relevant documents directly inside your CLI session.

Then we’ll take it further by integrating ChromaDB and LangChain to add Retrieval‑Augmented Generation (RAG), wiring that custom tool to your notes so the terminal can reason over your personal knowledge base. The result is a context‑aware assistant that doesn’t just respond — it acts, informed by your own data.

# Meet ShellGPT: Your AI-Powered Command Line Companion

ShellGPT is a versatile command-line productivity tool powered by large language models, such as GPT-4. It enhances your terminal experience by intelligently generating shell commands, code snippets, and documentation—all without leaving the CLI. Designed to streamline workflows and reduce context switching, ShellGPT supports Linux, macOS, and Windows, and works with major shells such as Bash, Zsh, PowerShell, and CMD.

### Key Features:

* **Command Generation**: Instantly generate shell commands tailored to your OS and shell.
    
* **Code Assistance**: Use --code to generate or annotate code directly from the terminal.
    
* **Chat & REPL Modes**: Maintain conversational sessions or interactively explore ideas.
    
* **Function Calling**: Define and execute custom Python functions via GPT.
    
* **Role Customisation**: Create roles to tailor GPT responses for specific tasks.
    
* **Local Model Support**: Optionally connect to local LLMs, such as Ollama.
    

What enables ShellGPT to be a powerful tool for automation and system administration is that, when running with admin or sudo privileges, ShellGPT can also execute system-level commands through function calling.

<div data-node-type="callout">
<div data-node-type="callout-emoji">🗃</div>
<div data-node-type="callout-text">To explore ShellGPT in depth, including installation instructions, usage examples, and advanced configuration options, head over to the official <a target="_self" rel="noopener noreferrer nofollow" href="https://github.com/TheR1D/shell_gpt" style="pointer-events: none">ShellGPT GitHub repository</a>.</div>
</div>

Here is an example provided in the GitHub repo where ShellGPT uses function call to generate a system command and then executes the command:

![Code snippet sShellGPT is tasked to list files in the  folder. ShellGPT then generates the code to list files and then executes the code, thus revealing  and .](https://cdn.hashnode.com/res/hashnode/image/upload/v1756658729668/c5f3c8c8-0a0d-46a6-b60f-21e95afaa1e7.png align="center")

# Unlocking Context: ShellGPT + ChromaDB

Granting ShellGPT access to your personal notes and documents through a vector database, such as ChromaDB, enables it to become a context-aware assistant. Instead of relying solely on generic knowledge, it can now reason over your own data—tailored to your workflows, preferences, and domain expertise.

### Key Benefits:

* **Semantic Search Over Your Knowledge:** Retrieve relevant information from your notes using natural language queries.
    
* **Contextual Command Suggestions:** ShellGPT can generate commands or code snippets based on the content of your documents.
    
* **Conversational Recall:** Ask ShellGPT questions like “What did I note about Docker networking?” and get precise answers drawn from your own writing.
    
* **Enhanced Learning & Debugging:** Use your notes as a knowledge base to troubleshoot errors, explore concepts, or revisit tutorials—all within the CLI.
    
* **Privacy-Preserving Intelligence:** Since ChromaDB runs locally, your data stays on your machine—giving you control without sacrificing capability.
    

# Setup Guide: Wiring ShellGPT to Your Notes with ChromaDB & LangChain

### Step 1: Initialize ChromaDB and Store Your Notes

To enable semantic search over your documents, you'll first need to initialize ChromaDB and populate it with embeddings.

<div data-node-type="callout">
<div data-node-type="callout-emoji">💡</div>
<div data-node-type="callout-text">If you do not already have a ChromaDB vector store set up, you can visit <a target="_self" rel="noopener noreferrer nofollow" href="https://bohowhizz.hashnode.dev/from-markdown-to-meaning" style="pointer-events: none">From Markdown to Meaning: Turn Obsidian Notes into a Conversational AI</a> to learn an example of how to convert your markdown notes to embeddings stored in ChromaDB. (Refer Code Section 1 - 5)</div>
</div>

### Step 2: Install ShellGPT and Enable Function Calling

ShellGPT will act as your conversational interface.

* To install: [ShellGPT GitHub repository](https://github.com/TheR1D/shell_gpt)
    
* To enable Function Calling: [Function Calling](https://github.com/TheR1D/shell_gpt?tab=readme-ov-file#function-calling)
    

### Step 3: Build a LangChain Driver for Retrieval

Install the following required packages in your environment

```python
pip install langchain
pip install langchain-chroma 
pip install langchain-openai
```

The following Python driver will handle semantic queries over your ChromaDB collection, utilising LangChain’s retrievers to retrieve 5 documents that match the query.

```python
# query.py
# Save this in ~/.config/shell_gpt/

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
```

Save the above as a Python file, for example, query.py in your Shell-GPT Folder: `~/.config/shell_gpt`

To test the driver file:

```bash
python query.py “your question here based on ChromaDB documents”
```

Here is an example of the query Python driver retrieving 5 documents based on my question:

````plaintext
 > python3 query.py "how to use python for pentesting?"
Received question: how to use python for pentesting?
Retrieved 5 documents.
Document 1:
  path:---------------------------------------\Python for Pentest\10 Extra Challenges.md
Document Name: 10 Extra Challenges.md
Path: ----------------------------------------\Python for Pentest\10 Extra Challenges.md
Based on what we have covered in this room, here are a few suggestions about how you could expand these tools or start building your own using Python:
- UseDNSrequests to enumerate potential subdomains
- Build the keylogger to send the capture keystrokes to a server you built using Python
- Grab the banner of services running on open ports
- Crawl the target website to download .js library files included
- Try to build a Windows executable for each and see if they work as stand-alone applications on a Windows target
- Implement threading in enumeration and brute-forcing scripts to make them run faster

----------------------------------------------------------------------------------------------------

Document 2:
  path: ----------------------------------------\Buffer Overflow - TCM.md
Fuzzing Python script: 1.py
```
````

<div data-node-type="callout">
<div data-node-type="callout-emoji">💡</div>
<div data-node-type="callout-text">LangChain’s base retriever offers minimal filtering and ranking—it simply returns documents based on raw similarity scores. To implement an accurate and focused document retriever that uses contextual compression and re-ranking, refer to <a target="_self" rel="noopener noreferrer nofollow" href="https://bohowhizz.hashnode.dev/from-markdown-to-meaning#heading-code-section-6-implementing-a-query-system" style="pointer-events: none">https://bohowhizz.hashnode.dev/from-markdown-to-meaning#heading-code-section-6-implementing-a-query-system</a></div>
</div>

### Step 4: Create a Custom Function for ShellGPT

ShellGPT already supports function calling, and comes with built‑in tools like `execute_shell_command`. In this step, we’ll **add our own custom function** to its toolset — one that acts as a bridge between ShellGPT and the [`query.py`](http://qury.py) driver we built earlier.

This new function will:

* **Accept** a natural‑language question from ShellGPT.
    
* **Pass** that question to the [query.py](http://query.py) driver you built in Step 3.
    
* **Retrieve** relevant documents from your ChromaDB store.
    
* **Return** those documents so ShellGPT can use the LLM to craft a context‑aware answer.
    
    By wiring this in, you’re effectively teaching ShellGPT to “look things up” in your personal notes before answering — turning it into a context‑aware terminal assistant.
    

Install the following required package in your environment

```python
pip install instructor
```

```python
# query_chat.py
# Save this in ~/.config/shell_gpt/functions
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
```

Save the above as a Python file, for example, query\_chat.py in your Shell-GPT Folder: `~/.config/shell_gpt/functions`

### **How It Works**

1. **ShellGPT** receives your query in CHAT or REPL mode.
    
2. The chromadb\_query function runs [query.py](http://query.py) with your question.
    
3. [query.py](http://query.py) uses LangChain to pull the top 5 relevant documents from ChromaDB.
    
4. ShellGPT uses those documents as context to generate a tailored, informed answer.
    

# RAG Demo with ShellGPT

In the following example, I asked ShellGPT: **“How to use Python to create a port scanner?”** — but with the added context **“from my notes.”**

ShellGPT retrieved 5 relevant documents from my ChromaDB store and used OpenAI’s LLM to generate a tailored response based on that personal context. This showcases Retrieval-Augmented Generation (RAG) in action: instead of relying solely on general knowledge, ShellGPT reasons over my own notes to deliver a precise, informed answer.

````markdown
>sgpt --repl temp
Entering REPL mode, press Ctrl+C to exit.
>>> from my notes how to use python to create a port scanner?
Running command: ['python3', '------------/.config/shell_gpt/query.py', 'how to
use python to create a port scanner']
Command output: Received question: how to use python to create a port scanner
Retrieved 5 documents.
Document 1:
  path: ------------------------------\Python for Pentest\4 Port Scanner.md
Document Name: 4 Port Scanner.md
Path: ---------------------------------\Python for Pentest\4 Port Scanner.md
In this task, we will be looking at a script to build a simple port scanner.
The code:
```python
.
.
.
----------------------------------------------------------------------------------------------------


▌ @FunctionCall chromadb_query(question="how to use python to create a port scanner")

To create a simple port scanner in Python, you can use the socket library. Here's a basic example:


 import socket

 def scan_ports(ip, ports):
     open_ports = []
.
.
.
This script attempts to connect to each port in the specified range and lists the open ones. Adjust the IP and port
range as needed.
````

### **Before vs After RAG**

| **<mark>Without RAG</mark>** | **<mark>With RAG</mark>** |
| --- | --- |
| Generic answer from model’s training data | Answer grounded in my own notes |
| May miss my preferred tools or methods | Matches my documented workflows |
| No awareness of past experiments | Recalls exactly what I’ve done before |

<div data-node-type="callout">
<div data-node-type="callout-emoji">💡</div>
<div data-node-type="callout-text"><strong>Pro Tip</strong>: If you're using ShellGPT in <strong>CHAT</strong> or <strong>REPL</strong> mode, you can go beyond just reading the output. You can, for example, ask ShellGPT to save the generated portscanner.py file to your desktop and even execute it against a custom IP and port range—all from the terminal.</div>
</div>

<div data-node-type="callout">
<div data-node-type="callout-emoji">⚠</div>
<div data-node-type="callout-text"><strong>Security Note:</strong> Since ShellGPT can execute system-level commands, always review generated commands before running them. Avoid using elevated privileges unless absolutely necessary, and keep sensitive files or credentials out of its reach. For risky or unfamiliar operations, test in a sandbox or VM first to protect your main environment.</div>
</div>

---

If this integration sparked ideas for your own setup, I’d love to hear how you’re using ShellGPT to personalise your terminal experience. Whether you're querying notes, automating system tasks, or just experimenting with RAG workflows, the beauty lies in adapting these tools to your unique context.