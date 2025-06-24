# AI Agent for Loan Approval Queries
This project builds a simple, intelligent AI agent capable of answering questions about a loan approval dataset using OpenAI's GPT-4o model. It mimics the behavior of a data analyst by summarizing dataset structure and responding to queries in natural language.

## Features
- Automatic Data Summary using pandas

 -Natural Language Query Handling via OpenAI GPT-4o

- Context-Aware Responses to user questions

- Interactive Command-Line Interface

## Tech Stack
Python

OpenAI API (GPT-4o)

Pandas

## Setup
Install OpenAI:

bash
Copy
Edit
pip install openai
Add your API key in the script:

python
Copy
Edit
client = OpenAI(api_key='your_api_key')
Load your dataset:

python
Copy
Edit
df = pd.read_csv("path_to/trainn.csv")
Run the script and ask questions:

bash
Copy
Edit
python AI_agent_for_loan_approval.py
Example Questions
"How many applicants are there?"

"What is the average loan amount?"

"How many loans were approved?"
