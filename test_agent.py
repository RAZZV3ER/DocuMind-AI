
from agent_engine import run_agent

from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()

# Initialize Groq client
groq_client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

# Model configuration
MODEL ="openai/gpt-oss-20b"

# Sample document text
document_text = """
Employee Name: Rahul Sharma
Annual CTC: Rs. 600000
Monthly Salary: Rs. 50000
Joining Date: 1 July 2026
Probation Period: 6 months
"""

# Multi-step agent task
task = (
    "Find the annual CTC in the document. "
    "You MUST call search_document first to retrieve "
    "the annual CTC from the document. "
    "Then you MUST call calculate to divide it by 12. "
    "Report both tool results and the final monthly amount."
)

# Run the agent
result = run_agent(
    task=task,
    document_text=document_text,
    groq_client=groq_client,
    model=MODEL
)

# Display final answer
print("\nFINAL ANSWER:")
print(result["answer"])

# Display execution trace
print("\nAGENT TRACE:")
for step in result["trace"]:
    print(step)

print("\nTOTAL STEPS:", result["steps"])
