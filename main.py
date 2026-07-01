from typing import List
import json
import random
import string
from datetime import datetime, timedelta 
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from dotenv import load_dotenv
load_dotenv()

@tool
def write_json(filepath: str, data: dict) -> str:
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return f"Successfully wrote JSON data to '{filepath}' ({len(json.dumps(data))} characters)"
    except Exception as e:
        return f"Error writing JSON: {e}"

@tool
def read_json(filepath: str) -> str:
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return json.dumps(data, indent=2)
    except FileNotFoundError:
        return f"Error: File '{filepath}' not found"
    except json.JSONDecodeError as e:
        return f"Error: Invalid JSON in file: {str(e)}"
    except Exception as e:
        return f"Error reading JSON: {str(e)}"

@tool
def generate_sample_users(
    first_names: List[str],
    last_names: List[str],
    domains: List[str],
    min_age: List[str],
    max_age: List[str]
) -> str:
    if not first_names:
        return {"error": "first_names list cannot be empty"}
    if not last_names:
        return {"error":"last_names list cannot be emmpty"}
    if not domains:
        return {"error":"domains list cannot be empty"}
    if min_age > max_age:
        return {"error":"min_age cannot be greater than the max_age"}
    if min_age < 0 or max_age < 0:
        return {"error":"Age cannot be negative"}
    
    users = []
    count = len(first_names)
    
    for i in range(count):
        first = first_names[i]
        last = last_names[i]
        domain = domains[i]
        email = f"{first.lower()}.{last.lower()}@{domain.lower()}"

        user = {
            "id": i+1,
            "FirstName": first,
            "LastName": last,
            "Email": email,
            "username": f"{first.lower()}{random.randint(100,999)}",
            "age": random.randint(min_age,max_age),
            "RegisteredAt": (datetime.now() - timedelta(date = random.randint(1,365))).isoformat()
        }
        users.append(user)

        return {"users": users, "count": len(users)}
    
TOOLS = [write_json, read_json, generate_sample_users]

model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

system_message = (
    "You are a DataGen, a helpful assistant that generates sample data for applications."
    "To Generate users, you neeed: first_names (list), last_names (list), domaine (list), min_age max_age."
    "When asked to save users, first generate them with the tool, then immediately user write_json with the result."
    "if the user refers to 'those users'  from a previous request, ask them to specify the details again."
)

agent = create_react_agent(model, TOOLS, prompt=system_message)

def run_agent(user_input: str, history: List[BaseMessage]) -> AIMessage:
    try:
        result = agent.invoke(
            {"messages": history + [HumanMessage(content=user_input)]},
            config={"recursion_limit": 50}
        )
        return result["messages"][-1]
    except Exception as e:
        return AIMessage(content = f"Error: {str(e)}\n\nPlease try rephrasing your request or provide more specific details.")
    

    