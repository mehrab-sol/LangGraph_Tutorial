from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
import os
from dotenv import load_dotenv

# api key config
load_dotenv()
api_key=os.getenv("OPENROUTER_API_KEY").strip()

if not api_key:
    raise ValueError("API key not found in the directory or the key is not valid Or expired!")


# 1 - connect the LLM

llm = ChatOpenAI(
    model = 'minimax/minimax-m2.5:free',
    base_url ="https://openrouter.ai/api/v1",
    api_key = api_key,
    temperature = 0.7,
)

# state
class State(TypedDict):
    question: str   # user input question
    category: str   # technical | General
    answer: str     # final answer 


# node_1 [LLM classify the questions]
def classify_node(state: State) -> dict:
    prompt = f"""Classify this question as either "technical" or "general".
Reply with ONLY one word: technical or general.
 
Question: {state['question']}"""
    
    response = llm.invoke([HumanMessage(content = prompt)])
    category = response.content.strip().lower()

    if "tech" in category:
        category = "technical"
    else:
        category = "genral"

    print(f"[Classify_node] -> {category}")
    return {"category": category}


# node_2_a [Technical question node]
def tech_node(state:State) -> dict:
    print(f"Answering the technical question.....")
    prompt = f"""You are a senior software engineer and ML expert.
Answer this technical question clearly and precisely in 3-4 sentences:
{state['question']}"""
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"answer": response.content}


# node_2_b [General question node]
def general_node(state:State) -> dict:
    print(f"Answering the General question.....")
    prompt = f"""You are a friendly assistant.
Answer this question in a warm, conversational tone in 2-3 sentences:
{state['question']}"""
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"answer": response.content}


# Router function
def route_by_category(state: State) -> str:
    if state['category'] == "technical":
        return "tech"
    return "general"



# Building the graph
graph = StateGraph(State)

graph.add_node("classify",classify_node)
graph.add_node("tech", tech_node)
graph.add_node("general", general_node)

graph.add_edge(START, "classify")

graph.add_conditional_edges(
    "classify",
    route_by_category,
    {
        "tech" : "tech",
        "general" : "general",
    }
)

graph.add_edge("tech", END)
graph.add_edge("general", END)

app = graph.compile()


if __name__ == "__main__":
        questions = [
        "What is the difference between QLoRA and LoRA?",
        "What is the capital of USA?",
        "How does gradient checkpointing reduce VRAM usage?",
        "Can you tell me 5 random colors?",
        ]

        for q in questions:
            print(f"\n{'='*55}")
            print(f"Questions is: {q}")

            result = app.invoke({"question": q, "category": "", "answer": ""})

            print(f"Category: {result['category']}")
            print(f"A: {result['answer']}")
        

