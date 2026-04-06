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


# 1 - State
class State(TypedDict):
    topic: str
    prompt: str  # node-1
    answer: str  # node-2



# 2 - node_1 [prompt formating]
def build_prompt_node(state: State) -> dict:
    print(f"[Node-1] -> Prompt for topic: '{state['topic']}'")
    prompt = f"Explain '{state['topic']}' in exactly 1 sentence. Be concise."
    return {"prompt": prompt}  # Only update the prompt


# 3 - node_2 [Call the LLM with the prompt from state]
def call_llm_node(state: State) -> dict:
    print(f"[Node-2] -> Calling the LLM...")
    response = llm.invoke([HumanMessage(content = state["prompt"])])
    return {"answer": response.content}


# 4 - graph structure
graph = StateGraph(State)

graph.add_node("build_prompt", build_prompt_node)
graph.add_node("call_llm", call_llm_node)

graph.add_edge(START, "build_prompt")
graph.add_edge("build_prompt", "call_llm") # node_1 -> node_2
graph.add_edge("call_llm", END)

app = graph.compile()


if __name__ == "__main__":
    topics = ["Langgraph", "QLoRA fine-tuning", "what a nural network is"]

    for topic in topics:
        print(f"\n{'='*50}")
        print(f"Topic: {topic}")
        result = app.invoke({"topic": topic, "prompt": "", "answer": ""})
        print(f"Answer: {result['answer']}")