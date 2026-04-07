from typing import TypedDict, Annotated
import operator
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
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



# MessagesState [Built-in state that stores chat history and automatically merges new messages with previous ones]
SYSTEM_PROMPT = "You are a helpful ML assistant. Be concise — max 2 sentences per reply."

def chat_node(state:MessagesState) -> dict:
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]   # system_prompt + full message history
    response = llm.invoke(messages)
    return {"messages" : [response]} 

# Build Graph
graph = StateGraph(MessagesState)

graph.add_node("chat", chat_node)
graph.add_edge(START, "chat")
graph.add_edge("chat", END)


# Memory checkpoint
checkpoint = MemorySaver()
app = graph.compile(checkpointer=checkpoint)


# Thread id [History is autometic]
def chat(thread_id: str, user_message: str) -> str:
    config = {"configurable" : {"thread_id": thread_id}}
    result = app.invoke(
        {"messaes" : [HumanMessage(content=user_message)]},
        config=config,
    )
    return result["messages"][-1].content



# Show chat history
def show_history(thread_id: str):
    config = {"configurable": {"thread_id": thread_id}}
    state = app.get_state(config)
    messages = state.values.get("messages", [])
    print(f"\n [Thread ID-{thread_id}: {len(messages)} messages in memory]")

    for m in messages:
        role = "you " if m.type == "human" else "Bot "
        print(f"{role}: {m.content[:80]}....")

if __name__ == "__main__":
    
    # Session A: a continuous conversation about ML
    print("=== Session A: ML conversation ===")
    r1 = chat("session-a", "What is fine-tuning in ML?")
    print(f"Bot: {r1}\n")
 
    r2 = chat("session-a", "How is it different from training from scratch?")
    print(f"Bot: {r2}\n")
 
    r3 = chat("session-a", "What did I ask you first?")   # tests memory!
    print(f"Bot: {r3}\n")
 
    show_history("session-a")
 

    # Session B: completely separate thread — no shared memory
    print("\n=== Session B: separate thread ===")
    r4 = chat("session-b", "What did I ask you in session A?")   # should say "nothing"
    print(f"Bot: {r4}")
    show_history("session-b")
 
    # Resume session A — it still remembers!
    print("\n=== Session A resumed ===")
    r5 = chat("session-a", "Summarize our whole conversation in one sentence.")
    print(f"Bot: {r5}")