from typing import TypedDict, Annotated
import operator
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


UNCERTAINITY_WORDS = ["i'm not sure", "perhaps", "maybe", "i think", "possibly", "uncertain"]

# state
class State(TypedDict):
    question: str
    answer : str
    attempts: Annotated[list[str], operator.add]
    retry_count : int
    status: str


# node_1 [call the llm and count retry]
def answare_llm(state: State) -> dict:
    retry = state["retry_count"]

    if retry == 0:
        prompt = f"Answer this question: {state['question']}"
    else:
        prompt = (
            f"Your previous answer was uncertain. Answer with full confidence, "
            f"no hedging. Question: {state['question']}"
        )
    print(f"[Answer_node] Attempt: {retry + 1}...")
    response = llm.invoke([HumanMessage(content = prompt)])
    answer = response.content

    return {
        "answer" : answer,
        "attempts" : [answer],
        "retry_count" : retry + 1,
    }


# node_2 [checking the confidence of the answer]
def check_confidence_node(state: State) -> dict:
    answer_lower = state['answer'].lower()
    uncertain = any(word in answer_lower for word in UNCERTAINITY_WORDS)

    if uncertain and state["retry_count"] < 3:
        print(f"[Check-Node] Uncertain answer detcted -> retrying....")
        return {"status" : "needs_retry"}
    else:
        reason = "Max retry reached" if state['retry_count'] >=3 else "Confident Answer"
        print(f"[Check_node] {reason} -> Done")
        return {"status" : "Done"}
    

# Conditional routing
def route_confidence(state: State) -> str:
    return state["status"]



# Building graph
graph = StateGraph(State)

graph.add_node("answer", answare_llm)
graph.add_node("check_confidence", check_confidence_node)

graph.add_edge(START, "answer")
graph.add_edge("answer", "check_confidence")

graph.add_conditional_edges(
    "check_confidence",
    route_confidence,
    {
        "needs_retry" : "answer",
        "Done" : END,
    }
)


app = graph.compile()

if __name__ == "__main__":
    questions = [
        "What is the capital of Australia?",
        "How is the prince or navrnia?",
    ]
 
    for q in questions:
        print(f"\n{'='*55}")
        print(f"Q: {q}")
        result = app.invoke({
            "question": q,
            "answer": "",
            "attempts": [],
            "retry_count": 0,
            "status": "needs_retry",
        })
        print(f"\nFinal answer (after {result['retry_count']} attempt(s)):")
        print(result["answer"])
        print(f"\nAll attempts recorded in state: {len(result['attempts'])}")

