from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph
from state import AgentState

llm = ChatOpenAI(model = "gpt-4o")

# node-1: calling the LLM
def call_llm(state: AgentState) -> dict:
    response = llm.invoke(state["messages"])
    return {"messages": [response]} # Only will return what's been changed

# node-2: validate the response
def validate_response(state: AgentState) -> dict:
    last_msg = state["messages"][-1].content
    if "error" in last_msg.lower():
        return {"current_step": "retry", "retry_count": state["retry_count"]+1}
    return {"current_step": "Done", "final_answer": last_msg}


# wiring nodes into graph
graph = StateGraph(AgentState)
graph.add_node("call_llm", call_llm)
graph.add_node("validate", validate_response)