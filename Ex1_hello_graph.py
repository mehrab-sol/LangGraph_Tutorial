from typing import TypedDict
from langgraph.graph import StateGraph, START, END

# 1 - State : every node reads from here and writes partial updates back
class State(TypedDict):
    name: str
    greeting: str


# 2 - node function [receives full state and returns only the keys it changed]
def greet_node(state: State) -> dict:
    print(f"Name received = '{state['name']}'")
    return {"greeting": f"Hello, {state['name']}! Welcome to LangGraph."}


# 3 - building the graph
graph = StateGraph(State)
graph.add_node("greet", greet_node) #node register
graph.add_edge(START, "greet") # start point
graph.add_edge("greet", END)  # end point


# 4 - compile the graph [It validate the structure & returns a runnable app]
app = graph.compile()



# 5 - Invoke [Pass the initial state & get the final state back]
if __name__ == "__main__":
    result = app.invoke({"name": "Mehrab", "greeting": ""})
    print("\n---------- Final State ----------")
    print(f"Name: {result['name']}")
    print(f"Greeting: {result['greeting']}")

    for name in ["Alison", "Bob", "Burgers"]:
        output = app.invoke({'name': name, 'greeting': ""})
        print(output["greeting"])