# LangGraph Exercises

5 standalone scripts that build on each other, from zero LLM to full memory. Each takes under 10 minutes to run and understand.

---

## Setup

```bash
pip install langgraph langchain langchain-openai
```
For exercises 2–5, create a .env file and paste your `api_key` like this:

```python
OPENROUTER_API_KEY = YOUR_KEY_HERE
or
Your_API_key_provider = YOUR_KEY_HERE
```
And change the ` api_key=os.getenv("Your_API_key_provider").strip()`

---

## Exercises

### Ex 1 — Hello LangGraph `ex1_hello_graph.py`
No LLM needed. Defines a `TypedDict` state, one node, compiles and invokes. Run this first to confirm the install works.

```bash
python ex1_hello_graph.py
```

**Concepts:** `StateGraph`, `TypedDict`, `add_node`, `add_edge`, `compile`, `invoke`

---

### Ex 2 — Two nodes + OpenRouter LLM `ex2_two_nodes_llm.py`
Node 1 builds a prompt, node 2 calls `minimax-m2.5` via `ChatOpenAI + base_url`. State carries data between them.

```bash
python ex2_two_nodes_llm.py
```

**Concepts:** `ChatOpenAI` with OpenRouter, multi-node pipeline, state handoff between nodes

---

### Ex 3 — Conditional edges `ex3_conditional_edges.py`
LLM classifies the question as `technical` or `general`, then a router function routes to the right specialist node.

```bash
python ex3_conditional_edges.py
```

**Concepts:** `add_conditional_edges`, router function `(state) → str`, `path_map`

---

### Ex 4 — Retry loop `ex4_retry_loop.py`
If the LLM answers with uncertainty words, the graph loops back and retries — up to 3 times. Uses `Annotated[list, operator.add]` to log every attempt without overwriting.

```bash
python ex4_retry_loop.py
```

**Concepts:** cycles, `retry_count` guard, `operator.add` reducer, loop edge

---

### Ex 5 — Multi-turn memory `ex5_memory.py`
`MemorySaver` checkpoints state after every node. Multiple `invoke()` calls on the same `thread_id` share history. Two threads stay completely isolated from each other.

```bash
python ex5_memory.py
```

**Concepts:** `MemorySaver`, `thread_id`, `MessagesState`, `get_state()`, multi-turn conversation

---

## How these map to the code reviewer project

| Exercise | Code reviewer equivalent |
|---|---|
| Ex 1 — state + single node | Every node in `nodes.py` |
| Ex 2 — two-node pipeline | `analyze_code` → `generate_patch` |
| Ex 3 — conditional routing | `approve` / `reject` / `edit` branch in `graph.py` |
| Ex 4 — retry loop | `regenerate` → `human_gate` loop |
| Ex 5 — memory + thread_id | `MemorySaver` + `interrupt_before` in `graph.py` |