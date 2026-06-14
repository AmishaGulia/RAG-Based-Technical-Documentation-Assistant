# LangGraph — Complete Guide

## What is LangGraph?

LangGraph is a library for building stateful, multi-actor applications with LLMs, used to create agent and multi-agent workflows. It is built on top of LangChain and extends it with graph-based orchestration.

Key differentiators from simple chains:
- **State persistence**: graph state is maintained across nodes
- **Cycles**: supports loops and retry logic (unlike DAG-only approaches)
- **Human-in-the-loop**: pause execution and resume after human approval
- **Streaming**: stream tokens and state updates in real time
- **Controllability**: fine-grained control over agent flow

---

## Core Concepts

### StateGraph

The primary graph type in LangGraph. You define:
1. A **state schema** (TypedDict or Pydantic model)
2. **Nodes** (Python functions that receive and return state)
3. **Edges** (connections between nodes, optionally conditional)

```python
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict

class State(TypedDict):
    messages: list[str]
    count: int

graph = StateGraph(State)
```

### Nodes

Nodes are Python functions (sync or async) that take state and return a dict of updates:

```python
def my_node(state: State) -> dict:
    # Read from state
    count = state["count"]
    # Return partial update
    return {"count": count + 1}

graph.add_node("my_node", my_node)
```

### Edges

Simple edges connect one node to the next unconditionally:

```python
graph.add_edge(START, "my_node")
graph.add_edge("my_node", END)
```

### Conditional Edges

Route to different nodes based on state:

```python
def router(state: State) -> str:
    if state["count"] > 5:
        return "stop"
    return "continue"

graph.add_conditional_edges(
    "my_node",
    router,
    {"stop": END, "continue": "my_node"},  # maps return values to node names
)
```

---

## Building a Complete Graph

```python
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict

class AgentState(TypedDict):
    input: str
    result: str
    iterations: int

def process(state: AgentState) -> dict:
    return {
        "result": state["input"].upper(),
        "iterations": state["iterations"] + 1,
    }

def should_retry(state: AgentState) -> str:
    if state["iterations"] < 3 and len(state["result"]) < 10:
        return "retry"
    return "done"

builder = StateGraph(AgentState)
builder.add_node("process", process)
builder.add_edge(START, "process")
builder.add_conditional_edges("process", should_retry, {"retry": "process", "done": END})

graph = builder.compile()

# Invoke
result = graph.invoke({"input": "hello", "result": "", "iterations": 0})
print(result)
```

---

## State Schema Design

### TypedDict (recommended for most cases)

```python
from typing_extensions import TypedDict
from typing import Annotated

def add(a: list, b: list) -> list:
    return a + b  # custom reducer

class State(TypedDict):
    messages: Annotated[list, add]  # append-only via reducer
    question: str
    answer: str
    retry_count: int
```

### Reducers

Reducers define how state fields are updated when multiple nodes write to them:

```python
from operator import add as add_op
from typing import Annotated

class State(TypedDict):
    # Default: last writer wins
    answer: str
    # Custom reducer: append new messages
    messages: Annotated[list, add_op]
```

---

## Checkpointing (Persistence)

Save graph state between runs:

```python
from langgraph.checkpoint.memory import MemorySaver

checkpointer = MemorySaver()
graph = builder.compile(checkpointer=checkpointer)

# Run with a thread ID for persistence
config = {"configurable": {"thread_id": "session-123"}}
result = graph.invoke({"input": "hello"}, config=config)

# Resume the same thread
result2 = graph.invoke({"input": "world"}, config=config)
```

### SQLite checkpointer (for production)

```python
from langgraph.checkpoint.sqlite import SqliteSaver

with SqliteSaver.from_conn_string("checkpoints.db") as checkpointer:
    graph = builder.compile(checkpointer=checkpointer)
```

---

## Human-in-the-Loop

Interrupt execution and wait for human input:

```python
graph = builder.compile(
    checkpointer=checkpointer,
    interrupt_before=["sensitive_action"],  # pause BEFORE this node
)

# First run — pauses before sensitive_action
result = graph.invoke(input_state, config)
# result["__interrupt__"] contains the pause point

# Human reviews and approves...
# Resume
result = graph.invoke(None, config)  # None = resume from checkpoint
```

---

## Streaming

Stream state updates as the graph executes:

```python
for chunk in graph.stream(initial_state):
    for node_name, node_output in chunk.items():
        print(f"Node '{node_name}' produced: {node_output}")
```

Stream individual LLM tokens:

```python
async for event in graph.astream_events(initial_state, version="v2"):
    if event["event"] == "on_chat_model_stream":
        print(event["data"]["chunk"].content, end="", flush=True)
```

---

## Retry Logic Pattern

Track retries with a counter in state:

```python
class RAGState(TypedDict):
    question: str
    documents: list
    retry_count: int
    max_retries: int

def route_after_retrieval(state: RAGState) -> str:
    if state["documents"]:
        return "generate"
    if state["retry_count"] < state["max_retries"]:
        return "retry"
    return "fallback"

def increment_retry(state: RAGState) -> dict:
    return {"retry_count": state["retry_count"] + 1}
```

---

## Async Nodes

Nodes can be async for I/O-bound operations:

```python
async def async_retrieval_node(state: State) -> dict:
    results = await vectorstore.asimilarity_search(state["query"])
    return {"documents": results}

graph.add_node("retrieval", async_retrieval_node)

# Invoke asynchronously
result = await graph.ainvoke(initial_state)
```

---

## Subgraphs

Compose graphs from other graphs:

```python
sub_graph = sub_builder.compile()

# Use subgraph as a node in the parent graph
parent_builder.add_node("sub_workflow", sub_graph)
```

---

## Error Handling

```python
def safe_node(state: State) -> dict:
    try:
        result = risky_operation(state)
        return {"result": result, "error": None}
    except Exception as e:
        return {"result": None, "error": str(e)}

def route_on_error(state: State) -> str:
    if state.get("error"):
        return "handle_error"
    return "continue"
```

---

## Multi-Agent Patterns

### Supervisor pattern

```python
def supervisor(state: MultiAgentState) -> dict:
    # Decide which agent to call next based on the task
    next_agent = llm.invoke(supervisor_prompt.format(**state))
    return {"next": next_agent}

graph.add_conditional_edges(
    "supervisor",
    lambda s: s["next"],
    {"agent_a": "agent_a", "agent_b": "agent_b", "FINISH": END},
)
```
