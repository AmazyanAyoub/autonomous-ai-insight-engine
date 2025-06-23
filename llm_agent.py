import json
import time 

from dotenv import load_dotenv, find_dotenv

from langgraph.graph import END, StateGraph, MessagesState
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import tools_condition, ToolNode


from langchain_core.tools import tool
from langchain_core.messages import SystemMessage

from langchain_groq import ChatGroq

from retriever import get_vectorstore

from db import SessionLocal, QueryLog

_ = load_dotenv(find_dotenv())


llm = ChatGroq(model="llama3-70b-8192", temperature=0)
vectordb = get_vectorstore()

@tool
def retrieve(query: str):
    """Retrieve information related to a query."""
    print("==== [RETRIEVE] ====")
    docs = vectordb.similarity_search(query, k=3)

    sources = []
    full_content = []

    for doc in docs:
        src = doc.metadata.get("source", "unknown")
        sources.append(src)
        full_content.append(f"[Source: {src}]\n{doc.page_content}")
    return {
        "content": "\n\n".join(full_content),
        "sources": sources
    }

def query_or_respond(state: MessagesState):
    """Generate tool call for retrieval or respond."""
    print("==== [QUERY OR RESPOND] ====")
    llm_with_tools = llm.bind_tools([retrieve])
    response = llm_with_tools.invoke(state["messages"])

    return {"messages": [response]}

tools = ToolNode([retrieve])

generate_prompt = """
# Role
You are a highly skilled and trustworthy **retrieval-based question answering assistant**. Your role is to answer user questions **using only the provided documents**. You do not rely on your own general knowledge or assumptions. You answer strictly based on the retrieved context and never guess.

# Task
Answer each user question using only the provided context documents by following this step-by-step process:
1. Carefully **read and understand the user’s question**.
2. Analyze the provided context documents for **relevant information**.
3. If the answer can be found in the documents, **quote or paraphrase** the information clearly and accurately.
4. If the answer **cannot** be found in the documents, respond clearly that the answer is not available in the provided context.
5. Never guess or generate information that is not grounded in the retrieved text.

# Specifics
- This task is extremely important to the **accuracy, reliability, and safety** of our RAG system — your precision is deeply appreciated.
- You must act with integrity and caution. If an answer is **not present**, clearly communicate that instead of fabricating one.
- Be concise but thorough. Use quotes when appropriate.
- Your honest judgment protects our users and builds trust in our system.

# Context
Our company uses a LangGraph-based RAG system that passes **retrieved documents** into this function. The function must generate a final answer that:
- Is **only** based on the content of the retrieved docs
- **Refuses** to answer if the content is not present
- **Does not hallucinate** or generate based on prior messages or general model knowledge

This function is part of our mission to provide **grounded**, **honest**, and **context-aware** AI answers.  
The variable `docs` contains the list of relevant context documents retrieved from our vector store.
The following are the retrieved documents you must use as your only source of truth:

{docs_content}

# Examples
## Example 1
Q: What year was the product AlphaX first launched?  
Docs:  
- "AlphaX was released in 2019 and quickly gained popularity due to its advanced features."  
A: AlphaX was first launched in 2019.

## Example 2
Q: What is the current CEO's name of our company?  
Docs:  
- "The company was founded by Samir Haddad and later expanded into Europe."  
A: Sorry, I could not find the answer in the provided context.

## Example 3
Q: Does the report mention anything about carbon emissions in 2023?  
Docs:  
- "The 2023 sustainability report showed a 12% drop in carbon emissions compared to 2022."  
A: Yes, the 2023 sustainability report mentions a 12% drop in carbon emissions compared to 2022.

## Example 4
Q: What are the three main pillars of the initiative?  
Docs:  
- "The initiative is based on three pillars: transparency, collaboration, and innovation."  
A: The three main pillars of the initiative are transparency, collaboration, and innovation.

## Example 5  
Q: Who won the Nobel Prize in Chemistry in 2021?  
Docs:  
- "This document focuses on the economic impact of vaccine distribution in 2021."  
A: Sorry, I could not find the answer in the provided context.

# Notes
- Do **not** use external knowledge under any circumstance.
- Always return factual responses **only** grounded in the provided documents.
- If the answer is not explicitly or implicitly found in the docs, respond:  
  **"Sorry, I could not find the answer in the provided context."**
- Do not attempt to guess or fabricate missing information.
- Trust only the given context — that’s your sole source of truth.
"""

def generate(state: MessagesState):
    """Generate answer."""
    print("==== [GENERATE] ====")
    recent_tool_messages = []
    for message in reversed(state["messages"]):
        if message.type == "tool":
            recent_tool_messages.append(message)
        else:
            break
    tool_messages = recent_tool_messages[::-1]

    docs_content = "\n\n".join(doc.content for doc in tool_messages)
    system_message_content = (
        generate_prompt.format(docs_content=docs_content)
    )
    conversation_messages = [
        message
        for message in state["messages"]
        if message.type in ("human", "system")
        or (message.type == "ai" and not message.tool_calls)
    ]
    prompt = [SystemMessage(system_message_content)] + conversation_messages

    response = llm.invoke(prompt)
    return {"messages": [response]}


class WindowedMemorySaver(MemorySaver):
    def __init__(self, max_messages=20):
        super().__init__()
        self.max_messages = max_messages
    
    def save(self, state):
        messages = state["messages"]
        # Keep only last N messages
        state["messages"] = messages[-self.max_messages:]
        super().save(state)

def build_graph():
    graph_builder = StateGraph(MessagesState)

    graph_builder.add_node(query_or_respond)
    graph_builder.add_node(tools)
    graph_builder.add_node(generate)

    graph_builder.set_entry_point("query_or_respond")
    graph_builder.add_conditional_edges(
        "query_or_respond",
        tools_condition,
        {END: END, "tools": "tools"},
    )
    graph_builder.add_edge("tools", "generate")
    graph_builder.add_edge("generate", END)

    graph = graph_builder.compile(checkpointer=WindowedMemorySaver())

    return graph

def extract_sources(state: MessagesState):
    sources = []
    for msg in state["messages"]:
        if msg.type == "tool":
            try:
                content_dict = eval(msg.content) if isinstance(msg.content, str) else msg.content
                for src in content_dict.get("sources", []):
                    if src not in sources:
                        sources.append(src)
            except Exception:
                continue
    return sources

graph = build_graph()

def run_query(user_query: str, config):
    state = {"messages": [{"role": "user", "content": user_query}]}
    final_state = graph.invoke(state, config=config)
    final_msg = final_state["messages"][-1]
    return {
        "answer": final_msg.content,
        "sources": extract_sources(final_state)
    }

from typing import Generator

def generate_token_stream(query: str, delay: float = 0.05) -> Generator[dict, None, None]:
    state = {"messages": [{"type": "human", "content": query}]}
    config = {"configurable": {"thread_id": "stream-thread"}}

    for token_chunk, metadata in graph.stream(
        state, config=config, stream_mode="messages"
    ):
        time.sleep(delay)
        yield {"event": "token", "data": token_chunk.content}
    
    yield {"event": "end", "data": "[DONE]"}

def stream_query_notebook(query: str, delay: float = 0.05):
    for chunk in generate_token_stream(query, delay=delay):
        print(chunk["data"], end="", flush=True)
    print()


