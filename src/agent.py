import os 
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode, tools_condition

from tools import semantic_search, filter_by_category

load_dotenv()

# System Prompt
SYSTEM_PROMPT = """You are an expert HR Assistant helping recruiters find the right candidates.

You have access to a database of 2484 resumes across 24 job categories.

IMPORTANT RULES:
- ONLY answer based on resumes retrieved from the database
- NEVER make up or hallucinate candidate information
- If no relevant candidates are found, say so clearly
- Always present candidates in a clean, structured format
- For each candidate found, summarize their key skills and experience

When presenting candidates, use this format:
---
Candidate [N]:
- Category: [job category]
- Key Skills: [summarize from resume]
- Experience: [summarize from resume]
---
You can help with:
- Finding candidates by skills or experience (use semantic_search)
- Filtering candidates by job category (use filter_by_category)
"""

tools = [semantic_search, filter_by_category]

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
llm_with_tools = llm.bind_tools(tools)

def agent_node(state: MessagesState):
    messages = state["messages"]
    
    # tambah system prompt di awal
    if not any(isinstance(m, SystemMessage) for m in messages):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages
    
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

# build graph
builder = StateGraph(MessagesState)
builder.add_node("agent", agent_node)
builder.add_node("tools", ToolNode(tools))

builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", tools_condition)
builder.add_edge("tools", "agent")
builder.add_edge("agent", END)

graph = builder.compile()

def chat(user_input: str, history: list = []):
    history.append(HumanMessage(content=user_input))
    result = graph.invoke({"messages": history})
    history = result["messages"]
    return result["messages"][-1].content, history

if __name__ == "__main__":
    print("HR Talent Search Agent ready!")
    print("Type 'exit' to quit\n")
    
    history = []
    while True:
        user_input = input("HR: ")
        if user_input.lower() == "exit":
            break
        response, history = chat(user_input, history)
        print(f"\nAgent: {response}\n")