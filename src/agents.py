import os
import streamlit as st
from typing import TypedDict, Annotated, Sequence
from langchain_groq import ChatGroq
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import StateGraph, END

# Fetch API Key securely via Streamlit Secrets
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY", ""))

# -------------------------------------------------------------
# MULTI-MODEL SELECTION STRATEGY (Assignment Requirement)
# -------------------------------------------------------------
# Lightweight Model for Intent Classification / Routing
router_llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.0,
    groq_api_key=GROQ_API_KEY
)

# Reasoning Model for Synthesis & Reflection
reasoning_llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.2,
    groq_api_key=GROQ_API_KEY
)

# State Definition for LangGraph Communication
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], "Chat messages"]
    context: str
    intent: str
    draft_response: str
    final_response: str

# -------------------------------------------------------------
# AGENT 1: Router Agent (Pattern 1: Router)
# -------------------------------------------------------------
def router_agent(state: AgentState):
    user_query = state["messages"][-1].content
    prompt = f"Classify the query as 'SUMMARY' or 'QA'. Query: {user_query}. Answer strictly 'SUMMARY' or 'QA'."
    res = router_llm.invoke([HumanMessage(content=prompt)]).content.strip().upper()
    return {"intent": "SUMMARY" if "SUMMARY" in res else "QA"}

# -------------------------------------------------------------
# AGENT 2: Reasoning Agent (Pattern 2: Tool-use / RAG Context)
# -------------------------------------------------------------
def reasoning_agent(state: AgentState):
    user_query = state["messages"][-1].content
    # Truncate context to max 3000 chars to avoid Rate Limit Errors
    context = state.get("context", "")[:3000]
    intent = state.get("intent", "QA")
    
    if intent == "SUMMARY":
        prompt = f"Provide a structured summary (Objectives, Methodology, Key Findings, Conclusion) using this research context:\n\n{context}"
    else:
        prompt = f"Answer precisely using ONLY the retrieved research paper context below:\n\nContext:\n{context}\n\nQuery: {user_query}"
        
    res = reasoning_llm.invoke([HumanMessage(content=prompt)]).content
    return {"draft_response": res}

# -------------------------------------------------------------
# AGENT 3: Reflection Agent (Pattern 3: Reflection / Self-Critique)
# -------------------------------------------------------------
def reflection_agent(state: AgentState):
    draft = state.get("draft_response", "")
    prompt = f"Critique and refine this academic response for clarity and technical accuracy. Return ONLY the final output:\n\n{draft}"
    refined_res = reasoning_llm.invoke([HumanMessage(content=prompt)]).content
    return {"final_response": refined_res}

# -------------------------------------------------------------
# LANGGRAPH AGENT ORCHESTRATION
# -------------------------------------------------------------
workflow = StateGraph(AgentState)

workflow.add_node("RouterAgent", router_agent)
workflow.add_node("ReasoningAgent", reasoning_agent)
workflow.add_node("ReflectionAgent", reflection_agent)

workflow.set_entry_point("RouterAgent")
workflow.add_edge("RouterAgent", "ReasoningAgent")
workflow.add_edge("ReasoningAgent", "ReflectionAgent")
workflow.add_edge("ReflectionAgent", END)

agent_executor = workflow.compile()

def process_research_query(retrieved_context: str, user_query: str) -> str:
    """Executes the multi-agent pipeline and returns the refined answer."""
    initial_state = {
        "messages": [HumanMessage(content=user_query)],
        "context": retrieved_context,
        "intent": "",
        "draft_response": "",
        "final_response": ""
    }
    output = agent_executor.invoke(initial_state)
    return output["final_response"]