import os
from typing import TypedDict, Annotated, Sequence
import operator
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

# Define Agent State
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]

# Initialize LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Define Node Function
def call_model(state: AgentState):
    messages = state['messages']
    response = llm.invoke(messages)
    return {"messages": [response]}

# Build LangGraph Workflow
workflow = StateGraph(AgentState)
workflow.add_node("agent", call_model)
workflow.set_entry_point("agent")
workflow.add_edge("agent", END)

# Compile Application
app_agent = workflow.compile()