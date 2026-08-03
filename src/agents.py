import os
from typing import TypedDict
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


router_llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.1, groq_api_key=GROQ_API_KEY)
reasoning_llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.3, groq_api_key=GROQ_API_KEY)


class ResearchState(TypedDict):
    query: str
    context: str
    initial_findings: str
    final_response: str


def research_planner_agent(state: ResearchState):
    print("--- AGENT 1: PLANNING & EXTRACTING ---")
    prompt = f"""
    You are an expert Research Planning Agent. Analyze the user query and retrieved corpus context.
    Query: {state['query']}
    Context: {state['context'][:2500]}
    
    Extract key factual data points and draft an initial structured response.
    """
    response = router_llm.invoke([HumanMessage(content=prompt)])
    return {"initial_findings": response.content}


def academic_refinement_agent(state: ResearchState):
    print("--- AGENT 2: REFLECTION & SYNTHESIS ---")
    prompt = f"""
    You are a Senior Academic Reviewer and Synthesizer Agent. 
    Review the initial findings provided by the Research Agent, check for clarity, and refine it into a formal academic response.
    
    User Query: {state['query']}
    Initial Findings: {state['initial_findings']}
    Context Reference: {state['context'][:2500]}
    
    Provide the final polished academic response:
    """
    response = reasoning_llm.invoke([HumanMessage(content=prompt)])
    return {"final_response": response.content}


workflow = StateGraph(ResearchState)


workflow.add_node("research_planner", research_planner_agent)
workflow.add_node("academic_refinement", academic_refinement_agent)


workflow.set_entry_point("research_planner")
workflow.add_edge("research_planner", "academic_refinement")
workflow.add_edge("academic_refinement", END)


app_graph = workflow.compile()

def process_research_query(retrieved_context: str, query: str) -> str:
    """Executes the multi-agent graph workflow"""
    initial_state = {
        "query": query,
        "context": retrieved_context,
        "initial_findings": "",
        "final_response": ""
    }
    result = app_graph.invoke(initial_state)
    return result["final_response"]

def generate_novel_research_paper(retrieved_context: str) -> str:
    """Novel Research Paper Generator using Reasoning Model"""
    prompt = f"""
    You are an expert Academic Researcher and Paper Writer. 
    Based on the following retrieved context, generate a NEW NOVEL RESEARCH PAPER DRAFT.
    
    Structure required:
    1. TITLE
    2. ABSTRACT
    3. INTRODUCTION & PROBLEM STATEMENT
    4. LITERATURE REVIEW & SYNTHESIS
    5. PROPOSED NOVEL METHODOLOGY
    6. EXPECTED OUTCOMES & FUTURE WORK
    
    Context:
    {retrieved_context[:4000]}
    """
    res = reasoning_llm.invoke([HumanMessage(content=prompt)]).content
    return res