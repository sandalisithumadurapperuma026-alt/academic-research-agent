#  Academic Research Assistant (Multi-Agent RAG System)

##  Project Overview
The **Academic Research Assistant** is an advanced multi-agent system designed to streamline literature reviews, analyze research paper corpora, answer complex academic queries, and assist researchers in drafting novel research papers using State-of-the-Art Large Language Models (LLMs) and Retrieval-Augmented Generation (RAG).

---

##  Architecture Diagram
The following diagram illustrates the high-level architecture of the Multi-Agent RAG pipeline:

```text
[User Query / Upload PDF] ---> [Streamlit UI Dashboard]
                                         │
                                         ▼
                            [FAISS Vector Store & Embeddings]
                                         │
                                         ▼
                            [LangGraph Multi-Agent Workflow]
             ┌───────────────────────────┼───────────────────────────┐
             ▼                           ▼                           ▼
     [Retriever Agent]          [Analysis Agent]          [Synthesis / Writer Agent]
             └───────────────────────────┼───────────────────────────┘
                                         ▼
                         [Groq LLM Response Generation & PDF Export]
                         ---

##  Model & Component Comparison Table
| Component | Selected Model / Tool | Purpose / Functionality |
| :--- | :--- | :--- |
| **Large Language Model** | `llama3-70b-8192` (via Groq) | High-speed reasoning, context synthesis, and academic drafting. |
| **Embedding Model** | `all-MiniLM-L6-v2` (HuggingFace) | Efficient text embedding generation for semantic search. |
| **Vector Database** | `FAISS` | Fast, in-memory local vector storage and similarity search. |
| **Workflow Framework** | `LangGraph` & `LangChain` | Managing multi-agent states, retrieval, and generation steps. |
| **PDF Export Engine** | `ReportLab` | Formatting and compiling generated research papers into clean PDFs. |

---

##  Agent Communication & Workflow
1. **Corpus Ingestion:** PDFs are loaded via the sidebar, split into chunks, and indexed into the FAISS vector store.
2. **Retriever Agent:** Performs similarity search against user inputs to extract the most relevant context paragraphs from the loaded corpus.
3. **Analysis Agent:** Evaluates the retrieved academic content, identifying key methodologies, gaps, and findings.
4. **Synthesis / Writer Agent:** Synthesizes the final output to answer queries interactively or draft structured academic sections.
---

##  Sample Queries & Evaluation (Top 5)
| # | Sample Query | Evaluation / System Response |
| :--- | :--- | :--- |
| **1** | *"What is the primary methodology used in the corpus?"* | Successfully extracts core experimental setups and algorithms with high precision. |
| **2** | *"Provide a structured summary of the research corpus."* | Generates a categorized breakdown covering objectives, methods, and key takeaways. |
| **3** | *"What are the main limitations identified in these studies?"* | Accurately isolates limitation sections across multiple embedded PDF documents. |
| **4** | *"Draft a novel abstract based on the analyzed literature."* | Synthesizes professional academic text following standard journal abstract formats. |
| **5** | *"Explain the performance metrics discussed in the papers."* | Retains context effectively to present quantitative and qualitative results accurately. |

---

##  Installation & Local Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/sanalisithumadurapperuma026-alt/academic-research-agent.git](https://github.com/sanalisithumadurapperuma026-alt/academic-research-agent.git)
   cd academic-research-agent
