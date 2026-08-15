# CSV RAG Experimentation

An experiment-driven RAG project built with **LangChain, OpenAI Embeddings, ChromaDB, and GPT-4.1-mini**.

The project explores how chunking, embeddings, and retrieval strategies affect RAG performance on pharmaceutical sales data, and uses the selected configuration to build an end-to-end RAG pipeline with LCEL.

## Architecture

```text
CSV Data
   ↓
Document Ingestion
   ↓
Chunking Experiments
   ↓
Embedding Experiments
   ↓
Chroma Vector Store
   ↓
Similarity Search
   ↓
RAG + LCEL
   ↓
Grounded Answer
Current Baseline
Component	Selected Configuration
Chunking	500 / 50
Embedding	text-embedding-3-large
Vector Store	ChromaDB
Retrieval	Similarity Search
Top K	3
LLM	gpt-4.1-mini
Framework	LangChain + LCEL
Project Structure
├── data/                 # Dataset
├── notebooks/            # Experiments & RAG pipeline
├── src/                  # Reusable application code
├── experiments/         # Experiment logs & results
├── docs/                # Architecture & design decisions
├── requirements.txt
├── .env.example
└── README.md
Experiments
01 → Data Ingestion
02 → Chunking Experiments
03 → Embedding Experiments
04 → Similarity Search
05 → RAG with LCEL

Each experiment is recorded with its configuration, observations, and conclusions.

Setup
pip install -r requirements.txt

Create .env:

OPENAI_API_KEY=your_api_key

Run the notebooks in order:

01 → 02 → 03 → 04 → 05
Goal

Build a well-evaluated and explainable RAG pipeline while documenting the reasoning behind each architectural decision.

Experiment first. Decide with evidence. Build the RAG pipeline.
