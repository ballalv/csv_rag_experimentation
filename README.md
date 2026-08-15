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
