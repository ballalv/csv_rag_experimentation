# Architecture

## 1. Overview

The project implements a CSV-based Retrieval-Augmented Generation (RAG) pipeline using LangChain.

The project also experiments with configurable parameters at different stages of the pipeline.

## 2. End-to-End Architecture

```text
                    CSV FILE
                       |
                       v
                 +-----------+
                 | CSVLoader |
                 +-----+-----+
                       |
                       v
              LangChain Documents
                       |
                       v
          RecursiveCharacterTextSplitter
                       |
                       v
                    Chunks
                       |
                       v
              Embedding Model
                       |
                       v
                  ChromaDB
                       |
                       v
               Similarity Search
                       |
                       v
                   Retriever
                       |
                       v
              Retrieved Documents
                       |
                       v
                  Context
                       |
                       v
                   Prompt
                       |
                       v
                 Chat Model
                       |
                       v
                    Answer