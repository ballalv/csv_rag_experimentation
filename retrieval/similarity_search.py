import os
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma


# ============================================================
# Configuration
# ============================================================

load_dotenv()

CHUNK_SIZE = 1000
TOP_K = 3

CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = f"openai_{CHUNK_SIZE}"

OUTPUT_FILE = "./retrieval/retrieval_results_size.txt"

EMBEDDING_MODEL = "text-embedding-3-small"


# ============================================================
# Validate API key
# ============================================================

if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY is not set.")


# ============================================================
# Create embedding model
# ============================================================

embedding_model = OpenAIEmbeddings(
    model=EMBEDDING_MODEL
)


# ============================================================
# Load existing Chroma collection
# ============================================================

vectorstore = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=embedding_model,
    persist_directory=f"{CHROMA_PATH}/{COLLECTION_NAME}"
)


# ============================================================
# Query
# ============================================================

query ="What is the quantity and sales value for WELIREG?"


# ============================================================
# Similarity Search - Top K
# ============================================================

results = vectorstore.similarity_search(
    query,
    k=TOP_K
)


# ============================================================
# Save results to file
# ============================================================

os.makedirs("./retrieval", exist_ok=True)

with open(OUTPUT_FILE, "a", encoding="utf-8") as file:

    file.write(f"Query: {query}\n")
    file.write(f"Chunk Size: {CHUNK_SIZE}\n")
    file.write(f"Chunk Overlap: {int(CHUNK_SIZE * 0.10)}\n")
    file.write(f"Top K: {TOP_K}\n")

    file.write("\n" + "=" * 80 + "\n")

    for i, doc in enumerate(results, start=1):

        file.write(f"\n--- Result {i} ---\n")
        file.write(f"Metadata: {doc.metadata}\n")
        file.write(f"Content:\n{doc.page_content}\n")

        file.write("\n" + "-" * 80 + "\n")


print("Similarity search completed.")
print(f"Results saved to: {OUTPUT_FILE}")