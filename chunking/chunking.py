from langchain_community.document_loaders import CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load CSV
file_path = "../data/pharma_sales.csv"

loader = CSVLoader(file_path=file_path)
documents = loader.load()

print("Number of documents:", len(documents))
print()

# Chunk-size experiments
chunk_sizes = [100, 200, 300, 500, 1000]

for chunk_size in chunk_sizes:

    chunk_overlap = int(chunk_size * 0.10)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    chunks = text_splitter.split_documents(documents)

    print(
        f"Chunk Size: {chunk_size} | "
        f"Overlap: {chunk_overlap} | "
        f"Documents: {len(documents)} | "
        f"Chunks: {len(chunks)}"
    )