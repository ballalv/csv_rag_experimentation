from langchain_community.document_loaders import CSVLoader

file_path = "../data/pharma_sales.csv"

loader = CSVLoader(file_path=file_path)

documents = loader.load()

print("Number of documents:", len(documents))

for doc in documents[:3]:
    print(doc)
    print("--------------------------------")