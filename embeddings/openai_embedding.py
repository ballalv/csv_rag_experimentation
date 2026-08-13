import os
import logging

from dotenv import load_dotenv

from langchain_community.document_loaders import CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma


# ============================================================
# 1. Load environment variables
# ============================================================

load_dotenv()


# ============================================================
# 2. Create logs directory
# ============================================================

LOG_DIR = "./logs"

os.makedirs(LOG_DIR, exist_ok=True)

LOG_FILE = os.path.join(
    LOG_DIR,
    "embedding_experiment.log"
)


# ============================================================
# 3. Configure logging
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(
            LOG_FILE,
            encoding="utf-8"
        ),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


# ============================================================
# 4. Validate API key
# ============================================================

if not os.getenv("OPENAI_API_KEY"):

    logger.error(
        "OPENAI_API_KEY is not set."
    )

    raise ValueError(
        "OPENAI_API_KEY is not set. "
        "Please add it to your .env file."
    )


logger.info("OpenAI API key detected.")


# ============================================================
# 5. Configuration
# ============================================================

FILE_PATH = "./data/pharma_sales.csv"

CHUNK_SIZES = [
    100,
    200,
    300,
    500,
    1000
]

CHROMA_BASE_PATH = "./chroma_db"

EMBEDDING_MODEL = "text-embedding-3-small"


# ============================================================
# 6. Load documents
# ============================================================

def load_documents():

    logger.info(
        "Loading CSV file: %s",
        FILE_PATH
    )

    try:

        loader = CSVLoader(
            file_path=FILE_PATH
        )

        documents = loader.load()

        logger.info(
            "Documents loaded successfully: %d",
            len(documents)
        )

        return documents

    except Exception:

        logger.exception(
            "Failed to load documents."
        )

        raise


# ============================================================
# 7. Create chunks
# ============================================================

def create_chunks(
    documents,
    chunk_size,
    chunk_overlap
):

    logger.info(
        "Creating chunks | size=%d | overlap=%d",
        chunk_size,
        chunk_overlap
    )

    try:

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

        chunks = text_splitter.split_documents(
            documents
        )

        logger.info(
            "Chunks created successfully | size=%d | overlap=%d | chunks=%d",
            chunk_size,
            chunk_overlap,
            len(chunks)
        )

        return chunks

    except Exception:

        logger.exception(
            "Chunking failed | size=%d | overlap=%d",
            chunk_size,
            chunk_overlap
        )

        raise


# ============================================================
# 8. Create OpenAI embedding model
# ============================================================

def create_embedding_model():

    logger.info(
        "Creating OpenAI embedding model: %s",
        EMBEDDING_MODEL
    )

    try:

        embedding_model = OpenAIEmbeddings(
            model=EMBEDDING_MODEL
        )

        logger.info(
            "Embedding model created successfully."
        )

        return embedding_model

    except Exception:

        logger.exception(
            "Failed to create OpenAI embedding model."
        )

        raise


# ============================================================
# 9. Store chunks and embeddings in ChromaDB
# ============================================================

def store_in_chroma(
    chunks,
    embedding_model,
    chunk_size
):

    collection_name = (
        f"openai_{chunk_size}"
    )

    persist_directory = os.path.join(
        CHROMA_BASE_PATH,
        collection_name
    )

    logger.info(
        "Starting ChromaDB storage | collection=%s",
        collection_name
    )

    logger.info(
        "Persist directory: %s",
        persist_directory
    )

    try:

        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embedding_model,
            collection_name=collection_name,
            persist_directory=persist_directory
        )

        logger.info(
            "Chunks and embeddings stored successfully | collection=%s",
            collection_name
        )

        return vectorstore

    except Exception:

        logger.exception(
            "Failed to store vectors in ChromaDB | collection=%s",
            collection_name
        )

        raise


# ============================================================
# 10. Main embedding experiment
# ============================================================

def run_embedding_experiment():

    logger.info(
        "=================================================="
    )

    logger.info(
        "Starting OpenAI embedding experiment"
    )

    logger.info(
        "Embedding model: %s",
        EMBEDDING_MODEL
    )

    logger.info(
        "Chunk sizes: %s",
        CHUNK_SIZES
    )

    logger.info(
        "=================================================="
    )


    # --------------------------------------------------------
    # Load documents only once
    # --------------------------------------------------------

    documents = load_documents()


    # --------------------------------------------------------
    # Create embedding model only once
    # --------------------------------------------------------

    embedding_model = create_embedding_model()


    # --------------------------------------------------------
    # Run every chunk-size experiment
    # --------------------------------------------------------

    for chunk_size in CHUNK_SIZES:

        chunk_overlap = int(
            chunk_size * 0.10
        )

        logger.info(
            "--------------------------------------------------"
        )

        logger.info(
            "STARTING EXPERIMENT | chunk_size=%d | overlap=%d",
            chunk_size,
            chunk_overlap
        )

        try:

            # Create chunks

            chunks = create_chunks(
                documents=documents,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )


            # Store chunks + embeddings in Chroma

            vectorstore = store_in_chroma(
                chunks=chunks,
                embedding_model=embedding_model,
                chunk_size=chunk_size
            )


            # Verify Chroma collection count

            collection_count = (
                vectorstore._collection.count()
            )

            logger.info(
                "Vectors stored in Chroma: %d",
                collection_count
            )

            logger.info(
                "Expected chunks: %d",
                len(chunks)
            )


            # Validate counts

            if collection_count != len(chunks):

                logger.error(
                    "COUNT MISMATCH | expected=%d | stored=%d",
                    len(chunks),
                    collection_count
                )

                raise RuntimeError(
                    f"Chroma count mismatch for chunk size "
                    f"{chunk_size}: expected {len(chunks)}, "
                    f"got {collection_count}"
                )


            logger.info(
                "VALIDATION PASSED | chunk_size=%d",
                chunk_size
            )

            logger.info(
                "EXPERIMENT COMPLETED | chunk_size=%d",
                chunk_size
            )


        except Exception:

            logger.exception(
                "EXPERIMENT FAILED | chunk_size=%d",
                chunk_size
            )

            # Continue with the next chunk size

            continue


    logger.info(
        "=================================================="
    )

    logger.info(
        "All embedding experiments completed."
    )

    logger.info(
        "=================================================="
    )


# ============================================================
# 11. Run experiment
# ============================================================

if __name__ == "__main__":

    run_embedding_experiment()