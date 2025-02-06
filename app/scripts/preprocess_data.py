import logging

from app.config.base_config import DATA_DIR
from app.utils.chunking import Chunking
from app.utils.embeddings import Embeddings
from app.utils.utils import set_logger

set_logger(console=True)
logger = logging.getLogger(__name__)


def preprocess_data():
    """
    Preprocessing the data and storing the embeddings into the vector database.
    """
    try:
        logger.info("Starting data preprocessing")

        chunking = Chunking(data_dir=DATA_DIR)
        chunks = chunking.process_directory()

        embeddings = Embeddings(chunks=chunks)
        embeddings.process_chunks()

        logger.info("Data processing complete, embeddings stored.")

    except Exception as e:
        logger.error(f"Error during data preprocessing : {e}")
        raise e


if __name__ == "__main__":
    preprocess_data()
