import numpy as np
import openai 
import os
from langchain.embeddings import OpenAIEmbeddings

def initialize_embedding_model():
    openai.api_key = os.getenv("OPENAI_API_KEY")
    headers = {
        "x-api-key": openai.api_key,
    }
    encoder = OpenAIEmbeddings(
        deployment="text-embedding-ada-002-v2", headers=headers, chunk_size=1
    )

    return encoder

def generate_corpus_embeddings(args: object, dataloader) -> np.ndarray:
    """
        Generates embeddings for the corpus of questions in the dataset
        Args:
            args: arguments passed to the program
        Returns:
            embeddings: embeddings for the corpus of questions in the dataset
    """
    
    corpus = [example['question'] for example in dataloader]
    encoder = initialize_embedding_model()
    embeddings = np.array(encoder.embed_documents(corpus))
    return embeddings