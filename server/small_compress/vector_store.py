import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from typing import List
import time
import uuid
from .logger import Logger


class VectorStore:
    def __init__(self, logger: Logger):
        # injected logger (defaults to a new Logger for backward compatibility)
        self.logger = logger
        self.CHROMA_DATA_PATH = "chroma_data/"
        self.EMBED_MODEL = "all-MiniLM-L6-v2"
        self.COLLECTION_NAME = "documents"

        self.client = chromadb.PersistentClient(
            settings=Settings(persist_directory=self.CHROMA_DATA_PATH)
            )

        # Let Chroma handle embeddings via its embedding_function hook
        self.embedder = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=self.EMBED_MODEL)

        # Your original collection (now uses embedding_function so you don't manually embed)
        self.collection = self.client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            embedding_function=self.embedder,
            metadata={"hnsw:space": "cosine"}
        )

        self.chunk_collection = self.client.get_or_create_collection(
            "compressor_chunks",
            embedding_function=self.embedder,
            metadata={"hnsw:space": "cosine"},
        )

    def add_words(self, words: List[str]):
        ids = [f"word_{i}" for i in range(len(words))]
        # No manual embedding; Chroma uses embedding_function
        self.collection.add(ids=ids, documents=words)

    def query(self, word: str, n_results: int = 3):
        # No manual embedding; query_texts uses embedding_function
        return self.collection.query(query_texts=[word], n_results=n_results)

    def filter_chunks_by_context(
        self,
        context: str,
        chunks: List[str],
        similarity_cutoff: float,
    ) -> List[str]:
        """
        Keeps chunks whose cosine_similarity(context, chunk) >= similarity_cutoff.

        Because this collection is configured with cosine distance:
          distance = 1 - cosine_similarity
        So keep if:
          distance <= 1 - similarity_cutoff
        """
        if not chunks:
            return []

        run_id = f"{int(time.time() * 1000)}_{uuid.uuid4().hex}"
        ids = [f"{run_id}_{i}" for i in range(len(chunks))]

        # Add chunks (Chroma embeds via embedding_function)
        self.chunk_collection.add(ids=ids, documents=chunks)

        try:
            res = self.chunk_collection.query(
                query_texts=[context],
                n_results=len(chunks),
                include=["documents", "distances"],
            )

            returned_ids = res["ids"][0]
            returned_dists = res["distances"][0]
            max_dist = 1.0 - similarity_cutoff

            kept_indices: List[int] = []
            for _id, dist in zip(returned_ids, returned_dists):
                if dist <= max_dist:
                    # ids are of the form "{run_id}_{i}" where i is the original index
                    idx = int(_id.rsplit("_", 1)[1])
                    kept_indices.append(idx)

            # Return chunks in their original input order
            kept_indices.sort()
            kept_chunks = [chunks[i] for i in kept_indices]

            # Log which chunks were filtered out (i.e., not kept)
            all_indices = set(range(len(chunks)))
            filtered_indices = sorted(all_indices.difference(kept_indices))
            if filtered_indices:
                filtered_texts = [chunks[i] for i in filtered_indices]
                try:
                    self.logger.log(
                        f"Filtered out {len(filtered_texts)} chunk(s): {filtered_texts}"
                    )
                except Exception:
                    # Ensure logging failures don't break functionality
                    pass

            return kept_chunks
        finally:
            # Prevent unbounded growth
            self.chunk_collection.delete(ids=ids)