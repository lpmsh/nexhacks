from typing import List
from tokenc import TokenClient, CompressResponse
import time

# use package-relative import so module works when imported as a package
from .vector_store import VectorStore
from .logger import Logger


class CustomCompressor:
    def __init__(self, tokenc_client: TokenClient, vector_store: VectorStore, logger: Logger, similarity_cutoff: float = 0.1, chunk_size: int = 3):
        self.token_client = tokenc_client

        # injected logger (defaults to a new Logger for backward compatibility)
        self.logger = logger

        # similarity cutoff (cosine similarity threshold)
        self.delta = similarity_cutoff

        # number of whitespace tokens per chunk
        self.chunk_size = chunk_size

        # allow injection for reuse/testing
        self.vector_store = vector_store

    def compress(self, text: str, context: str, aggressiveness: float) -> CompressResponse:
        """Compress a single text and return the final CompressResponse.
        Also sets `compression_time` on the returned response if available.
        """
        
        self.logger.log(f"Original text length: {len(text)} characters")
        self.logger.log(f"Original text: {text}")

        # 1) remove irrelevant info BEFORE compressing
        start = time.time()
        filtered_text = self.remove_irrelevant_info(text=text, context=context)
        self.logger.log(f"Filtered text length: {len(filtered_text)} characters")
        self.logger.log(f"Filtered text: {filtered_text}")

        compressed = self.token_client.compress_input(input=filtered_text, aggressiveness=aggressiveness)
        
        self.logger.log(f"Final compressed output with Token Company Algo: {compressed.output}")
        end = time.time()

        compressed.compression_time = end - start
        return compressed

    def remove_irrelevant_info(self, text: str, context: str) -> str:
        """
        Chunk `text`, then ask VectorStore to keep only chunks whose similarity
        to `context` is >= self.delta (internally via Chroma cosine distance).
        """
        
        chunks = self.chunk_text(text)
        self.logger.log(f"Chunked text into {chunks}")
        if not chunks:
            return text

        kept_chunks = self.vector_store.filter_chunks_by_context(
            context=context,
            chunks=chunks,
            similarity_cutoff=self.delta,
        )

        return " ".join(kept_chunks)

    def chunk_text(self, text: str) -> List[str]:
        """
        Simple whitespace token chunking.
        Example: chunk_size=3 -> ["a b c", "d e f", ...]
        """
        tokens = text.split()
        if not tokens:
            return []

        chunks: List[str] = []
        for i in range(0, len(tokens), self.chunk_size):
            chunks.append(" ".join(tokens[i : i + self.chunk_size]))
        return chunks

    def compress_many(self, texts: List[str], context: str, aggressiveness: float) -> List[CompressResponse]:
        return [self.compress(t, context, aggressiveness) for t in texts]