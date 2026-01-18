import math
from collections import Counter
from typing import Dict, List, Sequence

from .chunker import tokenize

Vector = Dict[str, float]


class SimpleEmbedder:
    """Lightweight TF-IDF style embedder to avoid heavy deps."""

    def __init__(self) -> None:
        self.idf: Dict[str, float] = {}
        self.fitted = False

    def fit(self, documents: Sequence[str]) -> None:
        doc_tokens = [set(tokenize(doc)) for doc in documents]
        doc_count = len(documents)
        df = Counter()
        for tokens in doc_tokens:
            df.update(tokens)

        self.idf = {
            token: math.log((doc_count + 1) / (freq + 1)) + 1.0 for token, freq in df.items()
        }
        self.fitted = True

    def transform(self, documents: Sequence[str]) -> List[Vector]:
        if not self.fitted:
            self.fit(documents)
        return [self._vectorize(doc) for doc in documents]

    def encode(self, document: str) -> Vector:
        if not self.fitted:
            self.fit([document])
        return self._vectorize(document)

    def _vectorize(self, document: str) -> Vector:
        counts = Counter(tokenize(document))
        vector: Vector = {}
        for token, count in counts.items():
            idf = self.idf.get(token, 1.0)
            tf = 1 + math.log(count)
            vector[token] = tf * idf
        norm = math.sqrt(sum(v * v for v in vector.values())) or 1.0
        for token in list(vector.keys()):
            vector[token] /= norm
        return vector


def cosine(a: Vector, b: Vector) -> float:
    if not a or not b:
        return 0.0
    if len(a) > len(b):
        a, b = b, a
    dot = sum(weight * b.get(token, 0.0) for token, weight in a.items())
    return dot


def similarity_matrix(embeddings: Sequence[Vector]) -> List[List[float]]:
    n = len(embeddings)
    matrix: List[List[float]] = [[0.0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        matrix[i][i] = 1.0
        for j in range(i + 1, n):
            sim = cosine(embeddings[i], embeddings[j])
            matrix[i][j] = sim
            matrix[j][i] = sim
    return matrix
