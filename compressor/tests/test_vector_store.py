import pytest

from components.vector_store import VectorStore

@pytest.fixture
def vector_store():
    vs = VectorStore()
    yield vs


def test_add_and_query_words(vector_store):
    words = ["apple", "banana", "cherry"]
    vector_store.add_words(words)
    result = vector_store.query("apple", n_results=2)
    assert "apple" in result["documents"][0]


def test_filter_chunks_by_context(vector_store):
    chunks = ["the quick brown fox", "jumps over the lazy dog", "hello world"]
    context = "quick brown"
    filtered = vector_store.filter_chunks_by_context(context, chunks, similarity_cutoff=0.2)
    assert isinstance(filtered, list)
    assert any("quick brown" in chunk or "fox" in chunk for chunk in filtered)
    
