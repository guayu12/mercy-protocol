import hashlib
import os
import threading
from datetime import datetime
from typing import List, Optional

from core.models import provider_key

# Separate collection name: the previous collection was built with Chroma's
# local ONNX embedder (384-dim), which is incompatible with OpenAI vectors.
COLLECTION_NAME = "personal_assistant_memory_openai"


class MemoryManager:
    """
    Handles long-term semantic memory for the Personal AI Assistant.
    Provides RAG capabilities by inserting past memories into the context.
    Embeddings are computed remotely by the OpenAI API (no local inference).
    """
    def __init__(self, db_path: str = "data/chroma_db", dedup_threshold: float = 0.92):
        self.db_path = db_path
        self.dedup_threshold = dedup_threshold
        self._write_lock = threading.Lock()  # makes duplicate check + insert atomic
        self.chroma_client = None
        self.embedding_function = None
        self.collection = None
        self._key_fingerprint = None

    def initialize(self) -> bool:
        return self._ensure_collection() is not None

    def _ensure_collection(self):
        key = provider_key("openai")
        if key is None:
            return None
        fingerprint = hashlib.sha256(key[0].encode()).hexdigest()[:12]
        if self.chroma_client is None:
            import chromadb

            self.chroma_client = chromadb.PersistentClient(path=self.db_path)
        if self.collection is None or fingerprint != self._key_fingerprint:
            from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

            self.embedding_function = OpenAIEmbeddingFunction(
                api_key=key[0],
                model_name=os.environ.get(
                    "OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"
                ),
                api_base=os.environ.get(
                    "OPENAI_BASE_URL", "https://api.openai.com/v1"
                ),
            )
            self.collection = self.chroma_client.get_or_create_collection(
                name=COLLECTION_NAME,
                embedding_function=self.embedding_function,
                metadata={"hnsw:space": "cosine"},
            )
            self._key_fingerprint = fingerprint
        return self.collection

    def find_duplicate(self, memory_text: str) -> Optional[str]:
        """Return an existing memory whose cosine similarity exceeds the threshold."""
        collection = self._ensure_collection()
        if collection is None or collection.count() == 0:
            return None
        results = collection.query(
            query_texts=[memory_text],
            n_results=1,
            include=["documents", "distances"],
        )
        if not results["documents"] or not results["documents"][0]:
            return None
        similarity = 1 - results["distances"][0][0]
        if similarity >= self.dedup_threshold:
            return results["documents"][0][0]
        return None

    def add_memory(self, memory_text: str, source: str = "direct_chat", tags: List[str] = None) -> bool:
        """Save an important fact or preference to long-term vector store.
        Returns False if a near-duplicate memory already exists."""
        with self._write_lock:
            return self._add_memory_unlocked(memory_text, source, tags)

    def _add_memory_unlocked(self, memory_text: str, source: str, tags: Optional[List[str]]) -> bool:
        collection = self._ensure_collection()
        if collection is None:
            return False
        if self.find_duplicate(memory_text) is not None:
            return False

        metadata = {
            "timestamp": datetime.now().isoformat(),
            "source": source
        }

        if tags:
            metadata["tags"] = ",".join(tags)

        doc_id = f"mem_{datetime.now().timestamp()}"

        collection.add(
            documents=[memory_text],
            metadatas=[metadata],
            ids=[doc_id]
        )
        return True

    def retrieve_memories(self, query: str, n_results: int = 3) -> List[str]:
        """Retrieve relevant long-term semantic memories."""
        collection = self._ensure_collection()
        if collection is None:
            return []
        count = collection.count()
        if count == 0 or not query.strip():
            return []

        results = collection.query(
            query_texts=[query],
            n_results=min(n_results, count)
        )

        if not results['documents']:
            return []

        return results['documents'][0]

    def format_context(self, query: str) -> str:
        """Helper to get a formatted string of relevant long-term memories for injection."""
        memories = self.retrieve_memories(query)
        if not memories:
            return ""

        formatted = "Here are some relevant things you learned about the user in the past:\n"
        for mem in memories:
            formatted += f"- {mem}\n"
        return formatted
