import os
import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from datetime import datetime
from typing import List, Optional

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
        os.makedirs(db_path, exist_ok=True)

        self.embedding_function = OpenAIEmbeddingFunction(
            api_key_env_var="OPENAI_API_KEY",
            model_name=os.environ.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
            api_base=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        )
        self.chroma_client = chromadb.PersistentClient(path=self.db_path)
        self.collection = self.chroma_client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=self.embedding_function,
            metadata={"hnsw:space": "cosine"},
        )

    def find_duplicate(self, memory_text: str) -> Optional[str]:
        """Return an existing memory whose cosine similarity exceeds the threshold."""
        if self.collection.count() == 0:
            return None
        results = self.collection.query(
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
        if self.find_duplicate(memory_text) is not None:
            return False

        metadata = {
            "timestamp": datetime.now().isoformat(),
            "source": source
        }

        if tags:
            metadata["tags"] = ",".join(tags)

        doc_id = f"mem_{datetime.now().timestamp()}"

        self.collection.add(
            documents=[memory_text],
            metadatas=[metadata],
            ids=[doc_id]
        )
        return True

    def retrieve_memories(self, query: str, n_results: int = 3) -> List[str]:
        """Retrieve relevant long-term semantic memories."""
        count = self.collection.count()
        if count == 0 or not query.strip():
            return []

        results = self.collection.query(
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
