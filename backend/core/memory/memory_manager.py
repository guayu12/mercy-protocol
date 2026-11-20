import os
import chromadb
from datetime import datetime
from typing import List, Dict, Optional

class MemoryManager:
    """
    Handles long-term semantic memory for the Personal AI Assistant.
    Provides RAG capabilities by inserting past memories into the context.
    """
    def __init__(self, db_path: str = "data/chroma_db"):
        self.db_path = db_path
        os.makedirs(db_path, exist_ok=True)
        
        # In a real app we'd use OpenAI embeddings or local BGE representations
        # Here we use Chroma's default sentence-transformers embedding
        self.chroma_client = chromadb.PersistentClient(path=self.db_path)
        self.collection = self.chroma_client.get_or_create_collection(name="personal_assistant_memory")

    def add_memory(self, memory_text: str, source: str = "direct_chat", tags: List[str] = None):
        """Save an important fact or preference to long-term vector store."""
        
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
        if self.collection.count() == 0:
            return []
            
        results = self.collection.query(
            query_texts=[query],
            n_results=min(n_results, self.collection.count())
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
        for i, mem in enumerate(memories):
            formatted += f"- {mem}\n"
        return formatted
