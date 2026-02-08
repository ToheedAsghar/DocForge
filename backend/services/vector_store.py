"""
VECTOR STORE SERVICE
- Manages Pinecone vector database
Handles:
    - Index creation and managment
    - Document isnertion (upsert)
    - Similarity Search (retrieval)
    - Metadata filtering
"""

import time
from backend.config import settings
from typing import List, Dict, Any, Optional
from pinecone import Pinecone, ServerlessSpec
from backend.services.embeddings import get_embedding_service

class VectorStore:
    def __init__(self) -> None:
        self.pc = Pinecone(api_key=settings.PINECONE_API_KEY)
        self.index_name = settings.PINECONE_INDEX_NAME
        self.namespace = settings.PINECONE_NAMESPACE
        self.dimension = settings.EMBEDDING_DIMENSIONS

        self.embedding_service = get_embedding_service()
        self.index = None

        self.initialize_index()

    def initialize_index(self) -> None:
        """
        GET (or create) the Pinecone index and connect to it.
        This is called automatically on initialization        
        """

        try:
            # Get list of existing index names
            existing_indexes = [index.name for index in self.pc.list_indexes()]

            if self.index_name not in existing_indexes:
                print(f"[INFO]\tCreating new index: {self.index_name}")
                self.pc.create_index(
                    name=self.index_name,
                    dimension=self.dimension,
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud="aws",
                        region=settings.PINECONE_ENVIRONMENT,
                    )
                )

                print(f"[INFO]\tWaiting for Index to be ready...")
                while not self.pc.describe_index(self.index_name).status['ready']:
                    time.sleep(1)
                print(f"[INFO]\tIndex {self.index_name} is ready!")
            else:
                print(f"[INFO]\tUsing existing index: {self.index_name}")

            self.index = self.pc.Index(self.index_name)

        except Exception as e:
            print(f"[ERROR]\tFailed to initialize index: {str(e)}")
            raise

    def upsert_documents(self, documents: List[dict[str, any]]) -> int:
        """
        Insert or Update Documents in the Vector Store
        Each document should have:
            - ID 
            - Text
            - Metadata (title, sources etc.)
        """

        if not documents:
            return 0

        texts = [doc['text'] for doc in documents]

        print(f"[INFO]\tGenerating Embedding for {len(texts)} documents ...")

        embeddings = self.embedding_service.embed_batch(texts)

        # prepare vectors for Pinecone
        vectors = []
        for i, doc in enumerate(documents):
            vector = {
                "id": doc["id"],
                "values": embeddings[i],
                "metadata": {
                    "text": doc["text"], # storing original text for retrieval
                    **doc.get("metadata", {}) # add any extra metadata
                }
            }

            vectors.append(vector)

        BATCH_SIZE = 100
        total_upserted = 0

        for i in range(0, len(vectors), BATCH_SIZE):
            batch = vectors[i: i+BATCH_SIZE]
            try:
                self.index.upsert(
                    vectors=batch,
                    namespace=self.namespace
                )

                total_upserted += len(batch)
                print(f"[INFO]\tUpserted {len(batch)} documents to Pinecone ...")
            except Exception as e:
                print(f"[ERROR]\tFailed to upsert batch {i//BATCH_SIZE + 1}: {str(e)}")
                raise

        print(f"[INFO]\tSuccessfully upserted {total_upserted} documents to Pinecone!")
        return total_upserted
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None,
        min_score: Optional[float] = None
    ) -> List[dict[str, any]]:
        """
        Perform similarity search on the vector store
        """

        query_embedding = self.embedding_service.embed_text(query)

        try:
            response = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True,
                namespace=self.namespace,
                filter=filter_dict
            )

            # Debug: Show raw results before filtering
            print(f"[DEBUG]\tPinecone returned {len(response.matches)} matches")
            for m in response.matches:
                print(f"[DEBUG]\t  - {m.id}: score={m.score:.3f}")

            results = []
            for match in response.matches:
                if min_score and match.score < min_score:
                    print(f"[DEBUG]\tFiltered out {match.id} (score {match.score:.3f} < min_score {min_score})")
                    continue
                result = {
                    "id": match.id,
                    "score": match.score,
                    "text": match.metadata.get("text", ""),
                    "metadata": {k: v for k, v in match.metadata.items() if k != "text"}
                }

                results.append(result)

        except Exception as e:
            print(f"[ERROR]\tFailed to search Pinecone: {str(e)}")
            raise

        return results

    def delete_by_id(self, ids: List[str]) -> bool:
        """
        Delete documents by ID from the vector store
        """

        try:
            self.index.delete(
                ids=ids,
                namespace=self.namespace
            )

            return True
        except Exception as e:
            print(f"[ERROR]\tFailed to delete documents: {str(e)}")
            return False

    def delete_all(self) -> bool:
        """
        Delete all documents in the namespace
        
        WARNING: THIS IS IRREVERSIBLE!
        """

        try:
            self.index.delete(
                delete_all=True,
                namespace=self.namespace
            )
            return True
        except Exception as e:
            print(f"[ERROR]\tFailed to delete all documents: {str(e)}")
            return False
        
    def get_stats(self) -> Dict[str, Any]:
        """Get Index Statistics (total vectors, dimension, namespaces etc.)"""

        try:
            stats = self.index.describe_index_stats()

            return {
                "total_vectors": stats.total_vector_count,
                "dimension": stats.dimension,
                "namespaces": dict(stats.namespaces)
            }

        except Exception as e:
            print(f"[ERROR]\tFailed to get index stats: {str(e)}")
            return {}

_vector_store = None

def get_vector_store() -> VectorStore:
    """
    GET or CREATE the singleton instance of the VectorStore
    """
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore()
    return _vector_store
