"""
INGESTION PIPELINE -

JOB: Coordinate Loading -> chunking -> embedding -> uploading to pinecone

PROCESS:
1. load documents from directory
2. chunk them into smaller pieces
3. generate embeddings for each chunk
4. upload to pinecone with metadata
"""

from pathlib import Path
from typing import List, Optional
from backend.services.vector_store import get_vector_store
from backend.ingestion.chunker import TextChunk, TextChunker
from backend.services.embeddings import get_embedding_service
from backend.ingestion.document_loader import DocumentLoader, Document

class IngestionPipeline:
    """
    Complete ETL pipeline for document ingestion.
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ) -> None:
        self.document_loader = DocumentLoader()
        self.chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.embedding_service = get_embedding_service()
        self.vector_store = get_vector_store()

    def ingest_directory(self, dir_path: str, recursive: bool = True, namespace: Optional[str] = None) -> None:
        documents = self.document_loader.load_directory(dir_path=dir_path, recursive=recursive)

        # 1. loading documents

        if not documents:
            print(f"[ERROR]\tNo documents found in {dir_path}")
            return {
                "documents_loaded": 0,
                "chunks_created": 0,
                "chunks_uploaded": 0
            }

        # 2. chunking documents

        chunks = self.chunker.chunk_documents(documents)

        if not chunks:
            print(f"[ERROR]\tNo chunks created from {len(documents)} documents")

            return {
                "documents_loaded": len(documents),
                "chunks_created": 0,
                "chunks_uploaded": 0
            }
       
       # 3. generating embeddings
        
        chunk_texts = [chunk.text for chunk in chunks]
        embeddings = self.embedding_service.embed_batch(chunk_texts)

        if not embeddings:
            print(f"[ERROR]\tNo embeddings generated for {len(chunk_texts)} chunks")

            return {
                "documents_loaded": len(documents),
                "chunks_created": len(chunks),
                "embeddings_generated": 0,
                "chunks_uploaded": 0
            }

        # 4. uploading to pinecone

        pinecone_vectors = []

        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            doc_id = chunk.metadata.get("document_id", f"doc_{i}")
            chunk_id = f"{doc_id}_chunk_{chunk.chunk_index}"
            
            vector = {
                "id": chunk_id,
                "values": embedding,
                "metadata": {
                    "text": chunk.text,
                    "document_id": doc_id,
                    "chunk_index": chunk.chunk_index,
                    "total_chunks": chunk.total_chunks,
                    **chunk.metadata
                }
            }

            pinecone_vectors.append(vector)

        # 5. uploading to pinecone

        uploaded_count = self.vector_store.upsert_documents(
            [{"id": v["id"], "text": v["metadata"]["text"], "metadata": v["metadata"]} 
             for v in pinecone_vectors]
        )
        
        # ========== STEP 6: Summary ==========
        
        print(f"\n{'='*60}")
        print(f"[INFO]\tINGESTION COMPLETE")
        print(f"{'='*60}")
        print(f"Documents loaded: {len(documents)}")
        print(f"Chunks created: {len(chunks)}")
        print(f"Chunks uploaded: {uploaded_count}")
        print(f"{'='*60}\n")
        
        # Return statistics
        return {
            "documents_loaded": len(documents),
            "chunks_created": len(chunks),
            "chunks_uploaded": uploaded_count,
            "documents": [
                {
                    "filename": doc.metadata.get("filename"),
                    "type": doc.metadata.get("type"),
                    "size": len(doc.content)
                }
                for doc in documents
            ]
        }

    def ingest_single_file(self, file_path: str) -> dict:
        """Ingest a single file into the vector store"""

        # load document
        document = self.document_loader(file_path)

        doc_id = Path(file_path).stem.replace(" ", "_")
        document.metadata["document_id"] = doc_id
        
        # chunk document
        chunks = self.chunker.chunk_text(document.content, document.metadata)

        if not chunks:
            print(f"[ERROR]\tNo chunks created")
            return {
                "documents_loaded": 1,
                "chunks_created": 0,
                "chunks_uploaded": 0
            }
        
        print(f"[INFO]\tCreated {len(chunks)} chunks")

        # generate embeddings for chunks
        embeddings = self.embedding_service.embed_batch([chunk.text for chunk in chunks])

        # upload to pinecone
        pinecone_vectors = []

        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            chunk_id = f"{doc_id}_chunk_{chunk.chunk_index}"

            vector = {
                "id": chunk_id,
                "values": embedding,
                "metadata": {
                    "text": chunk.text,
                    "document_id": doc_id,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "source": document.metadata.get("source", "unknown"),
                    "filename": document.metadata.get("filename", "unknown"),
                    "title": document.metadata.get("title", "unknown"),
                    "type": document.metadata.get("type", "text")
                }
            }
            pinecone_vectors.append(vector)

        # upload to pinecone
        self.vector_store.upsert(
            vectors=pinecone_vectors,
            namespace=self.vector_store.namespace
        )
        uploaded_cnt = len(pinecone_vectors)

        print(f"[INFO]\tUploaded {uploaded_cnt} chunks to Pinecone")

        # return
        return {
            "documents_loaded": 1,
            "chunks_created": len(chunks),
            "chunks_uploaded": uploaded_cnt,
            "document": {
                "filename": document.metadata.get("filename", "unknown"),
                "type": document.metadata.get("type", "text"),
                "size": len(document.content)
            }
        }

    def get_ingestion_stats(self) -> dict:
        """Get statistics abou what's in the vector store """
        stats = self.vector_store.get_stats()
        return stats

    def clear_all_documents(self) -> bool:
        """Clear all documents from the vector store
        
        WARNING: THIS IS IRREVERSIBLE!
        """
        result = self.vector_store.delete_all()

        if result:
            print(f"[INFO]\tSuccessfully cleared all documents from the vector store")
        else:
            print(f"[ERROR]\tFailed to clear all documents from the vector store")
        
        return result

# --- CONVENIENCE FUNCTIONS ---

def ingest_documents(
    dir_path: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    recursive: bool = True
) -> dict:
    """Quick function to ingest all documents from a directory"""
    pipeline = IngestionPipeline(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return pipeline.ingest_directory(dir_path=dir_path, recursive=recursive)

def ingest_file(
    file_path: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200
    ) -> dict:
        """Quick function to ingest a document from a directory"""
        pipeline = IngestionPipeline(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        return pipeline.ingest_single_file(file_path=file_path)

def get_stats() -> dict:
    """A quick function to check what's in the vector store"""
    pipeline = IngestionPipeline()
    return pipeline.get_ingestion_stats()

def clear_all() -> bool:
    """A quick function to clear all documents from the vector store
    
    WARNING: THIS IS IRREVERSIBLE!
    """
    pipeline = IngestionPipeline()
    return pipeline.clear_all_documents()
        

