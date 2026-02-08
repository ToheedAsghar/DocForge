"""
CHUNKER - SPLIT DOCUMENTS INTO CONTEXTUAL CHUNKS

JOB: breakdown long documents into managable chunks for embedding.

WHY:
1. Embeddings have max length (e.g. 8191 tokens for OpenAI)
2. Smaller chunks = more precise retrieval
3. Overlapping chunks = better context preservation
 
STRATEGY:
We'll use the Sliding Window Technique to create chunks:
- chunk size: 1000 characters
- overlap: 200 characters (20%)
- This ensures that the content isn't lost at the chunk boundaries
"""

from typing import List, Dict
import re

class TextChunk:
    """
    Represents a chunk of text with metadata.
    1. text
    2. metadata
    3. chunk_index
    """

    def __init__(
        self,
        text: str,
        metadata: Dict[str, str],
        chunk_index: int,
        total_chunks: int
    ) -> None:
        self.text = text
        self.metadata = metadata
        self.chunk_index = chunk_index
        self.total_chunks = total_chunks

    def __repr__(self) -> str:
        return f"TextChunk(text_length={len(self.text)}, metadata={self.metadata}, chunk_index={self.chunk_index}, total_chunks={self.total_chunks})"
        
class TextChunker:
    """Splits the text into chunks with overlapping context.
    
    Original text: "AAAA BBBB CCCC DDDD EEEE"
    
    Chunk 1: "AAAA BBBB CCCC"
    Chunk 2:      "BBBB CCCC DDDD"  (overlap with previous)
    Chunk 3:           "CCCC DDDD EEEE" (overlap with previous)
    
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        length_function: callable = len
    ):
        """Initialize the Chunker"""

        if chunk_overlap >= chunk_size:
            print(f"[WARNING]\tChunk overlap ({chunk_overlap}) is greater than or equal to chunk size ({chunk_size}). This will result in no overlap.")
            raise ValueError("Chunk overlap must be less than chunk size.")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.length_function = length_function

        print(f"[INFO]\tText Chunker Initialized")
    
    def chunk_text(self, text: str, metadata: Dict[str, str]) -> List[TextChunk]:
        """Splits texts into overlapping chunks
        
        PROCESS:
        1. clean the text
        2. split into sentence (for better performance)
        3. combine sentence into chunks of target size
        4. crate overlapping windows
        5. return chunk objects
        """

        if not text or not text.split():
            print(f"[WARNING]\tEmpty text provided. Returning empty chunks.")
            return []

        sentences = self._split_into_sentence(text)

        if not sentences:
            print(f"[WARNING]\tNo sentences found in the text. Returning empty chunks.")
            return []

        print(f"[INFO]\tChunking {len(text)} characters into chunks...")

        chunks = []
        current_chunk = []
        current_length = 0

        for sentence in sentences:

            sentence_length = self.length_function(sentence)
            if current_length + sentence_length > self.chunk_size and current_chunk:
                chunk_text = ' '.join(current_chunk)

                chunks.append(chunk_text)

                overlap_text = chunk_text[-self.chunk_overlap:] if len(chunk_text) > self.chunk_overlap else chunk_text
                
                #find where overlap starts in the current chunk
                overlap_sentences = []
                overlap_length = 0

                for s in reversed(current_chunk):
                    overlap_length += self.length_function(s)
                    overlap_sentences.insert(0, s)
                    if overlap_length >= self.chunk_overlap:
                        break

                current_chunk = overlap_sentences + [sentence]
                current_length = sum(self.length_function(s) for s in current_chunk)
            else:
                current_chunk.append(sentence)                                                                                                          
                current_length += sentence_length
        
        # the last chunk
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            chunks.append(chunk_text)
        
        print(f"[INFO]\tCreated {len(chunks)} chunks")
        
        # WHAT: Wrap each chunk with metadata
        # WHY: Need to track which chunk came from which document
        
        text_chunks = []
        n: int = len(chunks)
        
        for i, chunk_text in enumerate(chunks):
            # Create metadata for this specific chunk
            chunk_metadata = {
                **metadata,  # Copy all original metadata
                "chunk_index": i,
                "total_chunks": len(chunks),
                "chunk_size": len(chunk_text)
            }
            
            text_chunk = TextChunk(
                text=chunk_text,
                metadata=chunk_metadata,
                chunk_index=i,
                total_chunks=n
            )
            
            text_chunks.append(text_chunk)
        
        return text_chunks
            

    def _split_into_sentence(self, text: str) -> List[str]:
        """Split text into sentences"""

        sentence_pattern = r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s';
        sentences = re.split(sentence_pattern, text)
        sentences = [s.strip() for s in sentences if s.strip()]

        return sentences
    
    def chunk_documents(self, documents: List) -> List[TextChunk]:
        """
        Chunk multiple documents at onece.

        PROCESS:
        1. for each document, chunk the text
        2. collect all chunks from all documents
        3. return combined list
        """

        print(f"[INFO]\tChunking {len(documents)} documents...")

        all_chunks = []

        for i, doc in enumerate(documents, 1):
            doc_id = f"doc_{i}"
            doc.metadata["document_id"] = doc_id

            chunks = self.chunk_text(doc.content, doc.metadata)
            all_chunks.extend(chunks)
        
        print(f"[INFO]\tCreated {len(all_chunks)} chunks from {len(documents)} documents")

        return all_chunks
        