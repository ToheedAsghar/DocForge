"""
DOCUMEN LOADER - LOAD FILES FROM DISK

SUPPORTED DOCUMENTS:
- .pdf
"""

import os
import re
from pathlib import Path
from pydantic import FilePath
from typing import List, Dict, Optional


class Document:
    """
    Represents a loaded document with metadata.
    """

    def __init__(self, content: str, metadata: Dict[str, str]):
        """
        Initialize a Document
        """

        self.content = content
        self.metadata = metadata
    
    def __repr__(self) -> str:
        return f"Document(content_length={len(self.content)}, metadata={self.metadata})"
        
class DocumentLoader:
    """
    Loads documents from various file formats.

    PROCESS:
    1. Scan a directory for supported files
    2. for each file, detect its types
    3. Use appropriate loader for that type
    4. Extract text and metadata
    5. Return Document objects
    """

    def __init__(self):
        self.loaders = {
            ".pdf": self._load_pdf
        }

    def load_directory(self, dir_path: str, recursive: bool = True) -> List[Document]:
        """
        Load all documents from a directory.

        PROCESS:
        1. Scan the directory for supported files
        2. For each file, detect its type
        3. Use appropriate loader for that type
        4. Extract text and metadata
        5. Return Document objects
        """

        # scans the directory for supported files
        directory = Path(dir_path)

        if not directory.exists():
            print(f"[ERROR]\tDirectory not found: {dir_path}")
            raise FileNotFoundError(f"Directory not found {dir_path}")
        
        if not directory.is_dir():
            print(f"[ERROR]\tNot a directory: {dir_path}")
            raise NotADirectoryError(f"Not a directory: {dir_path}")

        print(f"[INFO]\tLoading documents from {directory}...")
        
        all_files = []

        if recursive:
            for root, dirs, files in os.walk(directory):
                for file in files:
                    file_path = Path(root) /file
                    all_files.append(file_path)
        else:
            all_files = [f for f in directory.iterdir() if f.is_file()]

        print(f"[INFO]\tFound {len(all_files)} files to process.")

        supported_documents = []

        for file_path in all_files:
            extension = file_path.suffix.lower()

            if extension in self.loaders:
                supported_documents.append(file_path)
            
        print(f"[INFO]\tFound {len(supported_documents)} supported documents.")

        # process each document and extract text from it

        documents = []
        errors = []

        for i, filepath in enumerate(supported_documents, 1):
            try:
                print(f"[INFO]\tProcessing Document {i} of {len(supported_documents)}: {filepath.name}...")
    
                doc = self.load_file(str(file_path))
                documents.append(doc)

                print(f"[INFO] Successfully loaded {len(doc.content)} characters from {filepath.name}")            
            except Exception as e:
                print(f"[ERROR]\tFailed to load {filepath.name}: {str(e)}")
                errors.append(str(e))

        # summary
        print(f"\n{'='*60}")
        
        
        if 0 != len(errors):
            print(f"[WARNING]\tFailed to load {len(errors)} documents.")
            print(f"[WARNING]\tErrors: {', '.join(errors)}")
       
        print(f"[INFO]\tDOCUMENT LOADING SUMMARY")
        print(f"[INFO]\tTotal Documents Loaded: {len(documents)}")
        print(f"[INFO]\tTotal Characters Loaded: {sum(len(doc.content) for doc in documents)}")
        print(f"[INFO]\tTotal Errors: {len(errors)}")

        return documents

    def load_file(self, file_path: str) -> Document:
        """
        Load a single file
        """

        path = FilePath(file_path)

        if not path.exists():
            print(f"[ERROR]\tFile not found: {file_path}")
            raise FileNotFoundError(f"File not found: {file_path}")
        
        extension = path.suffix.lower()

        if extension not in self.loaders:
            print(f"[ERROR]\tUnsupported file type: {extension}")
            raise ValueError(f"Unsupported file type: {extension}")
        
        loader_method = self.loaders[extension]
        return loader_method(path)
    
    ### ALL FILE TYPE LOADERS
    def _load_pdf(self, path: Path) -> Document:
        """
        LOAD PDF FILE
        """

        try:
            from pypdf import PdfReader
        except ImportError:
            raise ImportError(
                "pypdf is not installed. Please install it with 'pip install pypdf'"
            )
        
        content_parts = []
        try:
            with open(path, 'rb') as f:
                reader = PdfReader(f)
                
                # extract text from each page
                for pg_num in range(len(reader.pages)):
                    page = reader.pages[pg_num]
                    text = page.extract_text()
                    
                    if text.strip():
                        content_parts.append(text)
                    
            content = "\n\n".join(content_parts)
        except Exception as e:
            print(f"[ERROR]\tFailed to load PDF file: {path}: {str(e)}")
            raise RuntimeError(f"Failed to load PDF file: {path}: {str(e)}")

        content = self._clean_text(content)

        # get pdf title from metadata
        title = path.stem
        try:
            with open(path,'rb') as f:
                reader = PdfReader(f)
                if reader.metadata and reader.metadata.title:
                    title = reader.metadata.title
        except:
            pass

        metadata = {
            "title": title,
            "filename": path.name,
            "type": "pdf",
            "source": str(path)
        }

        return Document(content=content, metadata=metadata)

    
    def _clean_text(self, text: str) -> str:
        """
        Clean the text by removing extra whitespace and newlines.

        STEPS:
        1. Remove multiple newlines (collapse to 2 max)
        2. Remove excessive spaces
        3. Strip leading and trailing whitespace
        4. Remove special characters that cause issues
        """

        if not text:
            return ""
        
        # remove null bytes
        text = text.replace ("\x00", "")

        # normalize new lines
        text = re.sub(fr'\n{3,}', '\n\n', text)

        # remove excessive spaces
        text = re.sub(fr' {2,}', ' ', text)

        # remove whitespace at line ends
        lines = [line.strip() for line in text.splitlines()]
        text = "\n".join(lines)

        # remove leading/trailing whitespaces
        text = text.strip()

        return text



        

