from langchain_core.documents import Document
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import chromadb.errors

import gc
import os

import config

batch_size = 5

def generate_doc_id(chunk: Document, postfix: str) -> str:
    unique_string = f"{chunk.metadata.get('source')}---{postfix}"
    return unique_string

def create_chroma_store(
    dataset_directory: str = config.dataset_directory,
    chroma_persist_directory: str = config.chroma_persist_directory,
    huggingface_embedding_model_repo_path: str = config.huggingface_embedding_model_repo_path
) -> Chroma:

    embedding_model = HuggingFaceEmbeddings(model_name=huggingface_embedding_model_repo_path)

    store: Chroma

    if os.path.exists(chroma_persist_directory) and os.path.isdir(chroma_persist_directory):
        print(f"✅ Loading existing Chroma DB from: {chroma_persist_directory}")
        store = Chroma(
            embedding_function=embedding_model,
            persist_directory=chroma_persist_directory
        )
    else:
        print(f"📦 Creating new Chroma DB at: {chroma_persist_directory} using batch processing.")
        store = Chroma(
            embedding_function=embedding_model, 
            persist_directory=chroma_persist_directory
        )

    try:
        # Use lazy_load() to get a generator instead of loading all documents into memory
        loader = DirectoryLoader(
            path=dataset_directory, 
            glob="**/*.txt", 
            loader_cls=TextLoader, 
            show_progress=True, 
            use_multithreading=False,
            randomize_sample=True
        )
        # Use iterator to avoid loading all documents
        document_iterator = loader.lazy_load()
    except FileNotFoundError:
        raise FileNotFoundError(f"🚨 Warning: '{dataset_directory}' directory not found.")

    # Splitter for document chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)

    doc_batch: list[Document] = []

    try:
        for document in document_iterator:

            doc_batch.append(document)

            global batch_size
            if len(doc_batch) >= batch_size:
                print(f"Processing batch of {len(doc_batch)} documents...")
                # 3. Split the current batch of documents into chunks
                # Splitting a small batch is memory-efficient
                chunks = splitter.split_documents(doc_batch)

                if len(chunks) == 0:
                    doc_batch = []
                    continue

                # Assign an ID to every chunk
                for i, chunk in enumerate(chunks, 1):
                    chunk.id = generate_doc_id(chunk, str(i))

                existingIds = [doc.id for doc in store.get_by_ids([chunk.id for chunk in chunks if chunk.id is not None]) if doc.id is not None]
                unaddedChunks = [chunk for chunk in chunks if chunk.id is not None and chunk.id not in existingIds]

                if len(unaddedChunks) != 0:
                    try:
                        store.add_documents(unaddedChunks)  # pyright: ignore[reportUnusedCallResult]
                    except chromadb.errors.InternalError:
                        batch_size //= 2

                # E) Reset the batch list
                doc_batch = []

                gc.collect()  # pyright: ignore[reportUnusedCallResult]

                if len(unaddedChunks) != 0:
                    #sleep(61)
                    pass

        # Process the final batch (if any)
        if doc_batch:
            print(f"Processing final batch of {len(doc_batch)} documents...")
            chunks = splitter.split_documents(doc_batch)

            store.add_documents(chunks)  # pyright: ignore[reportUnusedCallResult]

    except KeyboardInterrupt:
        print("Process interrupted")
        pass
            
    return store

def main():
    vectorstore = create_chroma_store()  # pyright: ignore[reportUnusedVariable]

if __name__ == "__main__":
    main()