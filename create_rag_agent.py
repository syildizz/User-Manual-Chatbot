# rag_gemini_chroma_v1_0_1_fixed.py
import os
from typing import Any
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_huggingface import HuggingFaceEmbeddings
from langgraph.graph.state import CompiledStateGraph
from pydantic import SecretStr

from huggingface_hub import snapshot_download

# Core types and prompt
from langchain_core.documents import Document

# Document loaders, text splitters, vectorstore (community / ecosystem packages)
from langchain_chroma import Chroma

# Google Gemini provider and embeddings package
from langchain_google_genai import ChatGoogleGenerativeAI

from langchain.tools import tool

import config

# Instead, LCEL (Runnable components) is used for chain composition.

def get_chroma_store(
    chroma_persist_directory: str = config.chroma_persist_directory,
    huggingface_embedding_model_repo_path: str = config.huggingface_embedding_model_repo_path,
    huggingface_vector_embedding_database_repo_path: str = config.huggingface_vector_embedding_database_repo_path,
) -> Chroma:
    """
    Load an existing Chroma store if present, otherwise create from docs and persist.
    
    This version uses lazy loading and batch processing to prevent memory issues.
    """

    embedding_model = HuggingFaceEmbeddings(model_name=huggingface_embedding_model_repo_path)

    # 3) Check for existing Chroma DB and load it
    if os.path.exists(chroma_persist_directory) and os.path.isdir(chroma_persist_directory):
        print(f"✅ Loading existing Chroma DB from: {chroma_persist_directory}")
    else:
        print("📥 No local Chroma DB found. Pulling from Hugging Face dataset...")
        
        # Create local directory
        os.makedirs(chroma_persist_directory, exist_ok=True)
        
        # Download all files from the Hugging Face dataset
        snapshot_download(  # pyright: ignore[reportUnusedCallResult]
            repo_id=huggingface_vector_embedding_database_repo_path,
            repo_type="dataset",
            local_dir=chroma_persist_directory,
            ignore_patterns=["*.md", "*.json"],  # Optional: skip non-DB files like README,
        )
        
        print(f"✅ Pulled and persisted Chroma DB to: {chroma_persist_directory}")
        
    return Chroma(
        embedding_function=embedding_model,
        persist_directory=chroma_persist_directory
    )
    

def create_rag_agent(
    google_llm_model_name: str = config.google_llm_model_name,
    temperature: float = 0.3
) -> CompiledStateGraph[Any]:
    load_dotenv()  # pyright: ignore[reportUnusedCallResult]

    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        raise ValueError("Missing GEMINI_API_KEY in environment")

    vector_store = get_chroma_store()

    # 6) Create Gemini chat model (LLM)
    llm = ChatGoogleGenerativeAI(model=google_llm_model_name, temperature=temperature, google_api_key=SecretStr(gemini_api_key))

    # 7) Prompt template
    # Note: The prompt input variables must match the dict passed to the model
    system_prompt = """
        You are provided with a list of sample text that comes from various different user manuals. 
        Your task is to respond to the user using the samples provided to the best of your abilities. 
        The context text is in the following paragraph.

        """
    
    # Helper to format documents for the prompt
    def format_docs(docs: list[Document]) -> str:
        """Formats a list of documents into a single string."""
        return "\n".join(doc.page_content for doc in docs)

    # 8) Build RAG chain using LCEL (LangChain Expression Language)
    # The chain structure is:
    # { 'context': retriever | format_docs, 'input': RunnablePassthrough() } | prompt | llm

    @tool #fonksiyonun hem cevap (content) hem de kaynak/detay (artifact) döndüreceğini belirtir.
    def retrieve_context(query: str) -> str:
        '''Sorguyu yanıtlamaya yardımcı olacak bilgileri getir.'''
        retrieved_docs = vector_store.similarity_search(query, k=5)
        return format_docs(retrieved_docs)

    rag_agent = create_agent(llm, [retrieve_context], system_prompt=system_prompt)

    return rag_agent

if __name__ == "__main__":
    rag_agent = create_rag_agent()
    result: dict[str, Any] | Any = rag_agent.invoke( 
        {"messages": [{"role": "user", "content": "I want to replace the batteries of a sony brand remote. What can I do?"}]}
    )