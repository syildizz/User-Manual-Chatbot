# User Manual Chatbot

## Project Overview

This project is a chatbot developed as part of the **Akbank GenAI Bootcamp 2025**. 
The chatbot leverages a database of user manuals for various products to provide accurate and contextually relevant answers to technical questions. 
By utilizing **Retrieval-Augmented Generation (RAG)** technology, the chatbot retrieves relevant information from user manuals and combines it with the generative capabilities of the **Gemini-2.5-flash** model to deliver precise responses. 
The project includes a user-friendly interface built with **Gradio** which can be used to interact with the chatbot.

### Purpose

The goal of this project is to create an intelligent chatbot capable of answering technical queries about electronic devices and products by referencing user manuals. 
This enables users to quickly access accurate information without manually searching through lengthy documentation.

---

## Dataset

The dataset used in this project is sourced from the dataset described in the paper *[Question Answering over Electronic Devices: A New Benchmark Dataset and a Multi-Task Learning based QA Framework](https://arxiv.org/abs/2109.05897)*. 
It can be accessed via this [Google Drive link](https://drive.google.com/drive/folders/1-gX1DlmVodP6OVRJC3WBRZoGgxPuJvvt).

### Dataset Details
- **Format**: Text-based user manuals for various electronic devices.
- **Preprocessing**: The manuals are split into overlapping chunks to facilitate efficient retrieval.
- **Embedding Generation**: The text chunks are converted into embeddings using the [sentence-transformers/all-mpnet-base-v2](https://huggingface.co/sentence-transformers/all-mpnet-base-v2) model from HuggingFace.
- **Generation Script**: The [generate_vector_db.py](./generate_vector_db.py) script processes the dataset and generates the vector database. If the process is interrupted the generated embeddings are saved and when ran again, the program will generate new embeddings.
- **Vector Database**: The embeddings are stored in a Chroma vector database and can be used locally. However, a pregenerated database already exists can be used via the HuggingFace dataset [syildizz/user-manuals-chromadb](https://huggingface.co/datasets/syildizz/user-manuals-chromadb).

### Usage

When generating the dataset, the folders that are used for the input user-manual dataset and the output Chroma dataset by the [generate_vector_db.py](./generate_vector_db.py) script is specified in the [config.py](config.py) file.

```python
dataset_directory = "user_manual_dataset_folder_path"
chroma_persist_directory = "chroma_dataset_folder_path"
```

---

## Methods and Technologies

### Solution Architecture
The chatbot employs a **Retrieval-Augmented Generation (RAG)** pipeline to combine information retrieval with generative AI:
1. **Vector Database**: The embeddings are retrieved from a **Chroma** vector database for efficient similarity-based retrieval.
2. **Query Processing**: When a user submits a query, the system retrieves the most relevant manual chunks using similarity search.
3. **Response Generation**: The retrieved chunks are passed to the **Gemini-2.5-flash** model to generate a coherent and contextually accurate response.
4. **User Interface**: A **Gradio**-based interface allows users to interact with the chatbot seamlessly.

### Technologies Used
- **LLM**: Gemini-2.5-flash (`langchain-google-genai`)
- **Embedding Model**: sentence-transformers/all-mpnet-base-v2 (`langchain-huggingface`)
- **Vector Database**: Chroma (`langchain-chroma`, `chromadb`)
- **Text Splitting**: `langchain-text-splitters`
- **Interface**: Gradio (`gradio`)
- **Environment Management**: `python-dotenv`, `pydantic`
- **Other Libraries**: `langchain`, `langchain-core`, `langchain-community`

### Key Features
- **RAG-based Retrieval**: Ensures answers are grounded in the user manual dataset.
- **Incremental Vector Database**: The `generate_vector_db.py` script supports resumable processing.
- **Configurability**: Parameters like chunk size and overlap are adjustable in `config.py`.
- **Interactive UI**: Gradio interface for easy user interaction.

---

## Results

The chatbot successfully answers technical questions about electronic devices by retrieving relevant information from user manuals. 
Key outcomes include:

- **Accuracy**: The RAG pipeline ensures responses are highly relevant to the query, leveraging the structured manual dataset.
- **Scalability**: The incremental vector database generation supports large datasets and resumable processing.
- **Usability**: The Gradio interface provides a seamless experience for users to query the chatbot.
- **Deployment**: The project is live on HuggingFace Spaces at [Placeholder Link](https://huggingface.co/spaces/placeholder).

---

## Setup and Installation

### Prerequisites
- Python 
- Git
- Virtual environment (recommended)

### Installation Steps

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/syildizz/[your-repo-name].git
   cd [your-repo-name]
   ```

2. **Set Up a Virtual Environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install Dependencies**:
   Use the [requirements.txt](./requirements.txt) file to install dependencies via running:

   ```bash
   pip install -r requirements.txt
   ```

4. **Configuration**:
    The public configuration information is stored in the [config.py](./config.py) file. The global parameters in the config file specified can be changed if another value is desired for the project.

    Default values:
    ```python
    dataset_directory = "dataset"
    chroma_persist_directory = "chroma_db"
    huggingface_embedding_model_repo_path = "sentence-transformers/all-mpnet-base-v2"
    huggingface_vector_embedding_database_repo_path = "syildizz/user-manuals-chromadb"
    google_llm_model_name = "gemini-2.5-flash"
    ```

4. **Configure Environment Variables**:
   Create a `.env` file in the project root with the following:
   ```text
   GEMINI_API_KEY=[your-gemini-api-key]
   HUGGINGFACE_TOKEN=[your-huggingface-token]
   ```

5. **Generate Vector Database** (Optional):
   If you want to generate a local vector database, run:
   ```bash
   python generate_vector_db.py
   ```
   NOTE: Do not generate a vector database if you want to pull the public pregenerated database. 
   If a database does not exist in the next step, [app.py](./app.py) will pull the remote pregenerated database.

6. **Run the Application**:
   Launch the Gradio interface:
   ```bash
   python app.py
   ```
   The interface will be available at `http://localhost:7860`.

---

## Web Interface & Product Guide

The chatbot is deployed on HuggingFace Spaces at [Placeholder Link](https://huggingface.co/spaces/placeholder). 
The Gradio-based interface allows users to:

- Enter technical questions about electronic devices.
- Receive responses grounded in user manual content.

### Usage Instructions

1. Visit the HuggingFace Spaces link: [Placeholder Link](https://huggingface.co/spaces/placeholder).
2. Enter a question in the text input field (e.g., "How do I reset my [device name]?").
3. The chatbot will use relevant manual sections to generate a response.

### Screenshots

[Placeholder: Add screenshots or a short video demonstrating the interface]

---

**Live Demo**: [Placeholder HuggingFace Spaces Link](https://huggingface.co/spaces/placeholder)