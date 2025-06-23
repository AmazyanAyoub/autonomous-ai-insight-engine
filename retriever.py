# retriever.py
from pathlib import Path
from langchain_community.document_loaders import TextLoader, PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

CHROMA_DIR = "chroma"
DATA_DIR = Path("data")

embedding = OllamaEmbeddings(model="mahonzhan/all-MiniLM-L6-v2")

def load_documents():
    documents = []
    for file in DATA_DIR.glob("*"):
        if file.suffix == ".txt":
            loader = TextLoader(str(file), encoding="utf-8")
            documents.extend(loader.load())
        elif file.suffix == ".pdf":
            loader = PyMuPDFLoader(str(file))
            documents.extend(loader.load())
    return documents

def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    return splitter.split_documents(documents)

def get_vectorstore():
    
    if Path(CHROMA_DIR).exists():
        return Chroma(persist_directory=CHROMA_DIR, embedding_function=embedding)

    docs = load_documents()
    docs_split = split_documents(docs)
    vector_store = Chroma(
        embedding_function=embedding,
        persist_directory=CHROMA_DIR,
    )
    vector_store.add_documents(documents=docs_split)
    return vector_store
