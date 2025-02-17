import os
import chromadb
from chromadb.config import Settings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter

CHROMA_HOST = "192.168.100.3"
CHROMA_PORT = 8000
CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "default_collection")

chroma_client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT, settings=Settings())
embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

file_path = r"C:\Users\SD-LORENZO-PC\pyproject\rndML\rndVectorDB\ChromaLlama\data\1-PROPULSION\A2.SistemOliMinyakLincir(MPKPortsideStarboard)\1-I-A2-6.pdf"

if not os.path.exists(file_path):
    raise FileNotFoundError(f"File not found: {file_path}")

loader = PyPDFLoader(file_path)
document = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
chunked_documents = text_splitter.split_documents(document)

Chroma.from_documents(
    documents=chunked_documents,
    embedding=embedding_function,
    collection_name=CHROMA_COLLECTION_NAME,
    client=chroma_client,
)

print(f"Added {len(chunked_documents)} chunks to ChromaDB in collection '{CHROMA_COLLECTION_NAME}'")
