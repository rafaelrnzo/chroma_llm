import os
import chromadb
import ollama
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

OLLAMA_HOST = "http://192.168.100.3:11434"
OLLAMA_MODEL = "llama3.2:latest"
ollama.base_url = OLLAMA_HOST

chroma_client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT, settings=Settings())
embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

file_path = r"C:\Users\SD-LORENZO-PC\pyproject\rndML\rndVectorDB\ChromaLlama\com.pdf"

if not os.path.exists(file_path):
    raise FileNotFoundError(f"File not found: {file_path}")

loader = PyPDFLoader(file_path)
document = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
chunked_documents = text_splitter.split_documents(document)

vectorstore = Chroma.from_documents(
    documents=chunked_documents,
    embedding=embedding_function,
    collection_name=CHROMA_COLLECTION_NAME,
    client=chroma_client,
)

print(f"Added {len(chunked_documents)} chunks to ChromaDB in collection '{CHROMA_COLLECTION_NAME}'")

def ollama_llm(question, context):
    formatted_prompt = f"Question: {question}\n\nContext: {context}"
    try:
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[{'role': 'user', 'content': formatted_prompt}]
        )
        return response['message']
    except Exception as e:
        return f"Error: {e}"

retriever = vectorstore.as_retriever()

def combine_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def rag_chain(question):
    retrieved_docs = retriever.invoke(question)
    formatted_context = combine_docs(retrieved_docs)
    return ollama_llm(question, formatted_context)


question = input("Input Prompt: ")
response = rag_chain(question)
print(f"Question: {question}\nResponse: {response}")