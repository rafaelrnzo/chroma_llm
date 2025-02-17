import os
import chromadb
import ollama
from chromadb.config import Settings
from langchain_chroma import Chroma
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)

CHROMA_HOST = "192.168.100.3"
CHROMA_PORT = 8000
CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "default_collection")

OLLAMA_HOST = "http://192.168.100.3:11434"
OLLAMA_MODEL = "llama3.2:latest"
ollama.base_url = OLLAMA_HOST

chroma_client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT, settings=Settings())
embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

vectorstore = Chroma(
    embedding_function=embedding_function,
    collection_name=CHROMA_COLLECTION_NAME,
    client=chroma_client,
)
retriever = vectorstore.as_retriever()

def combine_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

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

def rag_chain(question):
    retrieved_docs = retriever.invoke(question)
    formatted_context = combine_docs(retrieved_docs)
    return ollama_llm(question, formatted_context)

question = input("Input Query: ")
response = rag_chain(question)

print(f"Query: {question}\nResponse:\n{response}")
