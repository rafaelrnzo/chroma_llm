import chromadb
from chromadb.config import Settings

CHROMA_HOST = "192.168.100.3"
CHROMA_PORT = 8000
CHROMA_COLLECTION_NAME = "default_collection"

def delete_all_chroma_data():
    try:
        chroma_client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT, settings=Settings())
        chroma_client.delete_collection(name=CHROMA_COLLECTION_NAME)
        print(f"Collection '{CHROMA_COLLECTION_NAME}' deleted successfully!")
    except Exception as e:
        print(f"Error deleting ChromaDB collection: {e}")

if __name__ == "__main__":
    delete_all_chroma_data()
