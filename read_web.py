import ollama
import bs4
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings

ollama.base_url = "http://192.168.100.3:11434"

website = "https://www.marktechpost.com/2024/04/21/coconut-a-high-quality-large-scale-dataset-for-next-gen-segmentation-models/"

loader = WebBaseLoader(web_paths=[website], bs_kwargs={
    "parse_only": bs4.SoupStrainer(class_=["td-post-content", "td-post-header", "td-post-title"])
})

docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

embeddings = OllamaEmbeddings(model="llama3.2:latest")
vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
retriever = vectorstore.as_retriever()

def ollama_llm(question, context):
    formatted_prompt = f"Question: {question}\n\nContext: {context}"
    try:
        response = ollama.chat(
            model="llama3.2:latest",
            messages=[{"role": "user", "content": formatted_prompt}]
        )
        return response["message"]["content"]
    except Exception as e:
        return f"Error: {e}"

def combine_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def rag_chain(question):
    retrieved_docs = retriever.invoke(question)
    formatted_context = combine_docs(retrieved_docs)
    return ollama_llm(question, formatted_context)

question = "COCONut represents a significant step forward in modernizing the COCO benchmark is this true or not?"
result = rag_chain(question)

print(f"Question: {question}")
print(f"Response: {result}")