import pandas as pd
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore

load_dotenv()

def load_and_preprocess(csv_path:str) -> list[Document]:
    df = pd.read_csv(csv_path)

    documents = []
    for _, row in df.iterrows():
        doc = Document(
            page_content=row['Resume_str'].strip(),
            metadata={
                "category": row['Category'],
                "id": int(row['ID'])
            }
        )
        documents.append(doc)

    print(f"Total dokumen: {len(documents)}")
    return documents

def chunk_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = splitter.split_documents(documents)
    print(f"Total chunks: {len(chunks)}")
    return chunks

def upload_to_qdrant(chunks):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    vectorstore = QdrantVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY"),
        collection_name="resume_collection"
    )

    print("Upload selesai!")
    return vectorstore

if __name__ == "__main__":
    docs = load_and_preprocess("data/Resume.csv")
    chunks = chunk_documents(docs)
    vectorstore = upload_to_qdrant(chunks)

    # Cek sample chunk pertama 
    print("\nSample chunk pertama:")
    print(chunks[0].page_content)
    print("\nMetadata-nya:")
    print(chunks[0].metadata)