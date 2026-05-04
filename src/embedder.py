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
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, PointStruct
    import time
    
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    client = QdrantClient(
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY"),
        timeout=120
    )
    
    # Buat collection kalau belum ada
    try:
        client.get_collection("resume_collection")
        print("Collection sudah ada, lanjut upload...")
    except:
        client.create_collection(
            collection_name="resume_collection",
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
        )
        print("Collection baru dibuat!")
    
    # Upload per batch kecil
    batch_size = 20
    total = len(chunks)
    
    for i in range(0, total, batch_size):
        batch = chunks[i:i+batch_size]
        texts = [c.page_content for c in batch]
        metadatas = [c.metadata for c in batch]
        
        # Embed batch
        vectors = embeddings.embed_documents(texts)
        
        # Buat points
        points = [
            PointStruct(
                id=i+j,
                vector=vectors[j],
                payload={"page_content": texts[j], **metadatas[j]}
            )
            for j in range(len(batch))
        ]
        
        # Upload dengan retry
        for attempt in range(3):
            try:
                client.upsert(collection_name="resume_collection", points=points)
                print(f"Uploaded batch {i//batch_size + 1}/{(total//batch_size)+1}")
                break
            except Exception as e:
                print(f"Attempt {attempt+1} failed: {e}, retrying...")
                time.sleep(5)
    
    print("Upload selesai!")

if __name__ == "__main__":
    docs = load_and_preprocess("data/Resume.csv")
    chunks = chunk_documents(docs)
    vectorstore = upload_to_qdrant(chunks)

    # Cek sample chunk pertama 
    print("\nSample chunk pertama:")
    print(chunks[0].page_content)
    print("\nMetadata-nya:")
    print(chunks[0].metadata)