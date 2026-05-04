import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore

load_dotenv(Path(__file__).parent.parent / '.env')

def get_vectorstore():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = QdrantVectorStore.from_existing_collection(
        embedding=embeddings,
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY"),
        collection_name="resume_collection"
    )
    return vectorstore

@tool
def semantic_search(query: str) -> str:
    """Search for relevant resumes based on skills, experience, or job description.
    Use this when the HR wants to find candidates by describing what they're looking for.
    Example: 'find data scientist with Python and machine learning experience'
    """
    vectorstore = get_vectorstore()
    results = vectorstore.similarity_search(query, k=5)
    
    if not results:
        return "No relevant candidates found for your query."
    
    output = f"Found {len(results)} relevant candidates:\n\n"
    for i, doc in enumerate(results, 1):
        output += f"Candidate {i}:\n"
        output += f"Category: {doc.metadata.get('category', 'Unknown')}\n"
        output += f"Resume excerpt: {doc.page_content[:300]}...\n\n"
    
    return output

@tool
def filter_by_category(category: str, keyword: str = "") -> str:
    """Filter resumes by job category. Optionally add a keyword to narrow results.
    Use this when HR specifies an exact job category.
    Available categories: INFORMATION-TECHNOLOGY, DATA-SCIENCE, FINANCE, HR, 
    MARKETING, SALES, ENGINEERING, HEALTHCARE, BANKING, ACCOUNTANT, etc.
    Example: category='HR', keyword='recruitment'
    """
    vectorstore = get_vectorstore()
    
    query = f"{category} {keyword}".strip()
    results = vectorstore.similarity_search(
        query,
        k=5,
        filter={"must": [{"key": "metadata.category", "match": {"value": category}}]}
    )
    
    if not results:
        return f"No candidates found in category '{category}'."
    
    output = f"Found {len(results)} candidates in '{category}':\n\n"
    for i, doc in enumerate(results, 1):
        output += f"Candidate {i}:\n"
        output += f"Resume excerpt: {doc.page_content[:300]}...\n\n"
    
    return output