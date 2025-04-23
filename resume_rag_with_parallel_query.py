# Required libraries
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path
from langchain_google_genai import GoogleGenerativeAIEmbeddings  # type: ignore
from langchain_qdrant import QdrantVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from concurrent.futures import ThreadPoolExecutor

GEMINI_API_KEY = "AIzaSyCz9zx5Vm97tlqkIBHMetCjqGYKqjYsCEQ"

# Load PDF document
pdf_path = Path(__file__).parent / "resume.pdf"
loader = PyPDFLoader(file_path=pdf_path)

# Load data
docs = loader.load()

# Split text to handle large PDFs
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
split_docs = text_splitter.split_documents(documents=docs)

# Initialize embedding model
embedder = GoogleGenerativeAIEmbeddings(model='models/text-embedding-004', google_api_key=GEMINI_API_KEY)

# Initialize the Qdrant vector store
retriever = QdrantVectorStore.from_existing_collection(
    url="http://localhost:6333",
    collection_name="learning_langchain",
    embedding=embedder
)

# Create chat model instance
chat_model = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-001",
    google_api_key=GEMINI_API_KEY
)

def generate_queries(user_query):
    # Use the LLM to generate sub-queries based on the user query
    prompt = f"Given the query '{user_query}', generate three different sub-queries to help retrieve more specific information related to the query. Each sub-query should focus on a different aspect of the query."
    sub_queries_response = chat_model.invoke(prompt)
    sub_queries = sub_queries_response.content.split("\n")  # Assuming each query is on a new line
    return sub_queries

def retrieve_documents(sub_query):
    # Perform similarity search for each sub-query on the same collection
    return retriever.similarity_search(query=sub_query, k=3)

def process_user_query(user_query):
    # 1. Generate sub-queries dynamically using the AI
    sub_queries = generate_queries(user_query)

    # 2. Perform similarity search for each sub-query in parallel
    with ThreadPoolExecutor() as executor:
        # Use ThreadPoolExecutor to run multiple queries in parallel
        future_to_query = {executor.submit(retrieve_documents, sub_query): sub_query for sub_query in sub_queries}
        results = []
        for future in future_to_query:
            results += future.result()

    # 3. Combine the results with the original user query to provide context
    context = "\n\n".join([doc.page_content for doc in results])
    prompt = f"Given the following context from a resume:\n\n{context}\n\nAnswer this: {user_query}"

    # 4. Get the AI-generated response based on the combined context
    answer = chat_model.invoke(prompt)

    print(f"\n🧠 Answer:\n{answer.content}")

while True:
    user_query = input("\n🤖 Ask something about the resume (or 'exit'): ")
    if user_query.lower() == "exit":
        break

    process_user_query(user_query)