# pip install langchain_community pypdf
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path
from langchain_google_genai import GoogleGenerativeAIEmbeddings # type: ignore
from langchain_qdrant import QdrantVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
import os

GEMINI_API_KEY = "AIzaSyCz9zx5Vm97tlqkIBHMetCjqGYKqjYsCEQ"

# Loade PDF
pdf_path = Path(__file__).parent / "resume.pdf"
loader = PyPDFLoader(file_path=pdf_path)

# Show Data
docs= loader.load()
# print(docs[0]) 

# Now we need the splitter because PDF have multiple records 
# And we can't pass all the data right so we need splitters

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap=200,
)

split_docs = text_splitter.split_documents(documents=docs)

# Chunking is done now 
#! --------------------------------------
# Embedding
embedder = GoogleGenerativeAIEmbeddings(model='models/text-embedding-004', google_api_key=GEMINI_API_KEY)

# vector_store = QdrantVectorStore.from_documents(
#     documents=[],
#     url="http://localhost:6333",
#     collection_name="learning_langchain",
#     embedding=embedder
# )


# vector_store.add_documents(documents=split_docs)

retriever = QdrantVectorStore.from_existing_collection(
    url="http://localhost:6333",
    collection_name="learning_langchain",
    embedding=embedder
)

chat_model = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-001",
    google_api_key=GEMINI_API_KEY
)

while True:
    user_query = input("\n🤖 Ask something about the resume (or 'exit'): ")
    if user_query.lower() == "exit":
        break

    # Do a similarity search in your vector DB
    results = retriever.similarity_search(query=user_query, k=3)

    # Optional: use Gemini to generate final answer
    context = "\n\n".join([doc.page_content for doc in results])
    prompt = f"Given the following context from a resume:\n\n{context}\n\nAnswer this: {user_query}"

    answer = chat_model.invoke(prompt)

    print(f"\n🧠 Answer:\n{answer.content}")



# invoke() Function

# Converts the string into a chat message format (like {role: "user", content: "Tell me a joke"})

# Sends it to the LLM

# Receives the response

# Returns a ChatMessage object (usually AIMessage)