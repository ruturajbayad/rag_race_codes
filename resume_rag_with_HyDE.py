from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings  # type: ignore
from langchain_qdrant import QdrantVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from pathlib import Path

# Initialize necessary configurations
GEMINI_API_KEY = "your_api_key"

# Load PDF document
pdf_path = Path(__file__).parent / "resume.pdf"
loader = PyPDFLoader(file_path=pdf_path)

# Load data
docs = loader.load()


# Split text to handle large PDFs
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
split_docs = text_splitter.split_documents(documents=docs)

# Initialize the embedding model
embedder = GoogleGenerativeAIEmbeddings(model='models/text-embedding-004', google_api_key=GEMINI_API_KEY)

# Initialize the Qdrant vector store
retriever = QdrantVectorStore.from_existing_collection(
    url="http://localhost:6333",
    collection_name="learning_langchain",
    embedding=embedder
)

# Create chat model instance for generating hypothetical answers
chat_model = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-001",
    google_api_key=GEMINI_API_KEY
)

def generate_hypothetical_answer(user_query):
    """
    Generate a hypothetical answer based on the user query.
    """
    prompt = f"Given the query '{user_query}', generate a hypothetical answer that could be relevant to the resume data."
    hypothetical_answer = chat_model.invoke(prompt).content
    return hypothetical_answer

# Assuming embed_query method is available
def embed_and_search(hypothetical_answer):
    # Use the embed_query method to generate the embedding for the hypothetical answer
    embedded_answer = embedder.embed_query(hypothetical_answer)
    
    # Perform similarity search
    search_results = retriever.similarity_search(embedded_answer, k=3)
    
    return search_results

def process_user_query(user_query):
    """
    Process the user query using the HyDE pipeline.
    """
    # Step 1: Generate a hypothetical answer from the LLM
    hypothetical_answer = generate_hypothetical_answer(user_query)
    print(f"Hypothetical Answer: {hypothetical_answer}")
    
    # Step 2: Embed the hypothetical answer and perform similarity search
    search_results = embed_and_search(hypothetical_answer)
    
    # Step 3: Combine search results and hypothetical answer to generate the final response
    context = "\n\n".join([result.page_content for result in search_results])
    prompt = f"Based on the following context from the resume:\n\n{context}\n\nAnswer this: {user_query}"
    
    # Step 4: Get the final response from the LLM
    final_answer = chat_model.invoke(prompt).content
    print(f"Final Answer: {final_answer}")

# Main Loop to interact with the user
while True:
    user_query = input("\nAsk something about the resume (or 'exit' to quit): ")
    if user_query.lower() == 'exit':
        break
    process_user_query(user_query)
