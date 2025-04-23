# Required libraries
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path
from langchain_google_genai import GoogleGenerativeAIEmbeddings  # type: ignore
from langchain_qdrant import QdrantVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
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

def generate_less_abstract_queries(user_query):
    prompt = (
        f"Given user query: '{user_query}', generate 3 simpler or clearer versions of this query. "
        "These versions should not change the meaning, just make it decomposed and easier to understand."
        # We should add the example
        """
            Example: User query = "Why is Python commonly used in AI applications?"
            The three query should be like : 
            “What is Python?”
            “What is AI?”
            “Why is Python commonly used in AI applications?”
        """
    )
    sub_queries_response = chat_model.invoke(prompt)
    sub_queries = [line.strip("- ").strip() for line in sub_queries_response.content.split("\n") if line.strip()]
    return sub_queries

def retrieve_documents(sub_query):
    return retriever.similarity_search(query=sub_query, k=3)

def process_user_query_less_abstract(user_query):
    sub_queries = generate_less_abstract_queries(user_query)

    print("\n🧠 Generated Sub-Queries:")
    for idx, sq in enumerate(sub_queries, 1):
        print(f"  {idx}. {sq}")

    # Perform parallel retrieval
    all_retrieved = {}
    with ThreadPoolExecutor() as executor:
        future_to_subq = {executor.submit(retrieve_documents, sq): sq for sq in sub_queries}
        for future in future_to_subq:
            sub_query = future_to_subq[future]
            docs = future.result()
            all_retrieved[sub_query] = docs

    print("\n📄 Retrieved Results:")
    for idx, (sub_query, docs) in enumerate(all_retrieved.items(), 1):
        print(f"\n🔍 Sub-Query {idx}: {sub_query}")
        for i, doc in enumerate(docs, 1):
            print(f"  Result {i}: {doc.page_content[:200].strip()}...")  # Show preview

    # Combine all results (de-duplicate by content)
    combined_docs = []
    seen = set()
    for docs in all_retrieved.values():
        for doc in docs:
            if doc.page_content not in seen:
                seen.add(doc.page_content)
                combined_docs.append(doc.page_content)

    context = "\n\n".join(combined_docs[:5])  # Use top 5 for final context

    final_prompt = (
        f"You are given information extracted from a resume below:\n\n{context}\n\n"
        f"Now, based on the above context and the original user query:\n'{user_query}'\n"
        f"Give a detailed and helpful answer."
    )

    answer = chat_model.invoke(final_prompt)

    print("\n✅ Final Answer:")
    print(answer.content)


while True:
    user_query = input("\n🤖 Ask something about the resume (or type 'exit'): ")
    if user_query.lower() == "exit":
        break
    process_user_query_less_abstract(user_query)
