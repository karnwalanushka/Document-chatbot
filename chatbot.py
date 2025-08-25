# from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, UnstructuredPowerPointLoader
# from langchain.text_splitter import CharacterTextSplitter
# from langchain.vectorstores import FAISS
# from langchain.embeddings import HuggingFaceEmbeddings
# from transformers import pipeline
# from pathlib import Path

# # -------------------------------
# # 1. Documents Load
# # -------------------------------
# def load_documents():
#     docs = []
#     data_path = Path("data")   # "data" folder me documents rakho

#     for file in data_path.glob("**/*"):
#         if file.suffix.lower() == ".pdf":
#             loader = PyPDFLoader(str(file))
#             docs.extend(loader.load())
#         elif file.suffix.lower() == ".docx":
#             loader = Docx2txtLoader(str(file))
#             docs.extend(loader.load())
#         elif file.suffix.lower() in [".ppt", ".pptx"]:
#             loader = UnstructuredPowerPointLoader(str(file))
#             docs.extend(loader.load())

#     return docs

# # -------------------------------
# # 2. Split into Chunks
# # -------------------------------
# def split_documents(docs):
#     splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
#     return splitter.split_documents(docs)

# # -------------------------------
# # 3. Create Vector DB (Offline Embeddings)
# # -------------------------------
# def create_vector_store(splitted_docs):
#     embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
#     vector_store = FAISS.from_documents(splitted_docs, embeddings)
#     return vector_store

# # -------------------------------
# # 4. Offline QnA Pipeline
# # -------------------------------
# def build_qna_pipeline():
#     return pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

# # -------------------------------
# # Main Program
# # -------------------------------
# if __name__ == "__main__":
#     print("📄 Loading documents...")
#     docs = load_documents()
#     print(f"✅ Loaded {len(docs)} documents.")

#     splitted_docs = split_documents(docs)
#     vector_store = create_vector_store(splitted_docs)
#     qa_pipeline = build_qna_pipeline()

#     print("🤖 Offline Document Chatbot Ready! Type 'exit' to quit.\n")
#     while True:
#         query = input("You: ")
#         if query.lower() in ["exit", "quit"]:
#             print("👋 Goodbye!")
#             break

#         # retrieve relevant text
#         results = vector_store.similarity_search(query, k=2)
#         context = " ".join([doc.page_content for doc in results])

#         # run QnA model
#         answer = qa_pipeline(question=query, context=context)
#         print("Bot:", answer["answer"], "\n")


from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader, UnstructuredPowerPointLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import pipeline
from pathlib import Path

# ----------------------------------------------------
# 1. Documents Loader
# ---------------------------------------------------
def load_documents():
    docs = []
    data_path = Path("data")   # "data" folder me documents rakho

    for file in data_path.glob("**/*.*"):
        if file.suffix == ".pdf":
            loader = PyPDFLoader(str(file))
        elif file.suffix == ".txt":
            loader = TextLoader(str(file), encoding="utf-8")
        elif file.suffix == ".docx":
            loader = Docx2txtLoader(str(file))
        elif file.suffix in [".ppt", ".pptx"]:
            loader = UnstructuredPowerPointLoader(str(file))
        else:
            print(f"⚠️ Skipping unsupported file: {file}")
            continue

        docs.extend(loader.load())

    return docs


# ---------------------------------------------------
# 2. Load Documents
# ---------------------------------------------------
print("📄 Loading documents...")
raw_docs = load_documents()
print("✅ Loaded", len(raw_docs), "documents.")

if not raw_docs:
    print("❌ No documents found! Please put PDF, TXT, DOCX, or PPT files in the data/ folder.")
    exit()


# ---------------------------------------------------
# 3. Text Splitter
# ---------------------------------------------------
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
splitted_docs = text_splitter.split_documents(raw_docs)


# ---------------------------------------------------
# 4. Embeddings + FAISS
# ---------------------------------------------------
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = FAISS.from_documents(splitted_docs, embeddings)


# ---------------------------------------------------
# 5. QA Pipeline
# ---------------------------------------------------
qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

def ask_question(query):
    docs = vector_store.similarity_search(query, k=3)
    context = " ".join([d.page_content for d in docs])

    answer = qa_pipeline({
        "context": context,
        "question": query
    })
    return answer["answer"]


# ---------------------------------------------------
# 6. Chat Loop
# ---------------------------------------------------
print("\n🤖 Chatbot Ready! Type 'quit' to exit.\n")

while True:
    query = input("You: ")
    if query.lower() in ["quit", "exit", "bye"]:
        print("👋 Goodbye!")
        break

    response = ask_question(query)
    print("Bot:", response)


