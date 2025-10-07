from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from tqdm import tqdm  # progress bar

from dotenv import load_dotenv
load_dotenv()

# === Step 1: Load raw PDF(s) ===
DATA_PATH = "data/"

def load_pdf_files(data):
    print("📂 Loading PDF files...")
    loader = DirectoryLoader(
        data,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )
    documents = loader.load()
    print(f"✅ Loaded {len(documents)} pages from {DATA_PATH}")
    return documents

documents = load_pdf_files(data=DATA_PATH)

# === Step 2: Create Chunks ===
def create_chunks(extracted_data):
    print("✂️ Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = []
    for chunk in tqdm(text_splitter.split_documents(extracted_data), desc="Chunking"):
        chunks.append(chunk)
    print(f"✅ Created {len(chunks)} text chunks")
    return chunks

text_chunks = create_chunks(extracted_data=documents)

# === Step 3: Create Vector Embeddings (better model) ===
def get_embedding_model():
    print("🧠 Loading embedding model (all-mpnet-base-v2)...")
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    return embedding_model

embedding_model = get_embedding_model()

# === Step 4: Store embeddings in FAISS ===
print("💾 Generating embeddings and saving to FAISS DB...")
db = FAISS.from_documents(tqdm(text_chunks, desc="Embedding"), embedding_model)
DB_FAISS_PATH = "vectorstore/db_faiss"
db.save_local(DB_FAISS_PATH)
print(f"✅ Saved embeddings to {DB_FAISS_PATH}")
