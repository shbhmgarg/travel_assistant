import os
import re
import glob

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

load_dotenv() 

DATA_DIR = "data"
VECTORSTORE_DIR = "vectorstore"
EMBED_MODEL = "models/gemini-embedding-001" 


def parse_frontmatter(text: str):
    match = re.match(r"^---\n(.*?)\n---\n(.*)$", text, re.DOTALL)
    if not match:
        return {}, text

    frontmatter_raw, body = match.groups()
    metadata = {}
    for line in frontmatter_raw.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        metadata[key.strip()] = value.strip().strip('"')
    return metadata, body


def load_documents() -> list[Document]:
    documents = []
    for filepath in sorted(glob.glob(os.path.join(DATA_DIR, "*.md"))):
        with open(filepath, "r", encoding="utf-8") as f:
            raw_text = f.read()

        metadata, body = parse_frontmatter(raw_text)
        metadata["file"] = os.path.basename(filepath)
        documents.append(Document(page_content=body, metadata=metadata))

    return documents


def build_vectorstore():
    print("Loading documents from", DATA_DIR)
    raw_docs = load_documents()
    print(f"Loaded {len(raw_docs)} source documents")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n## ", "\n### ", "\n\n", "\n", " "],
    )
    chunks = splitter.split_documents(raw_docs)
    print(f"Split into {len(chunks)} chunks")

    print(f"Embedding {len(chunks)} chunks with Gemini ({EMBED_MODEL})...")
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBED_MODEL)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    
    vectorstore.save_local(VECTORSTORE_DIR)
    print(f"Saved FAISS index to {VECTORSTORE_DIR}/")


if __name__ == "__main__":
    build_vectorstore()