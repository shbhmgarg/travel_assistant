"""
Quick sanity check: does the vector store actually retrieve relevant
chunks for a question? No AI generation involved yet — just raw
similarity search, so we can verify this layer works in isolation.

Run with:
    python test_retrieval.py
"""
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

VECTORSTORE_DIR = "vectorstore"
EMBED_MODEL = "models/gemini-embedding-001"


def main():
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBED_MODEL)

    # allow_dangerous_deserialization=True is required because loading a
    # FAISS index involves unpickling Python objects. This is safe here
    # because we created this file ourselves in the previous step — you
    # should only pass this flag for vectorstores you trust the origin of.
    vectorstore = FAISS.load_local(
        VECTORSTORE_DIR, embeddings, allow_dangerous_deserialization=True
    )

    # k=3 means "give me the 3 most similar chunks"
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    test_questions = [
        "What are the must-visit attractions in Singapore?",
        "How do I get around Singapore without a car?",
        "What should I do if it's raining?",
    ]

    for question in test_questions:
        print("=" * 70)
        print("QUESTION:", question)
        print("=" * 70)

        results = retriever.invoke(question)
        for i, doc in enumerate(results, start=1):
            title = doc.metadata.get("source_title", "Unknown")
            print(f"\n--- Result {i} (from: {title}) ---")
            print(doc.page_content[:300], "...")
        print("\n")


if __name__ == "__main__":
    main()