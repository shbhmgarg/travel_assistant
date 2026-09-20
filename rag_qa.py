
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

load_dotenv()

VECTORSTORE_DIR = "vectorstore"
EMBED_MODEL = "models/gemini-embedding-001"
CHAT_MODEL = "gemini-3.6-flash"

RAG_PROMPT_TEMPLATE = """You are a helpful Singapore travel assistant.

Answer the QUESTION using ONLY the information in the CONTEXT below.
Do not use any outside knowledge. If the context doesn't contain enough
information to answer, say so plainly instead of guessing.

After your answer, list which source(s) you used, by title.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:"""


def format_context(docs) -> str:
    blocks = []
    for doc in docs:
        title = doc.metadata.get("source_title", "Unknown source")
        blocks.append(f"[Source: {title}]\n{doc.page_content}")
    return "\n\n---\n\n".join(blocks)


def main():
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBED_MODEL)
    vectorstore = FAISS.load_local(
        VECTORSTORE_DIR, embeddings, allow_dangerous_deserialization=True
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    llm = ChatGoogleGenerativeAI(model=CHAT_MODEL, temperature=0.2)

    print("Ask a question about Singapore travel (or 'exit' to quit).\n")
    while True:
        question = input("You: ")
        if question.strip().lower() in ("exit", "quit"):
            break

        # 1. Retrieve relevant chunks
        docs = retriever.invoke(question)
        context = format_context(docs)

        # 2. Build the grounded prompt
        prompt = RAG_PROMPT_TEMPLATE.format(context=context, question=question)

        # 3. Ask the LLM to answer using only that context
        response = llm.invoke(prompt)

        print("\nAssistant:", response.content, "\n")


if __name__ == "__main__":
    main()