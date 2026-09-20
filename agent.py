"""
agent.py - The combined RAG + MCP travel assistant agent.

This ties together three things:
  1. The RAG tool (search_travel_knowledge_base) - grounded answers from
     our Singapore knowledge base (the .md files we embedded into FAISS).
  2. The MCP tools (get_weather_forecast, convert_currency) - live data
     fetched from mcp_server.py over the Model Context Protocol.
  3. A tool-calling LLM agent (Gemini) that decides, per user question,
     which of these tools (if any) it needs to call before answering.

Why create_agent() instead of create_tool_calling_agent()+AgentExecutor:
LangChain v1 replaced the old create_tool_calling_agent/AgentExecutor
pattern (and even langgraph's create_react_agent) with a single unified
factory: langchain.agents.create_agent(model, tools, system_prompt=...).
It returns a ready-to-run agent (a compiled LangGraph graph under the
hood) that you call with .invoke({"messages": [...]}) or
.ainvoke({"messages": [...]}) - no separate "executor" object needed.
"""

import asyncio
import logging
import warnings

# --- Quiet down noisy but harmless library chatter --------------------------
# 1. Python "warnings" (DeprecationWarning, UserWarning) - these are one-line
#    notices from libraries (e.g. "this import path is being sunset", or
#    "this parameter is ignored by this model"). They don't affect behavior,
#    so we filter them out before importing the libraries that raise them.
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# 2. The `mcp` library logs every request it handles (ListToolsRequest,
#    CallToolRequest, ...) at INFO level. Raising its logger to WARNING
#    keeps real errors visible but drops this routine chatter.
logging.getLogger("mcp").setLevel(logging.WARNING)
logging.getLogger("mcp.server").setLevel(logging.WARNING)
logging.getLogger("mcp.client").setLevel(logging.WARNING)

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_community.vectorstores import FAISS
from langchain_core.messages import AIMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_mcp_adapters.client import MultiServerMCPClient

load_dotenv()

VECTORSTORE_DIR = "vectorstore"
EMBED_MODEL = "models/gemini-embedding-001"
CHAT_MODEL = "gemini-3.6-flash"

SYSTEM_PROMPT = """You are a helpful, careful Singapore travel-planning assistant.

You have access to these tools:
- search_travel_knowledge_base: search a curated knowledge base of Singapore
  travel guides (attractions, itineraries, neighbourhoods, essential travel
  info). ALWAYS use this tool for questions about places to visit, things to
  do, itineraries, or general Singapore travel facts. Do not answer these
  from memory.
- get_weather_forecast: get LIVE current conditions and a multi-day forecast
  for a city. ALWAYS use this tool for any question about weather, rain,
  temperature, or what to pack/plan around. This will only work for Singapore.
- convert_currency: convert an amount between currencies using LIVE exchange
  rates. ALWAYS use this tool for any budget or currency conversion question.

Rules:
1. Never state a specific fact (an attraction, a price, a weather condition,
   an exchange rate) unless it came from one of your tools. If a tool
   returns no relevant result, say so honestly instead of guessing.
2. When you use the knowledge base, mention which source(s) the information
   came from (the tool result includes "[Source: ...]" labels).
3. Clearly distinguish between:
   - facts from the knowledge base (label as "According to <source>...")
   - live data from a tool (label as "Current weather..." / "Live exchange
     rate...")
   - your own general suggestions/opinions (label as "My suggestion...")
4. Remember details the user has already told you earlier in the
   conversation (budget, dates, preferences) and use them in later answers
   without asking again.
5. Keep answers well-structured (short paragraphs or bullet points) and
   focused on what was asked.
"""


def extract_text(content) -> str:
    """Pull plain text out of a LangChain message's `content`.

    Gemini (and other newer chat models) can return `content` as either:
      - a plain string, or
      - a list of content blocks, e.g.
        [{"type": "text", "text": "..."}, {"type": "...", "signature": "..."}]
    We only want the human-readable text blocks, joined together.
    """
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "\n".join(p for p in parts if p)

    return str(content)


def build_rag_tool():
    """Load the FAISS vector store and wrap retrieval in a LangChain @tool."""
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBED_MODEL)
    vectorstore = FAISS.load_local(
        VECTORSTORE_DIR, embeddings, allow_dangerous_deserialization=True
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    @tool
    def search_travel_knowledge_base(query: str) -> str:
        """Search the Singapore travel knowledge base for information about
        attractions, itineraries, neighbourhoods, and essential travel info.
        Use this for any question about places, things to do, or general
        Singapore travel facts."""
        docs = retriever.invoke(query)
        if not docs:
            return "NO_RESULTS: nothing relevant found in the knowledge base."

        blocks = []
        for doc in docs:
            source = doc.metadata.get("source_title", "Unknown source")
            blocks.append(f"[Source: {source}]\n{doc.page_content}")
        return "\n\n---\n\n".join(blocks)

    return search_travel_knowledge_base


async def build_agent():
    """Build the full agent: RAG tool + MCP tools + Gemini model."""
    rag_tool = build_rag_tool()

    mcp_client = MultiServerMCPClient(
        {
            "travel_tools": {
                "command": "python",
                "args": ["mcp_server.py"],
                "transport": "stdio",
            }
        }
    )
    mcp_tools = await mcp_client.get_tools()

    all_tools = [rag_tool] + mcp_tools

    llm = ChatGoogleGenerativeAI(model=CHAT_MODEL, temperature=0.2)

    agent = create_agent(llm, tools=all_tools, system_prompt=SYSTEM_PROMPT)
    return agent


class TravelAssistant:
    """Wraps the agent with simple multi-turn chat history."""

    def __init__(self):
        self._agent = None
        self.chat_history: list = []

    async def _ensure_agent(self):
        if self._agent is None:
            self._agent = await build_agent()

    async def achat(self, user_input: str) -> str:
        await self._ensure_agent()
        messages = self.chat_history + [HumanMessage(content=user_input)]
        result = await self._agent.ainvoke({"messages": messages})
        final_message = result["messages"][-1]
        response = extract_text(final_message.content)

        self.chat_history.append(HumanMessage(content=user_input))
        self.chat_history.append(AIMessage(content=response))
        return response

    def chat(self, user_input: str) -> str:
        return asyncio.run(self.achat(user_input))


async def _main():
    assistant = TravelAssistant()
    print("Singapore Travel Assistant (RAG + MCP). Type 'exit' to quit.\n")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("exit", "quit"):
            break
        if not user_input:
            continue
        response = await assistant.achat(user_input)
        print(f"\nAssistant: {response}\n")


if __name__ == "__main__":
    asyncio.run(_main())