"""
streamlit_app.py - Basic chat UI for the Singapore Travel Assistant.

Concept: Streamlit re-runs this entire script top-to-bottom on every user
interaction (every message sent). Anything that needs to survive across
reruns - our TravelAssistant instance (which holds the loaded agent and
the conversation history) and the list of displayed messages - is stored
in st.session_state, a dict-like object Streamlit keeps alive for the
duration of the browser session.

Run with:
    streamlit run streamlit_app.py
"""

import asyncio

import streamlit as st

from agent import TravelAssistant

st.set_page_config(page_title="Singapore Travel Assistant", page_icon="🧭")
st.title("🧭 Singapore Travel Assistant")
st.caption("RAG (knowledge base) + MCP (live weather & currency tools), powered by Gemini")

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("Architecture")

    st.markdown(
        """
**RAG**
- FAISS vector store

**MCP**
- Custom Weather MCP
- Custom Weather Forecast MCP
- Custom Currency MCP

**LLM**
- gemini-flash-lite-latest
"""
    )

    st.divider()

    st.subheader("Knowledge Sources")

    st.markdown(
        """
- Visit Singapore - Enjoy Singapore in 7 Days
- Visit Singapore - Essential Travel Information
- Wikivoyage - Singapore Travel Guide
- Visit Singapore - Things To Do
- Smart Singapore - Transport Method
- Smart Singapore - Tourise paases
"""
    )

    st.divider()

    st.subheader("Try these questions")

    sample_questions = [
        "Create a 3-day Singapore itinerary for next week.",

        (
            "Make the itinerary suitable for a family with two "
            "children and suggest indoor alternatives if rain "
            "is expected."
        ),

        "What transportation options can we use?",

        "What is the weather in Singapore?",

        "How much is INR 50,000 in SGD?",

        (
            "I have a budget of INR 60,000. Convert it to SGD "
            "and suggest a three-day itinerary."
        ),

        (
            "Suggest outdoor attractions and replace them with "
            "indoor options if rain is expected."
        ),

        (
            "Plan a family trip and include the latest weather "
            "forecast."
        ),

        (
            "Create a cultural itinerary and show my budget of "
            "INR 60,000 in the destination currency."
        ),
    ]

    for question in sample_questions:
        st.caption("• " + question)

    st.divider()

    if st.button(
        "Clear conversation",
        use_container_width=True,
    ):
        st.session_state.messages = []
        st.rerun()

# --- Persist the assistant and message list across reruns -------------------
if "assistant" not in st.session_state:
    st.session_state.assistant = TravelAssistant()

if "messages" not in st.session_state:
    st.session_state.messages = []  # list of {"role": "user"/"assistant", "content": str}

# --- Render the conversation so far -----------------------------------------
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Handle new input ---------------------------------------------------------
user_input = st.chat_input("Ask about Singapore travel, weather, or currency...")

if user_input:
    # Show the user's message immediately.
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Get the assistant's reply. TravelAssistant.achat() is async because the
    # MCP tools and the LLM call are async; Streamlit callbacks are sync, so
    # we run it with asyncio.run() here, once per message.
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = asyncio.run(st.session_state.assistant.achat(user_input))
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})