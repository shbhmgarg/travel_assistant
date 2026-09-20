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