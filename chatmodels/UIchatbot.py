import streamlit as st

st.title("AI Chatbot")

import logging

logging.getLogger("google_genai.models").setLevel(logging.ERROR)

from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage


@st.cache_resource
def get_model():
    return ChatGoogleGenerativeAI(model="gemini-2.5-flash")


model = get_model()

switcher = {
    "1": "sad",
    "2": "funny",
    "3": "angry",
}

# Step 1: choose the AI mode
if "messages" not in st.session_state:
    st.subheader("Choose your AI mode")
    for key, mode in switcher.items():
        if st.button(f"{mode.capitalize()} AI mode", key=f"mode_{key}", use_container_width=True):
            st.session_state.mode = mode
            st.session_state.messages = [
                SystemMessage(content=f"You are a {mode} Ai agent. and respond to the user in a {mode} way."),
            ]
            st.rerun()
    st.stop()

# Step 2: chat
st.caption(f"Mode: {st.session_state.mode}")

# Show previous messages (skip the system message)
for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.write(msg.content)
    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant"):
            st.write(msg.content)

# Chat input
if prompt := st.chat_input("Type your message..."):
    st.session_state.messages.append(HumanMessage(content=prompt))
    with st.chat_message("user"):
        st.write(prompt)

    response = model.invoke(st.session_state.messages)
    st.session_state.messages.append(AIMessage(content=response.content))
    with st.chat_message("assistant"):
        st.write(response.content)