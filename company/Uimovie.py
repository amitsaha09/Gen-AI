import streamlit as st

st.title("Movie Information Extractor")

import logging

logging.getLogger("google_genai.models").setLevel(logging.ERROR)

from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate


@st.cache_resource
def get_model():
    return ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")


model = get_model()

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a movie information extraction assistant.

Your job is to extract useful information from a movie-related paragraph.

Extract:
- Movie title
- Genre
- Release year
- Director
- Main actors or characters
- Setting
- Plot
- Main conflict
- Important events
- Themes

Rules:
- Use only information explicitly provided.
- Do not make up or assume information.
- If information is not mentioned, write "Not mentioned".
- Keep the extracted information concise and clear."""
        ),
        (
            "human",
            """Extract useful information from the following movie paragraph:

{text}"""
        )
    ]
)


def get_text(content):
    """The model may return a plain string or a list of content blocks."""
    if isinstance(content, str):
        return content
    return "".join(
        block.get("text", "") if isinstance(block, dict) else str(block)
        for block in content
    )


para = st.text_area("Enter a movie-related paragraph:", height=200)

if st.button("Extract"):
    if para.strip():
        with st.spinner("Extracting..."):
            final_prompt = prompt.invoke({"text": para})
            response = model.invoke(final_prompt)
        st.markdown(get_text(response.content))
    else:
        st.warning("Please enter a paragraph first.")