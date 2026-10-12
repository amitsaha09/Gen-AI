import streamlit as st

st.title("Movie Information Extractor")

import logging

logging.getLogger("google_genai.models").setLevel(logging.ERROR)

from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
from typing import List, Optional
from langchain_core.output_parsers import PydanticOutputParser


class MovieInfo(BaseModel):
    title: str
    genre: List[str]
    release_year: Optional[int]
    director: Optional[str]
    cast: List[str]
    rating: Optional[float]
    summary: str


parser = PydanticOutputParser(pydantic_object=MovieInfo)


@st.cache_resource
def get_model():
    return ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")


model = get_model()

prompt = ChatPromptTemplate.from_messages([
    ('system', """Extract movie information from the paragraph 
     {format_instructions}"""),
     ("human", "{paragraph}")]
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
            final_prompt = prompt.invoke(
                {
                    "paragraph": para,
                    "format_instructions": parser.get_format_instructions()
                }
            )
            response = model.invoke(final_prompt)
            text = get_text(response.content)

        try:
            movie = parser.parse(text)
            st.subheader(movie.title)
            st.write(f"**Genre:** {', '.join(movie.genre) or 'Not mentioned'}")
            st.write(f"**Release year:** {movie.release_year or 'Not mentioned'}")
            st.write(f"**Director:** {movie.director or 'Not mentioned'}")
            st.write(f"**Cast:** {', '.join(movie.cast) or 'Not mentioned'}")
            st.write(f"**Rating:** {movie.rating if movie.rating is not None else 'Not mentioned'}")
            st.write(f"**Summary:** {movie.summary}")
        except Exception:
            # If the model's output can't be parsed, show it as-is
            st.warning("Couldn't parse the response, showing the raw output:")
            st.code(text)
    else:
        st.warning("Please enter a paragraph first.")