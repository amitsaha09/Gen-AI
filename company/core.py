import logging

logging.getLogger("google_genai.models").setLevel(logging.ERROR)

from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate


model = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite"
)

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

para = input("Enter a movie-related paragraph: ")

final_prompt = prompt.invoke(
    {"text": para}
)

response = model.invoke(final_prompt)

print(response.content)