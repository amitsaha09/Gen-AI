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

model = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite"
)

prompt = ChatPromptTemplate.from_messages([
    ('system', """Extract movie information from the paragraph 
     {format_instructions}"""),
     ("human", "{paragraph}")]
)


para = input("Enter a movie-related paragraph: ")

final_prompt = prompt.invoke(
    {
        "paragraph": para,
        "format_instructions": parser.get_format_instructions()
    }
)

response = model.invoke(final_prompt)

print(response.content)