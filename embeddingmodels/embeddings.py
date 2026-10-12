from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

load_dotenv()
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",
    dimensions= 64
    )
vector = embeddings.embed_query("You are going to Gen Ai")
print(vector)