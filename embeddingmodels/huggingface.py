from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

text = [
    "You are going to Gen Ai",
    "You are going to teach me",
    "hello world"
]

vector = embeddings.embed_documents(text)

print(vector)