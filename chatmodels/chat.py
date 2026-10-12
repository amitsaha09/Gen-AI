from dotenv import load_dotenv

load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI

model = ChatGoogleGenerativeAI(model="gemini-3.7-flash")
response = model.invoke("give me a paragraph about AI?", temperature=0.7, max_output_tokens=200, max_tokens=20)

print(response.content[0]["text"])