import logging

logging.getLogger("google_genai.models").setLevel(logging.ERROR)

from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import  AIMessage, HumanMessage, SystemMessage
model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash"
)

print("Choose your AI mode")
print("press '1' for sad AI mode")
print("press '2' for funny AI mode")
print("press '3' for angry AI mode")

choice = input("Enter your choice: ")
switcher = {
    "1": "sad",
    "2": "funny",
    "3": "angry"
}
mode = switcher.get(choice, "sad")
messages = [
    SystemMessage(content=f"You are a {mode} Ai agent. and respond to the user in a {mode} way."),
]

print("Type '0' to quit.")

while True:
    prompt = input("you: ")
    messages.append(HumanMessage(content=prompt))
    if prompt == "0":
        break

    response = model.invoke(messages)
    messages.append(AIMessage(content=response.content))
    print("bot:", response.content)

print(messages)