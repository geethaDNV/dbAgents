#from langchain.schema import HumanMessage, SystemMessage
from langchain_core.messages import HumanMessage, SystemMessage
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import ssl
import httpx

os.environ["PYTHONHTTPSVERIFY"] = "0"
# Load environment variables from .env file
load_dotenv()

openai_key = os.getenv("OPENAI_API_KEY")

http_client = httpx.Client(verify=False)

#llm_name = "gpt-3.5-turbo"
#model = ChatOpenAI(api_key=openai_key, model=llm_name)

model = ChatOpenAI(
    api_key=openai_key,
    model="openai/gpt-4.1",
    base_url="https://models.github.ai/inference",
    http_client=http_client 
)

messages = [
    SystemMessage(
        content="You are a helpful assistant who is extremely competent as a Computer Scientist! Your name is Rob."
    ),
    HumanMessage(content="who was the very first computer scientist?"),
]


# res = model.invoke(messages)
# print(res)


def first_agent(messages):
    res = model.invoke(messages)
    return res


def run_agent():
    print("Simple AI Agent: Type 'exit' to quit")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Goodbye!")
            break
        print("AI Agent is thinking...")
        messages = [HumanMessage(content=user_input)]
        response = first_agent(messages)
        print("AI Agent: getting the response...")
        print(f"AI Agent: {response.content}")


if __name__ == "__main__":
    run_agent()
