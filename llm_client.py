import os
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv(dotenv_path=r"C:\Users\Patel\OneDrive\Desktop\LangChain\Models\.env")
print("After load_dotenv, token is:", os.getenv("HUGGINGFACEHUB_API_TOKEN"))


def get_model():
    api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if not api_token:
        raise ValueError("HUGGINGFACEHUB_API_TOKEN is not set")

    llm = HuggingFaceEndpoint(
        repo_id="meta-llama/Llama-3.1-8B-Instruct",
        task="text-generation",
        huggingfacehub_api_token=api_token,
    )
    return ChatHuggingFace(llm=llm)


def ask_model():
    return get_model()  # no parameters, returns model instance



print("Token:", os.getenv("HUGGINGFACEHUB_API_TOKEN"))
