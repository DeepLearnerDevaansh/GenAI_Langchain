import os
from dotenv import load_dotenv
import streamlit as st
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain.prompts import PromptTemplate





load_dotenv()  # Loads environment variables from .env file


api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if api_token is None:
    raise ValueError("HUGGINGFACEHUB_API_TOKEN is not set in the environment")

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    huggingfacehub_api_token=api_token
)

model = ChatHuggingFace(llm=llm)




st.header('Reasearch Tool')

# Dropdown to select research paper name
paper_input = st.selectbox(
    "Select Research Paper Name",
    [
        "Attention Is All You Need",
        "BERT: Pre-training of Deep Bidirectional Transformers",
        "GPT-3: Language Models are Few-Shot Learners",
        "Diffusion Models Beat GANs on Image Synthesis"
    ]
)

# Dropdown to select explanation style
style_input = st.selectbox(
    "Select Explanation Style",
    ["Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical"]
)

# Dropdown to select explanation length
length_input = st.selectbox(
    "Select Explanation Length",
    ["Short (1-2 paragraphs)", "Medium (3-5 paragraphs)", "Long (6+ paragraphs)"]
)



template = """
You are an AI research assistant. Use the following details to tailor your response:

Research Paper: {paper}
Explanation Style: {style}
Explanation Length: {length}

User Query:
{query}

If there are any mathematical details or analogies relevant to the topic in the paper, please include and explain them clearly.

Provide a detailed and well-structured answer.
"""

prompt = PromptTemplate(
    input_variables=["paper", "style", "length", "query"],
    template=template,
    validate_template=True
)




if st.button('summmarize'):
    result = model.invoke(paper_input)
    st.write(result.content)






