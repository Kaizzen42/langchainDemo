import os
from apikey import cohere_apikey
from langchain.embeddings.cohere import CohereEmbeddings

import streamlit as st

from pdfminer.high_level import extract_text
from langchain.document_loaders import PyPDFLoader

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import Cohere
from langchain.chains import RetrievalQA
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma

# Setup environment
print(f"Embeddings loaded: Type={type(CohereEmbeddings)}")
os.environ['COHERE_API_KEY'] = cohere_apikey

# Prompt
st.title("🦜️🔗 LangChain Banu")
prompt = st.text_input("Wasssssuuuppppp Mate?")
st.write(prompt)

# Main
# Read PDF and convert to split docs
loader = PyPDFLoader("pdfs/uber_etas.pdf")
documents = loader.load()
documents = documents[0:3] #Cohere API free tier size limitation at 4096 tokens.
print(f"Documents len={len(documents)}")
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)
print(f"Texts len={len(texts)}")

# Init db
db = Chroma.from_documents(texts, CohereEmbeddings())
retriever = db.as_retriever()
llm = Cohere(cohere_api_key=cohere_apikey)

qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

query="What types of ETA prediction models does uber use?"
st.write(query)
res = qa.run(query)
st.write(res)