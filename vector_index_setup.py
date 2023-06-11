from qdrant_client import QdrantClient
from langchain.vectorstores import Qdrant
from qdrant_client.models import Distance, VectorParams
from qdrant_client.models import PointStruct
from qdrant_client.models import Filter, FieldCondition, Range

import os
from apikey import cohere_apikey

import streamlit as st

from pdfminer.high_level import extract_text
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.chains import RetrievalQAChain, loadQARefineChain
from langchain.llms import Cohere
from langchain.chains.question_answering import load_qa_chain

import tensorflow_hub as hub

# Setup environment
embeddings = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

os.environ['COHERE_API_KEY'] = cohere_apikey

# Functions
def pdf_to_text(file_path):
    return extract_text(file_path)

def embed_documents(docs):
    content = [doc.page_content for doc in docs]
    print(f"content: Type={type(content[0])}, size = {len(content)}")
    print(f"Sample content: {content[0]}")
    return embeddings(content).numpy().tolist()

# Main
# Read PDF and convert to split docs
etas_text = pdf_to_text("pdfs/uber_etas.pdf")
print(f"Etas_text Paper: Type={type(etas_text)}, size = {len(etas_text)}")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
docs = text_splitter.create_documents([etas_text])
print(f"Created {len(docs)} docs.")

embedded_docs = embed_documents(docs[10:20])
print(f"embedded_docs: Type={type(embedded_docs)}, size = {len(embedded_docs)}")
print(f"embedded_doc sample: {embedded_docs[0]}")#, shape={embedded_docs[0].shape}")
# embedded_docs = embeddings.embed_documents(docs)

# Building the vector index
# https://github.com/qdrant/qdrant-client
# client = QdrantClient(url="http://localhost:6333")
client = QdrantClient(":memory:")
client.recreate_collection(
    collection_name="my_collection",
    vectors_config=VectorParams(size=512, distance=Distance.COSINE),
)
client.upsert(
    collection_name="my_collection",
    points=[
        PointStruct(
            id=idx,
            vector=vector,#.tolist(),
            payload={"color": "red", "rand_number": idx % 10}
        )
        for idx, vector in enumerate(embedded_docs) #vectors
    ]
)

print(f"Quad count = {client.count(collection_name='my_collection')}")
# qdrant = Qdrant.from_documents(
#     docs, embedded_docs,
#     host="localhost", port=6333,
#     collection_name="my_documents",
# )
# print(f"qdrant: Type={type(qdrant)}, size = {len(qdrant)}")

# Prompt
st.title("🦜️🔗 LangChain Banu")
prompt = st.text_input("Wasssssuuuppppp Mate?")
st.write(prompt)

# client = QdrantClient(path="vector_db/vector_index")
llm = Cohere(cohere_api_key=cohere_apikey)
chain = load_qa_chain(llm, chain_type="stuff")
# chain = RetrievalQAChain(
#     combine_documents_chain=loadQARefineChain(llm),
#     retriever=qdrant.as_retriever(),
# )
query="What types of ETA prediction models does uber use?"
query_vector = embeddings([query]).numpy().tolist()[0]
print(f"Query vector = {query_vector}")
similar_vectors = client.search(
    collection_name="my_collection",
    query_vector=query_vector,
    with_vectors=True,
    with_payload=True,
    limit=5  # Return 5 closest points
)
# similar_vectors = qdrant.search(query=query)
res = chain.run( input_documents=similar_vectors, question=query)
st.write(res)

# https://python.langchain.com/en/latest/modules/indexes/getting_started.html