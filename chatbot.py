import os
import tempfile
import chromadb
import ollama
import streamlit as st
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import CrossEncoder
from streamlit.runtime.uploaded_file_manager import UploadedFile

# --- STEP 3: SYSTEM PROMPT ---
system_prompt = """
You are a professional Academic Advisor. Your goal is to recommend college courses based solely on the RIASEC context provided.
1. Identify RIASEC types/scores from the question.
2. List "Good college majors" and "Related Pathways" from the text.
3. Provide a brief justification based on the text.
Base your response ONLY on the provided context.
"""

# --- CUSTOM EMBEDDING FUNCTION (Fixes JSONDecodeError) ---
class MyOllamaEmbedder:
    def __call__(self, input):
        # Direct call to ollama library to bypass ChromaDB wrapper bugs
        embeddings = []
        for text in input:
            res = ollama.embeddings(model="mxbai-embed-large", prompt=text)
            embeddings.append(res["embedding"])
        return embeddings

# --- STEP 2: DATASET INGESTION ---
def process_document(uploaded_file: UploadedFile) -> list[Document]:
    temp_file = tempfile.NamedTemporaryFile("wb", suffix=".pdf", delete=False)
    temp_file.write(uploaded_file.read())
    temp_file_path = temp_file.name
    temp_file.close() 

    try:
        loader = PyMuPDFLoader(temp_file_path)
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        return text_splitter.split_documents(docs)
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

def get_vector_collection() -> chromadb.Collection:
    chroma_client = chromadb.PersistentClient(path="./riasec_vector_db")
    # Using our custom embedder here
    return chroma_client.get_or_create_collection(
        name="riasec_recommendations",
        embedding_function=MyOllamaEmbedder(), 
        metadata={"hnsw:space": "cosine"},
    )

def add_to_vector_collection(all_splits: list[Document], file_name: str):
    collection = get_vector_collection()
    documents, ids = [], []
    for idx, split in enumerate(all_splits):
        documents.append(split.page_content)
        ids.append(f"{file_name}_{idx}")

    collection.upsert(documents=documents, ids=ids)
    st.success("✅ RIASEC Dataset ingested successfully!")

# --- STEP 3 & 4: RETRIEVAL AND CHAT ---
def query_collection(prompt: str):
    collection = get_vector_collection()
    return collection.query(query_texts=[prompt], n_results=5)

def re_rank_cross_encoders(prompt: str, documents: list[str]) -> str:
    relevant_text = ""
    encoder_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    ranks = encoder_model.rank(prompt, documents, top_k=3)
    for rank in ranks:
        relevant_text += documents[rank["corpus_id"]] + "\n\n"
    return relevant_text

def call_llm(context: str, prompt: str):
    response = ollama.chat(
        model="llama3.2:3b",
        stream=True,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context: {context}, Question: {prompt}"},
        ],
    )
    for chunk in response:
        if not chunk["done"]:
            yield chunk["message"]["content"]

# --- STREAMLIT UI ---
st.set_page_config(page_title="RIASEC Course Advisor")
with st.sidebar:
    st.header("Setup")
    uploaded_file = st.file_uploader("Upload RIASEC PDF", type=["pdf"])
    if st.button("⚡️ Process Dataset") and uploaded_file:
        with st.spinner("Processing..."):
            chunks = process_document(uploaded_file)
            add_to_vector_collection(chunks, uploaded_file.name)

st.header("🎓 RIASEC Course Recommender")
if prompt := st.chat_input("Enter your scores..."):
    st.chat_message("user").write(prompt)
    with st.chat_message("assistant"):
        results = query_collection(prompt)
        if results and results["documents"]:
            context = re_rank_cross_encoders(prompt, results["documents"][0])
            st.write_stream(call_llm(context, prompt))
