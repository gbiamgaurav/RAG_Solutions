import os 
import streamlit as st 
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains.retrieval import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
import time
from dotenv import load_dotenv
load_dotenv()

os.environ["NVIDIA_API_KEY"] = os.getenv("NVIDIA_API_KEY")

llm = ChatNVIDIA(model="meta/llam3-70b-instruct") # NVIDIA NIM Inferencing


def vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings = NVIDIAEmbeddings()
        st.session_state.loader = PyPDFDirectoryLoader("./us_cencus")
        st.session_state.docs = st.session_state.loader.load()
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=20)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:50])
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embedding)



st.title("Nvidia NIM Demo")

prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question
    <context>
    {context}
    <context>
    Questions: {input}
    
    """
)

prompt1 = st.text_input("Enter your questions from Documents")

if st.button("Document Embedding"):
    vector_embedding()
    st.write("Vector Store DB is Ready")

if prompt1:
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retreiver()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    start = time.process_time()
    response = retrieval_chain.invoke({"input": prompt1})
    print("Response time: ", time.process_time() - start)
    st.write(response["answer"])

    # With a streamlit expander
    with st.expander("Document similarity seach"):
        # Find the relevant chunks
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("---------------------------")