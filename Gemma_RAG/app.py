import os 
import streamlit as st 
from langchain_groq import ChatGroq
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain.chains.retrieval import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFDirectoryLoader


from dotenv import load_dotenv
load_dotenv()

## Load the GROQ_API
groq_api_key = os.getenv("GROQ_API_KEY")

llm = ChatGroq(api_key=groq_api_key,
               model="Gemma-7b-It")

prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the context only.
    Please provide the most accurate response based on the question
    <context>
    {context}
    <context>
    Question: {input}

    """
)

def create_vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings = OllamaEmbeddings()
        st.session_state.loader = PyPDFDirectoryLoader("research_papers") # Data ingestion
        st.session_state.documents = st.session_state.loader.load() # Doc loading
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,
                                                                        chunk_overlap=20)
        
        st.session_state.final_docs = st.session_state.text_splitter.split_documents(st.session_state.documents[:100])
        st.session_state.vectors = Chroma.from_documents(st.session_state.final_docs,
                                                         st.session_state.embeddings)


user_prompt = st.text_input("Enter your query")
if st.button("Document Embedding"):
    create_vector_embedding()
    st.write("Vector DB is Ready")


import time

if user_prompt:
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    start = time.process_time()
    response = retrieval_chain.invoke({"input": user_prompt})
    print(f"Response time: {time.process_time()- start}")

    st.write(response["answer"])


    ## With a streamlit expander
    with st.expander("Document similarity search"):
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("----------------------------")