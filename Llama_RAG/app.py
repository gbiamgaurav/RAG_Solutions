import os 
import streamlit as st 
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain_chroma.vectorstores import Chroma
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.runnables.history import RunnableWithMessageHistory
from dotenv import load_dotenv
load_dotenv()

api_key = os.getenv("HF_TOKEN")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


## Streamlit app
st.title("Conversational RAG with PDF and Chat history")
st.write("Upload the PDF's & chat with them")


groq_api_key = st.text_input("Enter your GROQ API Key", type="password")

if groq_api_key:
    llm = ChatGroq(api_key=groq_api_key,
                   model="llama3-70b-8192")
    
    session_id = st.text_input("Session ID", value="default_session")

    if "store" not in st.session_state:
        st.session_state.store={}

    # Upload Docs
    uploaded_files = st.file_uploader("Upload your PDFs", 
                                      accept_multiple_files=True, 
                                      type="pdf")
    
    # Process uploaded docs
    if uploaded_files:
        documents = []
        for file in uploaded_files:
            temp_pdf = f"./temp.pdf"
            with open(temp_pdf, "wb") as file:
                file.write(uploaded_files.getvalue())
                file_name = uploaded_files.name

            loader = PyPDFLoader(temp_pdf)
            docs = loader.load()
            documents.extend(docs)

        # Split and create embeddings for the documents
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000,
                                                       chunk_overlap=200)
        splits = text_splitter.split_documents(documents)
        vectorstore = Chroma.from_documents(documents=splits,
                                            embedding=embeddings)
        retriever = vectorstore.as_retriever()


        contextualize_system_prompt = (
            "Given a chat history and the latest user question"
            "which might reference context in the chat history,"
            "formulate a standalone question which can be understood"
            "without the chat history. Do Not answer the question,"
            "just reformulate it if needed and otherwise return it as is."
        )

        contextualize_system_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )


        history_aware_retriever = create_history_aware_retriever(llm, 
                                                                retriever, 
                                                                contextualize_system_prompt)


        ## Answer question prompt
        system_prompt = (
            "You are an assistant for question answering tasks."
            "use the following pieces of retrieved context to answer"
            "the question. If you don't know the answer, say that you"
            "don't know. Use three sentences maximum and keep the"
            "answer concise"
            "\n\n"
            "{context}")


        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )


        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        def get_session_history(session:str)->BaseChatMessageHistory:
            if session_id not in st.session_state.store:
                st.session_state.store[session_id] = ChatMessageHistory()
            return st.session_state.store[session_id]
        

        conversation_rag_chain = RunnableWithMessageHistory(
            rag_chain, get_session_history,
            input_messages_key="input",
            history_messages_key="chathistory",
            output_messages_key="answer"
        )

        user_input = st.text_input("Your question")
        if user_input:
            session_history = get_session_history(session_id)
            response = conversation_rag_chain.invoke(
                {"input": user_input},
                config = {
                    "configurable": {"session_id": session_id}
                },
            )
            st.write(st.session_state.store)
            st.success("Assistant:", response["answer"])
            st.write("Chat History:", session_history.messages)

    else:
        st.warning("Please enter the GROQ API Key")































