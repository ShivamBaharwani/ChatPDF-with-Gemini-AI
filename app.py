import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import ConversationalRetrievalChain
from dotenv import load_dotenv

# Load environment variables from .env file (for local dev)
load_dotenv()

# Get the API Key from Render environment or secrets
api_key = os.getenv("GOOGLE_API_KEY") or st.secrets.get("GOOGLE_API_KEY")

# Check API Key presence
if not api_key:
    st.error("GOOGLE_API_KEY not found! Please set it in Render Environment Variables or Streamlit Secrets.")
    st.stop()

# Configure Google GenAI
genai.configure(api_key=api_key)

# ===================== PDF Processing =====================

def get_pdf_text(pdf_docs):
    """Extracts text from multiple PDF files"""
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text

def get_text_chunks(text):
    """Splits large text into manageable chunks"""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

# ===================== Vector Store =====================

def get_vector_store(text_chunks):
    """Creates and persists Chroma vector store from text chunks"""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    persist_directory = "db"

    # Create new Chroma DB (persistent)
    vector_store = Chroma.from_texts(
        texts=text_chunks,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    
    vector_store.persist()

# ===================== Conversational Chain =====================

def get_conversational_chain(vector_store):
    """Creates ConversationalRetrievalChain with Google Gemini"""
    model = ChatGoogleGenerativeAI(
        model="models/gemini-1.5-pro-latest",
        temperature=0.3
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=model,
        retriever=vector_store.as_retriever(search_kwargs={"k": 5}),
        return_source_documents=True
    )

    return chain

# ===================== User Input Handling =====================

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    persist_directory = "db"

    # Load existing Chroma vector store
    vector_store = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )

    # Get conversational chain with retriever
    chain = get_conversational_chain(vector_store)

    # Run the chain to get the response
    result = chain({
        "question": user_question,
        "chat_history": st.session_state.get('chat_history', [])
    })

    # Extract and display the answer
    st.write("Reply:", result['answer'])

    # Update chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    st.session_state.chat_history.append((user_question, result['answer']))

    # Optional: Show the source documents (for transparency/debugging)
    with st.expander("Source Documents"):
        for i, doc in enumerate(result["source_documents"]):
            st.markdown(f"**Document {i + 1}:** {doc.metadata.get('source', 'N/A')}")
            st.write(doc.page_content)

# ===================== Streamlit UI =====================

def main():
    st.set_page_config("Chat PDF with Gemini", page_icon="🤖📄")
    st.header("Chat with PDF using Gemini 🤖📄")

    user_question = st.text_input("Ask a question from the PDF files:")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("📂 Upload and Process Files")
        pdf_docs = st.file_uploader("Upload your PDF files:", accept_multiple_files=True)
        
        if st.button("Submit & Process"):
            if pdf_docs:
                with st.spinner("Processing PDFs..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("✅ Done! You can now chat with your PDFs.")
            else:
                st.warning("⚠️ Please upload at least one PDF file.")

if __name__ == "__main__":
    main()
