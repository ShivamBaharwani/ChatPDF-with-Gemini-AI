import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY") or st.secrets.get("GOOGLE_API_KEY")

# Check API Key presence
if not api_key:
    st.error("GOOGLE_API_KEY not found! Please set it in .env (local) or Streamlit Secrets (cloud).")
    st.stop()

genai.configure(api_key=api_key)

# Get PDF text content
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""  # Safely handle empty pages
    return text

# Split text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

# Create and save vector store using FAISS
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")  # Saves index locally

# Build the conversational chain
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context. Make sure to provide all the details. 
    If the answer is not in the provided context, just say "answer is not available in the context", don't make up an answer.

    Context:\n{context}\n
    Question:\n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(
        model="models/gemini-1.5-pro-latest",
        temperature=0.3
    )

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain

# Process user input and run the conversational chain
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # Load the FAISS vector store
    vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

    docs = vector_store.similarity_search(user_question)

    chain = get_conversational_chain()

    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)

    st.write("Reply:", response["output_text"])

# Streamlit main app
def main():
    st.set_page_config("Chat PDF")
    st.header("Chat with PDF using Gemini 🤖📄")

    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            if pdf_docs:
                with st.spinner("Processing..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("Done! Your PDF is ready to chat. 🎉")
            else:
                st.warning("Please upload at least one PDF file.")

if __name__ == "__main__":
    main()
