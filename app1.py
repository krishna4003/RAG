import streamlit as st
from PyPDF2 import PdfReader
from pdf2image import convert_from_bytes
import pytesseract
from PIL import Image
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
import os
import shutil

# -----------------------------
# Dynamic Tesseract OCR Setup
# -----------------------------
tesseract_path = shutil.which("tesseract")
if tesseract_path is None:
    raise Exception(
        "Tesseract OCR not found. Install it via Conda (conda install -c conda-forge tesseract) "
        "or Homebrew/apt depending on your OS."
    )

pytesseract.pytesseract.tesseract_cmd = tesseract_path
print("Using Tesseract at:", tesseract_path)

# -----------------------------
# Load environment variables
# -----------------------------
load_dotenv()
GROK_API_KEY = os.getenv("GROK_API_KEY")  # Your Grok API Key

# -----------------------------
# Helper Functions
# -----------------------------
def get_pdf_text(pdf_docs):
    """Extract text from PDFs (normal + scanned)"""
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
            else:
                pdf.seek(0)
                images = convert_from_bytes(pdf.read())
                for img in images:
                    text += pytesseract.image_to_string(img)
    return text

def get_image_text(image_docs):
    """Extract text from images"""
    text = ""
    for img_file in image_docs:
        image = Image.open(img_file)
        text += pytesseract.image_to_string(image)
    return text

def get_text_chunks(text):
    """Split text into chunks for embeddings"""
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    return splitter.split_text(text)

def build_vector_store(text_chunks):
    """Create FAISS vector store"""
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    return vector_store

def get_qa_chain():
    """Build QA chain using Grok LLM"""
    prompt_template = """
Answer the question as accurately as possible using the provided context.
If the answer is not in the context, say "answer is not available in the context".

Context:\n{context}

Question:\n{question}

Answer:
"""
    grok_model = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(grok_model, chain_type="stuff", prompt=prompt)

def answer_question(user_question):
    """Retrieve docs from vector store and answer question"""
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    if not os.path.exists("faiss_index"):
        st.warning("Please upload and process files first!")
        return
    vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = vector_store.similarity_search(user_question)
    qa_chain = get_qa_chain()
    response = qa_chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    st.write("💁 Reply:", response["output_text"])

# -----------------------------
# Streamlit App
# -----------------------------
def main():
    st.set_page_config(page_title="Chat with PDFs & Images (RAG)")
    st.title("📄 Chat with PDFs, Scanned PDFs & Images (Hugging Face + Grok LLM)")

    # Sidebar - upload files
    with st.sidebar:
        st.header("Upload Files")
        pdf_docs = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)
        image_docs = st.file_uploader("Upload Images", type=["png","jpg","jpeg"], accept_multiple_files=True)

        if st.button("Process Files"):
            if pdf_docs or image_docs:
                with st.spinner("Processing files..."):
                    raw_text = ""
                    if pdf_docs:
                        raw_text += get_pdf_text(pdf_docs)
                    if image_docs:
                        raw_text += get_image_text(image_docs)

                    if raw_text.strip() == "":
                        st.warning("No text could be extracted from the uploaded files.")
                    else:
                        chunks = get_text_chunks(raw_text)
                        build_vector_store(chunks)
                        st.success("Files processed and vector store created!")
            else:
                st.warning("Please upload at least one PDF or image.")

    # Main - ask questions
    user_question = st.text_input("Ask a question from the uploaded files")
    if user_question:
        answer_question(user_question)

if __name__ == "__main__":
    main()
