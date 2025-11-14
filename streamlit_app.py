import streamlit as st
from transformers import pipeline
import PyPDF2

# --- 1. Model Caching + title---
st.title("PDF/Text Summarizer")
st.write("Upload a PDF or paste text to get a concise summary.")
st.sidebar.info("Developed by Hemant Tyagi")
st.sidebar.info("GitHub:\nhttps://github.com/Hemant-tyagi9/Ternary-CPU",
                "This app is build to summarize the lengthy chapters to small short notes which saves your time.")

@st.cache_resource
def get_summarizer():
    return pipeline("summarization", model="facebook/bart-large-cnn")

summarizer = get_summarizer()

# --- 2. Initialize Session State for Mode Management ---
if 'mode' not in st.session_state:
    st.session_state.mode = 'pdf'

# --- 3. UI Elements: File Uploader and Text Button ---
col1, col2 = st.columns(2)

with col1:
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf", 
                                     disabled=(st.session_state.mode == 'text'))
    if uploaded_file and st.session_state.mode != 'pdf':
        st.session_state.mode = 'pdf'
        st.rerun()

with col2:
    if st.button("Click to paste text here"):
        # Toggle between 'pdf' and 'text' mode
        st.session_state.mode = 'text' if st.session_state.mode == 'pdf' else 'pdf'
        st.rerun()

# --- 4. Content Input Logic based on Mode ---
ptxt = ""
input_text = ""

if st.session_state.mode == 'pdf' and uploaded_file:
    # Read and process PDF
    try:
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                ptxt += page_text
        input_text = ptxt # Set input_text for summarization
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        st.stop() # Stop execution if PDF reading fails

elif st.session_state.mode == 'text':
    # This text area is ALWAYS rendered in 'text' mode, preserving its value
    input_text = st.text_area("Paste any article, essay, or research paper", height=300)

# --- 5. Summary Parameters ---
max_len = st.slider("Summary length", 50, 200, 130)

# --- 6. Clear Button ---
if st.button("Clear"):
    st.session_state.mode = 'pdf' # Reset mode
    st.experimental_rerun()

# --- 7. Summarize Button and Execution ---
if st.button("Summarize Now"):
    
    if not input_text:
        st.warning("Please provide text either by uploading a PDF or pasting content.")
        
    elif len(input_text) < 100:
        st.error("Text is too short for summarization (min 100 characters recommended).")
        
    else:
        with st.spinner("Summarizing..."):
            try:
                summary = summarizer(
                    input_text, 
                    max_length=max_len, 
                    min_length=30, 
                    do_sample=False
                )[0]['summary_text']
                
                # Display result
                st.success("Summary:")
                st.write(summary)
            except Exception as e:
                st.error(f"An error occurred during summarization: {e}")
