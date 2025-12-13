import fitz  # PyMuPDF
import streamlit as st

def extract_text_from_pdf(pdf_file):
    """
    Extract text from uploaded PDF file.
    
    Args:
        pdf_file: Streamlit uploaded file object
        
    Returns:
        str: Extracted text from the PDF
    """
    try:
        # Save uploaded file temporarily
        temp_path = "uploads/temp.pdf"
        with open(temp_path, "wb") as f:
            f.write(pdf_file.getbuffer())

        # Extract text using PyMuPDF
        pages_text = []
        with fitz.open(temp_path) as doc:
            for page in doc:
                page_text = page.get_text()
                pages_text.append(page_text)

        # Combined text (useful for parsers that expect a single string)
        combined_text = "\n\n".join([p for p in pages_text if p and p.strip()])

        return combined_text, pages_text
    except Exception as e:
        st.error(f"Error extracting text from PDF: {str(e)}")
        return ""