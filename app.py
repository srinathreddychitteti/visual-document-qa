import streamlit as st
import tempfile
import asyncio
from logic import get_answer_from_pdf

st.set_page_config(page_title="Visual Document Q&A", layout="centered")

st.title(" Visual Document Q&A")
st.markdown("Upload a PDF and ask any question about its text, charts, or tables.")

uploaded_file = st.file_uploader("Upload your PDF", type=["pdf"])

question = st.text_input("Enter your question:")

if st.button("Ask"):
    if not uploaded_file:
        st.warning("Please upload a PDF file first.")
    elif not question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Analyzing document... please wait "):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name
            try:
                answer = asyncio.run(asyncio.to_thread(get_answer_from_pdf, tmp_path, question))
            except Exception as e:
                answer = f"Error processing document: {e}"

            st.subheader("Answer:")
            st.write(answer)
