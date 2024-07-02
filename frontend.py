# app.py
import streamlit as st
import os
import tempfile
from processing import process_pdf
from rag_chain import crag_pipeline

def main():
    st.title("Summarizer")
    st.write("Upload a PDF file to summarize it")

    uploaded_files = st.file_uploader("Choose a PDF files", type="pdf", accept_multiple_files=True)

    if uploaded_files:
        with tempfile.TemporaryDirectory() as temp_dir:
            all_text=""
            for uploaded_file in uploaded_files:

                temp_pdf_path = os.path.join(temp_dir, uploaded_file.name)
                with open(temp_pdf_path, "wb") as f:
                    f.write(uploaded_file.getvalue())

                with st.spinner("Processing PDF..."):
                    categorized_elements = process_pdf(temp_pdf_path, temp_dir)

                pdf_text = "\n\n".join([
                    "\n".join(categorized_elements.get(category, []))
                    for category in ["Title", "NarrativeText", "Text", "ListItem"]
                ])
                all_text += f"\n\n--- Document: {uploaded_file.name} ---\n\n" + pdf_text

            st.subheader("Ask a question about the document")
            question = st.text_input("Enter your question:")
            
            if st.button("Generate Answer"):
                with st.spinner("Generating answer with Corrective RAG..."):
                    answer = crag_pipeline(question, all_text)
                
                st.subheader("Answer")
                st.write(answer)

            # st.subheader("Processed PDF Elements")
            # for category, elements in categorized_elements.items():
            #     if elements:
            #         st.write(f"**{category}:**")
            #         for element in elements[:3]:
            #             st.write(element)
            #         if len(elements) > 3:
            #             st.write(f"... and {len(elements) - 3} more")

if __name__ == "__main__":
    main()