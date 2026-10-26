import streamlit as st
from rag_pipeline import load_pipeline, ask

@st.cache_resource
def get_pipeline():
    return load_pipeline()

st.title("ui.pyPA DOC Family & Friends Chatbot")
st.write("Ask questions about visiting, mail, phone calls, and more.")

vectordb, tokenizer, model = get_pipeline()

question = st.text_input("Your question:")

if st.button("Ask") and question:
    with st.spinner("Finding answer..."):
        answer, docs = ask(question, vectordb, tokenizer, model)
    st.write(f"**Answer:** {answer}")

    with st.expander("Sources used"):
        for doc in docs:
            st.write(f"- {doc.metadata.get('source', 'unknown')}")


