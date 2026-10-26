# PA DOC Visitation Chatbot

A locally run RAG chatbot that answers questions from families of incarcerated individuals about visitation, based on official PA DOC guides.

Stack: Python, LangChain, Hugging Face Transformers, Phi-4-mini-instruct, Chroma, Streamlit, uv

Run it: uv run streamlit run ui.py

Get the model: hf download microsoft/Phi-4-mini-instruct --local-dir ./phi-4-mini-instruct

Note: Model weights, source PDFs, and the vector database are not included in this repo.
