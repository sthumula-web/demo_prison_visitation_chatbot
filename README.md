# PA DOC Visitation Chatbot

A locally run RAG chatbot that answers questions from families of incarcerated individuals about visitation, based on official PA DOC guides.

Stack: Python, LangChain, Hugging Face Transformers, Phi-4-mini-instruct, Chroma, Streamlit, uv

Run it: uv run streamlit run ui.py

Get the model: hf download microsoft/Phi-4-mini-instruct --local-dir ./phi-4-mini-instruct

Note: Model weights, source PDFs, and the vector database are not included in this repo.

## About this project
This is a personal demo built entirely from publicly available PA DOC
visitor guides and web pages. It runs fully loncally and contains fully publically avalaible data 

I built this prototype to demonstrate how a retrieval-augmented chatbot
for visitation questions works. I later worked with the PA DOC team on
the production version of this chatbot. That version is internal and is
not part of this repository.

This demo is not affiliated with or endorsed by the Pennsylvania
Department of Corrections. Answers may be incomplete or out of date.
Always confirm with the official PA DOC website.