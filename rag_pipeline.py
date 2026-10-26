import os
import sys
import torch

sys.path.append('../..')

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ["USER_AGENT"] = "PA-DOC-Chatbot/1.0"

def load_pipeline():
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    vectordb = Chroma(
        persist_directory='docs/chroma/',
        embedding_function=embedding
    )
    print(f"Loaded vector db with {vectordb._collection.count()} chunks")

    model_path = "/Users/sisirathumula/programs/CORRECT_demo/phi-4-mini-instruct"
    print("Loading Phi-4...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.float16,
        device_map="mps"
    )
    print("Phi-4 loaded.")

    return vectordb, tokenizer, model

def ask(question, vectordb, tokenizer, model, k=5, fetch_k=35):
    relevant_docs = vectordb.max_marginal_relevance_search(
        question, k=k, fetch_k=fetch_k
    )

    context = "\n\n".join([doc.page_content for doc in relevant_docs])

    system_prompt = """You are a helpful assistant for the Pennsylvania Department of Corrections.
You help family members and friends of inmates understand rules, visitation, mail, phone calls, and other DOC policies.
Answer questions using the provided context. Use reasoning and inference across the context to give complete, helpful answers.
If the topic is completely absent from the context, say "I don't have that information, please contact the facility directly."
Be thorough, clear, and compassionate — the people asking are families of incarcerated individuals."""

    user_message = f"""Context:
{context}

Question: {question}

Answer:"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]

    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=1024,
        temperature=0.3,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )
    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[-1]:],
        skip_special_tokens=True
    )
    return response, relevant_docs