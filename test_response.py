from rag_pipeline import load_pipeline, ask

question = "how can i visit a loved one in jail?"

vectordb, tokenizer, model = load_pipeline()
answer, docs = ask(question, vectordb, tokenizer, model)
print(f"\nAnswer: {answer}")

print("\nSources used:")
for doc in docs:
    print(f"  - {doc.metadata.get('source', 'unknown')}")