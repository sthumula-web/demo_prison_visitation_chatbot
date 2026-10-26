from rag_pipeline import load_pipeline

question = "how to send an inmate money?"

vectordb, tokenizer, model = load_pipeline()

print("\n" + "="*60)
print("MMR SEARCH:")
mmr_results = vectordb.max_marginal_relevance_search(question, k=5, fetch_k=25)
for i, doc in enumerate(mmr_results):
    print(f"\nChunk {i+1} - Source: {doc.metadata.get('source', 'unknown')}")
    print(doc.page_content)