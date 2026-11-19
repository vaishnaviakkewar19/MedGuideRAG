# MedGuideRAG
Beginner-friendly two-stage RAG demo in C# (.NET 8). Chunks 8 GLP-1 medication guides, embeds them with Ollama (nomic-embed-text), retrieves the top 10 by cosine similarity, then reranks to the top 3 with an explainable keyword reranker or an LLM judge (llama3.2). Prints before/after rankings in the console. No NuGet packages. Not medical advice.
