# MedGuideRAG

Two-stage RAG in C# over GLP-1 medication guides: vector search + reranking, with before/after results in the console.

> **Educational demo only. Not medical advice.** The guides in `docs/` are simplified summaries. Items marked `[VERIFY]` should be checked against the official FDA labels on [DailyMed](https://dailymed.nlm.nih.gov/).

## What is reranking?

Search happens in two steps, a bit like hiring:

1. **Vector search** is the quick resume screen. It turns the question and every document chunk into vectors (lists of numbers that capture meaning) and keeps the chunks whose meaning is closest. It's fast, but it's fuzzy: it finds the right *topic*, but it can't always find the right *answer*.
2. **Reranking** is the interview. It takes the shortlist and looks at each candidate more carefully, then re-orders them. That's slower, which is why it only runs on a short list.

## The two-stage flow

```
question ──► embed ──► cosine similarity over all 68 chunks ──► TOP 10   (stage 1: retrieve)
                                                                   │
                                                                   ▼
                                                  reranker scores all 10 ──► TOP 3   (stage 2: rerank)
```

For each question the app prints:
1. **BEFORE**: all 10 chunks from vector search (rank, score, drug, section).
2. **AFTER**: the top 3 after reranking, with how each one moved (for example `#6 -> #3`).
3. **CONTEXT PASSED TO THE LLM**: the full text of those 3 chunks, numbered [1] to [3]. This is exactly what a future answer-generation step would put in its prompt.

A full diagram is in [`diagrams/medguiderag-flow.excalidraw`](diagrams/medguiderag-flow.excalidraw). Open it at [excalidraw.com](https://excalidraw.com) with *Open*, or with the Excalidraw VS Code extension.

Two rerankers are included (see `Rerankers.cs`):

| Reranker | How it scores | Pros / cons |
|---|---|---|
| `keyword` (default) | Starts from the vector score and adds or subtracts points with simple, commented rules: drug name, ingredient, diabetes vs weight, tablet vs injection, daily vs weekly, and which section the question asks about (missed dose, storage, dosing...). | Instant, and it explains every decision. It only knows the rules you write. |
| `llm` | Asks a local chat model (default `llama3.2`) to score each chunk from 0 to 10 and reply with only the number. | No rules to write, but it's slower (10 calls per question). A small 3B model is a rough judge and often gives ties. Try a bigger model with `--model`. |

## Why these documents are hard for basic search

The 7 drug guides all use **the same 9 section headings**, and many drugs share an ingredient:

| Ingredient | Diabetes | Weight management |
|---|---|---|
| semaglutide | Ozempic (weekly injection), Rybelsus (**daily tablet**) | Wegovy (weekly injection) |
| tirzepatide | Mounjaro (weekly injection) | Zepbound (weekly injection) |
| liraglutide | Victoza (**daily** injection) | Saxenda (**daily** injection) |

So "If you miss a dose" exists 7 times with very similar wording, but the rules are different:
- **Same ingredient, different use:** Ozempic and Wegovy are both semaglutide, but one is for diabetes and the other for weight.
- **Weekly vs daily:** a missed weekly dose may still be taken days later, while a missed daily dose is usually skipped.
- **Injection vs tablet:** only Rybelsus has strict empty-stomach rules.

There is also a distractor, `glp1-overview.md`. It mentions every drug and every topic but has no specific answers, so vector search likes it and a good reranker should push it down.

## Setup

1. Install the [.NET 8 SDK](https://dotnet.microsoft.com/download/dotnet/8.0).
2. Install [Ollama](https://ollama.com/download) and make sure it's running (it listens on `http://localhost:11434`).
3. Pull the models:
   ```bash
   ollama pull nomic-embed-text   # embeddings (needed always)
   ollama pull llama3.2           # chat model (needed only for --reranker llm)
   ```

## Run

```bash
dotnet run                                   # keyword reranker (default)
dotnet run -- --reranker llm                 # LLM reranker with llama3.2
dotnet run -- --reranker llm --model qwen2.5:7b   # any other Ollama chat model
```

The app runs 6 demo questions first, then lets you type your own. Press Enter on an empty line to exit.

## Project layout

| File | What it does |
|---|---|
| `Program.cs` | Reads arguments, runs the demo questions and the input loop, and prints the colored before/after results |
| `Chunker.cs` | Splits each guide on `##` headings into chunks, each with its drug, ingredient and section |
| `Embedder.cs` | Calls Ollama `/api/embed` with plain `HttpClient`, plus cosine similarity and top-K retrieval |
| `Rerankers.cs` | The `IReranker` interface, `KeywordReranker`, and `LlmReranker` |
| `docs/` | 7 drug guides plus 1 distractor overview |

There are no NuGet packages, and everything is kept in memory.

## Disclaimer

Educational demo only. Not medical advice. Always follow your prescriber, your pharmacist, and the official FDA label.

## License

MIT. See [LICENSE](LICENSE).
