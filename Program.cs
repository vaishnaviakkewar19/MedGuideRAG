// MedGuideRAG - a two-stage RAG demo: vector search (stage 1) + reranking (stage 2).
// Educational demo only. Not medical advice.
//
// Run:  dotnet run                          (keyword reranker, the default)
//       dotnet run -- --reranker llm        (LLM reranker using llama3.2)
//       dotnet run -- --reranker llm --model qwen2.5:3b   (any Ollama chat model)

using MedGuideRAG;

const string OllamaUrl = "http://localhost:11434";
const int RetrieveK = 10;   // stage 1: cast a wide net
const int FinalN = 3;       // stage 2: keep only the best few
const int ShowBefore = RetrieveK; // print ALL stage-1 results, so you can see every candidate the reranker gets

// ---------- Read command-line arguments ----------
string rerankerName = GetArg(args, "--reranker") ?? "keyword";
string chatModel = GetArg(args, "--model") ?? "llama3.2";

// One HttpClient for the whole app (creating many can exhaust sockets).
// Long timeout because the first call loads the model into memory.
var http = new HttpClient { BaseAddress = new Uri(OllamaUrl), Timeout = TimeSpan.FromMinutes(5) };

IReranker reranker = rerankerName.ToLowerInvariant() switch
{
    "keyword" => new KeywordReranker(),
    "llm"     => new LlmReranker(http, chatModel),
    _ => throw new ArgumentException($"Unknown reranker '{rerankerName}'. Use 'keyword' or 'llm'."),
};

WriteLine(ConsoleColor.Cyan, "=== MedGuideRAG: vector search + reranking over GLP-1 medication guides ===");
WriteLine(ConsoleColor.DarkYellow, "Educational demo only. Not medical advice.");
WriteLine(ConsoleColor.Gray, $"Reranker: {reranker.Name}\n");

// ---------- STEP 2: chunk ----------
// docs/ is copied next to the .exe by the .csproj, so this works from any folder.
var docsFolder = Path.Combine(AppContext.BaseDirectory, "docs");
var chunks = Chunker.LoadAndChunk(docsFolder);
WriteLine(ConsoleColor.Gray, $"Created {chunks.Count} chunks from {chunks.Select(c => c.Drug).Distinct().Count()} documents.");

// ---------- STEP 3: embed (once, kept in memory) ----------
var embedder = new Embedder(http);
try
{
    Console.Write("Embedding chunks with nomic-embed-text... ");
    await embedder.EmbedChunksAsync(chunks);
    WriteLine(ConsoleColor.Gray, $"done ({chunks[0].Embedding.Length} numbers per vector).");
}
catch (Exception ex) when (ex is HttpRequestException or TaskCanceledException)
{
    // The most common beginner problem: Ollama isn't running or the model isn't pulled.
    WriteLine(ConsoleColor.Red, $"\nCould not reach Ollama at {OllamaUrl}: {ex.Message}");
    WriteLine(ConsoleColor.Red, "Start Ollama, then run:  ollama pull nomic-embed-text  (and  ollama pull llama3.2  for --reranker llm)");
    return 1;
}

// ---------- STEP 7a: demo questions ----------
string[] demoQuestions =
{
    "I missed my weight-loss injection 3 days ago, what should I do?",
    "Can I take my diabetes tablet with breakfast?",
    "I inject my weight-loss medicine every day. What if I miss a dose?",
    "Does tirzepatide affect birth control pills?",
    "What is the starting dose of Mounjaro?",
    "How long can the Wegovy pen stay out of the fridge?",
};

foreach (var question in demoQuestions)
    await AskAsync(question);

// ---------- STEP 7b: your own questions ----------
while (true)
{
    WriteLine(ConsoleColor.Cyan, "\nType your own question (or press Enter on an empty line to exit):");
    Console.Write("> ");
    // Trim spaces and an invisible "byte order mark" (﻿) that some
    // terminals (e.g. PowerShell pipes) add, so an "empty" line really is empty.
    var input = Console.ReadLine()?.Trim().Trim('﻿').Trim();
    if (string.IsNullOrEmpty(input)) break; // null = input stream closed
    await AskAsync(input);
}

WriteLine(ConsoleColor.Gray, "Goodbye!");
return 0;

// =====================================================================
// One full question: retrieve -> rerank -> print before vs after.
// =====================================================================
async Task AskAsync(string question)
{
    WriteLine(ConsoleColor.White, "\n" + new string('=', 80));
    WriteLine(ConsoleColor.Yellow, $"QUESTION: {question}");
    WriteLine(ConsoleColor.White, new string('=', 80));

    // STEP 4: stage 1 - vector search for the top 10.
    var questionVector = await embedder.EmbedAsync(question);
    var before = Retriever.TopK(questionVector, chunks, RetrieveK);

    WriteLine(ConsoleColor.Magenta, $"\nBEFORE reranking (vector search, top {ShowBefore} candidates sent to the reranker):");
    for (int i = 0; i < Math.Min(ShowBefore, before.Count); i++)
    {
        var r = before[i];
        Console.WriteLine($"  #{i + 1,-2} score {r.Score:F3}  {r.Chunk.Drug,-15} | {r.Chunk.Section}");
    }

    // STEP 5: stage 2 - rerank the 10 candidates, keep the top 3.
    List<ScoredChunk> after;
    try
    {
        after = await reranker.RerankAsync(question, before, FinalN);
    }
    catch (Exception ex)
    {
        WriteLine(ConsoleColor.Red, $"Reranking failed: {ex.Message}");
        return;
    }

    WriteLine(ConsoleColor.Green, $"\nAFTER reranking ({reranker.Name}, top {FinalN}):");
    for (int i = 0; i < after.Count; i++)
    {
        var r = after[i];
        // Where was this chunk in the stage-1 list? (+1 because ranks start at 1)
        int oldRank = before.FindIndex(b => ReferenceEquals(b.Chunk, r.Chunk)) + 1;
        int newRank = i + 1;

        Console.Write($"  #{newRank,-2} score {r.Score,6:F3}  {r.Chunk.Drug,-15} | {r.Chunk.Section,-36} ");
        // Green = moved up, red = moved down, gray = stayed. This is the whole
        // point of the demo: watch the right chunk climb.
        var moveColor = oldRank > newRank ? ConsoleColor.Green : oldRank < newRank ? ConsoleColor.Red : ConsoleColor.DarkGray;
        Write(moveColor, $"#{oldRank} -> #{newRank}");
        WriteLine(ConsoleColor.DarkGray, string.IsNullOrEmpty(r.Reason) ? "" : $"   ({r.Reason})");
    }

    // The top 3 reranked chunks are the "context" for the LLM. In the next
    // RAG step (answer generation) this exact block would go into the prompt,
    // e.g. "Answer the question using ONLY this context: ...". Printing it now
    // shows you precisely what the LLM would be allowed to see.
    WriteLine(ConsoleColor.Cyan, $"\nCONTEXT PASSED TO THE LLM (top {after.Count} chunks after reranking):");
    Console.WriteLine(BuildContext(after));
}

// Formats the final chunks as a numbered context block. The [1], [2], [3]
// labels let a future answer step cite its sources ("see [1]").
static string BuildContext(List<ScoredChunk> finalChunks)
{
    var sb = new System.Text.StringBuilder();
    for (int i = 0; i < finalChunks.Count; i++)
    {
        var c = finalChunks[i].Chunk;
        sb.AppendLine($"  [{i + 1}] {c.Drug} ({c.Ingredient}) - {c.Section}");
        foreach (var line in c.Text.Split('\n'))
            sb.AppendLine("      " + line);
        sb.AppendLine();
    }
    return sb.ToString();
}

// ---------- tiny helpers ----------
static string? GetArg(string[] args, string name)
{
    int i = Array.FindIndex(args, a => a.Equals(name, StringComparison.OrdinalIgnoreCase));
    return i >= 0 && i + 1 < args.Length ? args[i + 1] : null;
}

static void Write(ConsoleColor color, string text)
{
    var old = Console.ForegroundColor;
    Console.ForegroundColor = color;
    Console.Write(text);
    Console.ForegroundColor = old;
}

static void WriteLine(ConsoleColor color, string text) => Write(color, text + Environment.NewLine);
