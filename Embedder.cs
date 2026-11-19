using System.Net.Http.Json;
using System.Text.Json.Serialization;

namespace MedGuideRAG;

/// <summary>
/// Turns text into vectors (lists of numbers) using a local Ollama server.
/// Texts with similar MEANING get vectors that point in similar directions,
/// which is what lets us search by meaning instead of exact words.
/// </summary>
public class Embedder
{
    private readonly HttpClient _http;
    private readonly string _model;

    public Embedder(HttpClient http, string model = "nomic-embed-text")
    {
        _http = http;
        _model = model;
    }

    // These small classes match the JSON shape of Ollama's /api/embed endpoint.
    private record EmbedRequest(
        [property: JsonPropertyName("model")] string Model,
        [property: JsonPropertyName("input")] List<string> Input);

    private record EmbedResponse(
        [property: JsonPropertyName("embeddings")] List<float[]> Embeddings);

    /// <summary>Embeds many texts. We send them in batches because one HTTP
    /// call per chunk is slow, but one giant call can time out.</summary>
    public async Task<List<float[]>> EmbedManyAsync(List<string> texts, int batchSize = 16)
    {
        var results = new List<float[]>();
        for (int i = 0; i < texts.Count; i += batchSize)
        {
            var batch = texts.Skip(i).Take(batchSize).ToList();
            var response = await _http.PostAsJsonAsync("/api/embed", new EmbedRequest(_model, batch));
            response.EnsureSuccessStatusCode(); // throws if, e.g., the model isn't pulled
            var body = await response.Content.ReadFromJsonAsync<EmbedResponse>()
                       ?? throw new Exception("Ollama returned an empty embedding response.");
            results.AddRange(body.Embeddings);
        }
        return results;
    }

    public async Task<float[]> EmbedAsync(string text) => (await EmbedManyAsync(new List<string> { text }))[0];

    /// <summary>Embeds every chunk once at startup and stores the vector on it (in memory).</summary>
    public async Task EmbedChunksAsync(List<Chunk> chunks)
    {
        var vectors = await EmbedManyAsync(chunks.Select(c => c.EmbeddingInput).ToList());
        for (int i = 0; i < chunks.Count; i++) chunks[i].Embedding = vectors[i];
    }

    /// <summary>
    /// Cosine similarity = how much two vectors point in the same direction.
    /// 1.0 = same meaning, 0 = unrelated. We use cosine (not plain distance)
    /// because it ignores vector length and only compares direction.
    /// </summary>
    public static double CosineSimilarity(float[] a, float[] b)
    {
        double dot = 0, normA = 0, normB = 0;
        for (int i = 0; i < a.Length; i++)
        {
            dot   += a[i] * b[i];
            normA += a[i] * a[i];
            normB += b[i] * b[i];
        }
        if (normA == 0 || normB == 0) return 0; // avoid divide-by-zero
        return dot / (Math.Sqrt(normA) * Math.Sqrt(normB));
    }
}

/// <summary>
/// A chunk plus a score. Used for both search results and reranked results.
/// Reason is a short "why" from the reranker, so you can see its thinking.
/// </summary>
public record ScoredChunk(Chunk Chunk, double Score, string Reason = "");

public static class Retriever
{
    /// <summary>
    /// STAGE 1 (retrieve): compare the question vector with every chunk vector
    /// and keep the best K. This is fast but "fuzzy": it finds chunks about the
    /// right TOPIC, but often can't tell Wegovy from Ozempic, or weekly from daily.
    /// That's why we grab 10 (a wide net) and let the reranker pick the best 3.
    /// </summary>
    public static List<ScoredChunk> TopK(float[] questionVector, List<Chunk> chunks, int k = 10) =>
        chunks
            .Select(c => new ScoredChunk(c, Embedder.CosineSimilarity(questionVector, c.Embedding)))
            .OrderByDescending(s => s.Score)
            .Take(k)
            .ToList();
}
