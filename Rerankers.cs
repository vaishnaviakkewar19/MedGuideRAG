using System.Globalization;
using System.Net.Http.Json;
using System.Text.Json.Serialization;
using System.Text.RegularExpressions;

namespace MedGuideRAG;

/// <summary>
/// STAGE 2 (rerank): take the ~10 candidates from vector search and re-order
/// them with a slower but smarter scoring method. Vector search is good at
/// "roughly the right topic"; a reranker is good at "exactly the right answer".
/// Both implementations share this interface so Program.cs can swap them with
/// a command-line flag.
/// </summary>
public interface IReranker
{
    string Name { get; }
    Task<List<ScoredChunk>> RerankAsync(string question, List<ScoredChunk> candidates, int topN = 3);
}

// =====================================================================
// a) KeywordReranker: simple rules you can read and reason about.
// =====================================================================
public class KeywordReranker : IReranker
{
    public string Name => "keyword";

    // Document-level facts about each drug. WHY do we need this? A chunk like
    // "Wegovy - If you miss a dose" never says the word "weight" in its text,
    // yet the user asking about a "weight-loss injection" needs exactly that
    // chunk. Knowing what each DRUG is for lets every one of its chunks benefit.
    private record DrugFacts(string Condition, string Form, string Frequency);

    private static readonly Dictionary<string, DrugFacts> Facts = new()
    {
        ["Ozempic"]  = new("diabetes", "injection", "weekly"),
        ["Wegovy"]   = new("weight",   "injection", "weekly"),
        ["Rybelsus"] = new("diabetes", "tablet",    "daily"),
        ["Mounjaro"] = new("diabetes", "injection", "weekly"),
        ["Zepbound"] = new("weight",   "injection", "weekly"),
        ["Victoza"]  = new("diabetes", "injection", "daily"),
        ["Saxenda"]  = new("weight",   "injection", "daily"),
        // The overview has no facts on purpose: it is a distractor that talks
        // about everything, so it should never get these boosts.
    };

    // Words that tell us WHICH SECTION the user wants. Order matters a little:
    // the first matching section wins, so "missed a dose" maps to the
    // missed-dose section before "dose" can map it to the dosing schedule.
    private static readonly (string Section, string[] Words)[] SectionIntents =
    {
        ("If you miss a dose",                    new[] { "miss", "missed", "forgot", "forget", "skipped" }),
        ("Storage",                               new[] { "storage", "store", "fridge", "refrigerator", "room temperature", "freeze", "travel" }),
        ("Interactions and special situations",   new[] { "birth control", "contracepti", "interact", "affect", "other medicine", "pregnan", "surgery", "alcohol" }),
        ("How to take it",                        new[] { "breakfast", "food", "meal", "eat", "water", "inject where", "how do i take", "how to take", "with or without" }),
        ("Common side effects",                   new[] { "side effect", "nausea", "vomit", "diarrhea", "constipation" }),
        ("Who should not take it",                new[] { "should not", "shouldn't", "allerg", "thyroid cancer", "safe for" }),
        ("Dosing schedule",                       new[] { "dose", "dosing", "starting", "mg", "increase", "how much" }),
        ("How it works",                          new[] { "how does it work", "how it works", "mechanism" }),
        ("What it's used for",                    new[] { "used for", "what is it for", "approved for" }),
    };

    // Common words that carry no meaning for matching ("the", "my"...).
    private static readonly HashSet<string> StopWords = new(StringComparer.OrdinalIgnoreCase)
    {
        "a","an","the","i","my","me","is","it","to","of","and","or","in","on","for","with","what",
        "should","do","does","can","if","how","be","at","this","that","am","are","was","you","your","ago","long"
    };

    public Task<List<ScoredChunk>> RerankAsync(string question, List<ScoredChunk> candidates, int topN = 3)
    {
        var q = question.ToLowerInvariant();

        // Work out what the QUESTION is asking for, once.
        string? wantCondition =
            ContainsAny(q, "weight", "obesity", "lose", "loss") ? "weight" :
            ContainsAny(q, "diabetes", "blood sugar", "a1c") ? "diabetes" : null;
        // Note: we deliberately don't treat "pill" as a tablet word, because
        // "birth control pills" would then wrongly point to Rybelsus.
        string? wantForm =
            ContainsAny(q, "tablet", "by mouth", "oral") ? "tablet" :
            ContainsAny(q, "inject", "injection", "pen", "shot") ? "injection" : null;
        string? wantFrequency =
            ContainsAny(q, "daily", "every day", "once a day", "each day") ? "daily" :
            ContainsAny(q, "weekly", "every week", "once a week") ? "weekly" : null;
        string? wantSection = SectionIntents.FirstOrDefault(s => ContainsAny(q, s.Words)).Section;

        var keywords = Regex.Matches(q, "[a-z0-9]+").Select(m => m.Value)
                            .Where(w => w.Length > 2 && !StopWords.Contains(w)).Distinct().ToList();

        var results = new List<ScoredChunk>();
        foreach (var candidate in candidates)
        {
            var c = candidate.Chunk;
            var reasons = new List<string>();

            // We START from the vector score so the reranker refines the search
            // result instead of throwing it away. The rules below nudge it.
            double score = candidate.Score;

            // Rule 1: the question names this exact drug -> big boost.
            // A brand name is the strongest possible signal of intent.
            if (q.Contains(c.Drug.ToLowerInvariant()))
            {
                score += 0.30; reasons.Add("drug name");
            }
            // Rule 1b: the question names a DIFFERENT drug -> penalty.
            // "Wegovy" in the question means Ozempic chunks are wrong, even
            // though they are about the same ingredient and look very similar.
            else if (Facts.Keys.Any(d => q.Contains(d.ToLowerInvariant())))
            {
                score -= 0.20; reasons.Add("other drug named");
            }

            // Rule 2: the question names the ingredient -> medium boost.
            // Weaker than a brand name because one ingredient = several brands.
            if (c.Ingredient.Split(", ").Any(ing => q.Contains(ing)) && Facts.ContainsKey(c.Drug))
            {
                score += 0.20; reasons.Add("ingredient");
            }

            if (Facts.TryGetValue(c.Drug, out var f))
            {
                // Rule 3: condition. Same ingredient is sold for diabetes AND for
                // weight (Ozempic vs Wegovy), so this splits look-alike chunks.
                if (wantCondition != null)
                {
                    if (f.Condition == wantCondition) { score += 0.15; reasons.Add(wantCondition); }
                    else                              { score -= 0.10; reasons.Add($"not {wantCondition}"); }
                }

                // Rule 4: form. "Tablet" means Rybelsus; an injection question
                // should not get the tablet rules (and vice versa).
                if (wantForm != null)
                {
                    if (f.Form == wantForm) { score += 0.15; reasons.Add(wantForm); }
                    else                    { score -= 0.10; reasons.Add($"not {wantForm}"); }
                }

                // Rule 5: frequency. Missed-dose rules for DAILY drugs ("skip it,
                // take tomorrow") are very different from WEEKLY drugs ("take
                // within 4-5 days"), so this matters a lot for safety.
                if (wantFrequency != null)
                {
                    if (f.Frequency == wantFrequency) { score += 0.15; reasons.Add(wantFrequency); }
                    else                              { score -= 0.10; reasons.Add($"not {wantFrequency}"); }
                }
            }

            // Rule 6: section intent. "missed" -> missed-dose section,
            // "fridge" -> storage, etc. Vector search often returns the right
            // drug but the wrong section; this fixes that.
            if (wantSection != null && c.Section == wantSection)
            {
                score += 0.25; reasons.Add($"section: {wantSection}");
            }

            // Rule 7: plain word overlap. Small bonus per question word found
            // in the chunk, capped so it can't outweigh the rules above.
            var text = c.EmbeddingInput.ToLowerInvariant();
            int hits = keywords.Count(k => text.Contains(k));
            if (hits > 0)
            {
                score += Math.Min(0.10, hits * 0.02); reasons.Add($"{hits} word(s)");
            }

            results.Add(candidate with { Score = score, Reason = string.Join(", ", reasons) });
        }

        return Task.FromResult(results.OrderByDescending(r => r.Score).Take(topN).ToList());
    }

    // Matches words at the START of a word (\b), so "pen" matches "pens" but
    // not "happen", and "miss" matches "missed" but not "dismiss".
    private static bool ContainsAny(string text, params string[] words) =>
        words.Any(w => Regex.IsMatch(text, @"\b" + Regex.Escape(w)));
}

// =====================================================================
// b) LlmReranker: ask a small chat model "how well does this answer it?"
// =====================================================================
public class LlmReranker : IReranker
{
    private readonly HttpClient _http;
    private readonly string _model;

    public LlmReranker(HttpClient http, string model = "llama3.2")
    {
        _http = http;
        _model = model;
    }

    public string Name => $"llm ({_model})";

    // JSON shapes for Ollama's /api/chat endpoint.
    private record ChatMessage(
        [property: JsonPropertyName("role")] string Role,
        [property: JsonPropertyName("content")] string Content);

    private record ChatRequest(
        [property: JsonPropertyName("model")] string Model,
        [property: JsonPropertyName("messages")] List<ChatMessage> Messages,
        [property: JsonPropertyName("stream")] bool Stream,
        [property: JsonPropertyName("options")] Dictionary<string, object> Options);

    private record ChatResponse(
        [property: JsonPropertyName("message")] ChatMessage? Message);

    // A clear rubric matters a lot for SMALL models: without one, they give
    // almost every passage a 7 or 8 and the reranking does nothing.
    private const string SystemPrompt =
        "You are a strict relevance judge for medication guide passages.\n" +
        "Rate from 0 to 10 whether the passage contains the ANSWER to the question.\n" +
        "Rubric:\n" +
        "10 = the passage states the exact answer, for the medicine the question is about.\n" +
        "6 = right medicine, but the answer is only partly there.\n" +
        "3 = same topic, but no answer, OR the passage is about a medicine that does not match the question " +
        "(wrong brand, diabetes vs weight loss, tablet vs injection, daily vs weekly).\n" +
        "0 = unrelated.\n" +
        "Reply with ONLY the number. No words.";

    public async Task<List<ScoredChunk>> RerankAsync(string question, List<ScoredChunk> candidates, int topN = 3)
    {
        var results = new List<ScoredChunk>();

        // One call per chunk. This is slow (that's the price of reranking!),
        // which is exactly why we only rerank 10 chunks and not all 68.
        foreach (var candidate in candidates)
        {
            // We include the drug, ingredient, and section in the prompt because
            // the model can't judge "is this the right medicine?" without them.
            var c = candidate.Chunk;
            var userPrompt = $"Question: {question}\n\nPassage from the {c.Drug} ({c.Ingredient}) guide, section \"{c.Section}\":\n{c.Text}\n\nScore (0-10):";
            var request = new ChatRequest(
                _model,
                new List<ChatMessage> { new("system", SystemPrompt), new("user", userPrompt) },
                Stream: false,
                // temperature 0 = same answer every run, so the demo is repeatable.
                Options: new Dictionary<string, object> { ["temperature"] = 0 });

            string reply = "";
            try
            {
                var response = await _http.PostAsJsonAsync("/api/chat", request);
                response.EnsureSuccessStatusCode();
                var body = await response.Content.ReadFromJsonAsync<ChatResponse>();
                reply = body?.Message?.Content ?? "";
            }
            catch (HttpRequestException ex)
            {
                reply = $"error: {ex.Message}";
            }

            double llmScore = ParseScore(reply);
            results.Add(candidate with { Score = llmScore, Reason = $"LLM said \"{Shorten(reply)}\"" });
        }

        // Ties are common with 0-10 scores, so break them using the original
        // vector score (kept in the original candidate list order).
        return results
            .Select((r, i) => (r, i))
            .OrderByDescending(x => x.r.Score)
            .ThenBy(x => x.i)
            .Select(x => x.r)
            .Take(topN)
            .ToList();
    }

    /// <summary>
    /// Small models don't always obey "ONLY the number". They might say
    /// "Score: 8" or "8/10" or "I'd give it 7.5". So we grab the FIRST number
    /// we see, clamp it to 0..10, and use 0 if there is no number at all.
    /// </summary>
    public static double ParseScore(string reply)
    {
        var match = Regex.Match(reply, @"\d+(\.\d+)?");
        if (!match.Success) return 0;
        if (!double.TryParse(match.Value, NumberStyles.Float, CultureInfo.InvariantCulture, out var value)) return 0;
        return Math.Clamp(value, 0, 10);
    }

    private static string Shorten(string s)
    {
        s = s.Replace("\n", " ").Trim();
        return s.Length <= 20 ? s : s[..20] + "...";
    }
}
