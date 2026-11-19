namespace MedGuideRAG;

/// <summary>
/// One searchable piece of a document. In RAG, we never search whole files:
/// we search small "chunks" so the answer we find is focused on one topic.
/// </summary>
public class Chunk
{
    public required string Drug { get; init; }        // e.g. "Wegovy"
    public required string Ingredient { get; init; }  // e.g. "semaglutide"
    public required string Section { get; init; }     // e.g. "If you miss a dose"
    public required string Text { get; init; }        // the section body

    // Filled in later by the Embedder (Step 3). It starts empty because
    // chunking and embedding are separate steps.
    public float[] Embedding { get; set; } = Array.Empty<float>();

    // The exact string we send to the embedding model. We put the drug and
    // section name in front of the text because a section like
    // "If you miss a dose" looks almost identical across 7 drugs. Without the
    // drug name, the vector would have no idea WHICH drug it belongs to.
    public string EmbeddingInput => $"{Drug} - {Section}: {Text}";
}

public static class Chunker
{
    // Drug name and ingredient for each file. We use a simple lookup table
    // instead of trying to "guess" from the text, because guessing is error
    // prone and this keeps the metadata 100% correct.
    private static readonly Dictionary<string, (string Drug, string Ingredient)> DrugInfo = new()
    {
        ["ozempic"]       = ("Ozempic",  "semaglutide"),
        ["wegovy"]        = ("Wegovy",   "semaglutide"),
        ["rybelsus"]      = ("Rybelsus", "semaglutide"),
        ["mounjaro"]      = ("Mounjaro", "tirzepatide"),
        ["zepbound"]      = ("Zepbound", "tirzepatide"),
        ["victoza"]       = ("Victoza",  "liraglutide"),
        ["saxenda"]       = ("Saxenda",  "liraglutide"),
        ["glp1-overview"] = ("GLP-1 Overview", "semaglutide, tirzepatide, liraglutide"),
    };

    /// <summary>
    /// Reads every .md file in the folder and splits it on "## " headings.
    /// Each heading + the lines under it become one chunk.
    /// </summary>
    public static List<Chunk> LoadAndChunk(string docsFolder)
    {
        var chunks = new List<Chunk>();

        // Sort the file list so the output is the same every run.
        foreach (var path in Directory.GetFiles(docsFolder, "*.md").OrderBy(p => p))
        {
            var fileKey = Path.GetFileNameWithoutExtension(path).ToLowerInvariant();
            var (drug, ingredient) = DrugInfo.TryGetValue(fileKey, out var info)
                ? info
                : (fileKey, "unknown"); // unknown files still work, just with less metadata

            string? currentSection = null;
            var currentLines = new List<string>();

            // Local helper: turn the lines collected so far into a chunk.
            void Flush()
            {
                var text = string.Join("\n", currentLines).Trim();
                // Skip the text before the first "##" (title + disclaimer):
                // it has no useful content to search.
                if (currentSection != null && text.Length > 0)
                {
                    chunks.Add(new Chunk { Drug = drug, Ingredient = ingredient, Section = currentSection, Text = text });
                }
                currentLines.Clear();
            }

            foreach (var line in File.ReadAllLines(path))
            {
                if (line.StartsWith("## "))
                {
                    Flush();                              // close the previous section
                    currentSection = line[3..].Trim();    // the heading becomes the section name
                }
                else
                {
                    currentLines.Add(line);
                }
            }
            Flush(); // don't forget the last section in the file
        }

        return chunks;
    }
}
