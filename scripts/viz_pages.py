"""The /visualize/ section: page content (steps double as captions and as indexable
text) plus the HTML shell around each 3D scene. build_site.py calls render()."""
import html
import json
import shutil
from pathlib import Path

from viz_data import build as build_data

THREE_VERSION = "0.160.0"
ASSETS_SRC = Path(__file__).resolve().parent / "site_assets" / "viz"

NOTE_LINKS = {  # note stem -> visualization slug, for "See it in 3D" callouts
    "01_foundations_transformers": "playground",
    "04_chunking_strategies": "chunking",
    "02_embeddings_and_vectors": "embedding-and-storage",
    "05_vector_databases": "embedding-and-storage",
    "06_retrieval_mechanics": "retrieval",
}

PAGES = [
    {
        "slug": "chunking", "scene": "chunking", "stage": "Stage 1 · Chunking",
        "title": "Chunking, Visualized in 3D",
        "seo_title": "RAG Chunking Visualized: How Text Is Split, Step by Step",
        "description": "Watch a real document split into chunks in 3D: paragraph breaks, 1,000-character cuts, overlap and metadata, using the same splitter settings as the pipeline.",
        "lede": "Watch one of the course lessons go through the exact splitter this repo uses: <code>RecursiveCharacterTextSplitter</code> with 1,000-character chunks and 200 characters of overlap.",
        "badge": "Real data: the lesson text and every split point come from running the pipeline's splitter at build time.",
        "picker": "lesson",
        "code": [("src/pipeline/ingest.py", "chunk() splits each document and copies its metadata to every chunk")],
        "learn": [("learn/04-chunking-strategies/", "Module 04 · Chunking Strategies")],
        "steps": [
            ("One document", "A lesson file loaded from <code>data/raw/</code>. Too long to embed or retrieve as one unit, so it has to be split."),
            ("Look for natural breaks", "The splitter tries separators in order: blank lines (paragraphs) first, then single newlines, then spaces. Grey lines mark the paragraph breaks it prefers to cut at."),
            ("Cut at about 1,000 characters", "Scanning down the text, it closes a chunk at the last good break before 1,000 characters. The cyan lines are the real cut points for this lesson."),
            ("Chunks separate", "Each piece becomes its own chunk. Hover a chunk to see its exact size and where it sits in the original text."),
            ("Overlap carries context", "The amber bands are text repeated between neighbouring chunks, so a sentence near a cut is still whole in at least one chunk. The default overlap is up to 200 characters."),
            ("Characters are not tokens", "<code>chunk_size</code> counts characters, not tokens. In English, 1,000 characters is roughly 250 tokens, which is what models and costs actually measure."),
            ("Stamped with metadata", "Every chunk carries its source <code>filename</code>. Without it, the final answer couldn't cite where a fact came from. Next: turning each chunk into numbers."),
        ],
    },
    {
        "slug": "embedding-and-storage", "scene": "space", "stage": "Stage 2 · Embedding & storage",
        "title": "Embeddings and Vector Storage, Visualized in 3D",
        "seo_title": "Embeddings & Vector Databases Visualized in 3D | How RAG Stores Data",
        "description": "See text chunks become vectors and points in 3D space, cluster by topic, link to their nearest neighbours, and get stored in a vector database with metadata.",
        "lede": "Every chunk becomes a list of numbers, and those numbers become a position. Similar chunks land close together, and the index links neighbours so search doesn't have to check everything.",
        "badge": "Approximation: positions are TF-IDF keyword vectors reduced to 3D with PCA, computed offline. Real Gemini embeddings group chunks by meaning rather than shared words, so their layout would differ.",
        "picker": None,
        "code": [("src/pipeline/ingest.py", "embeds every chunk and writes it to Chroma with its metadata"),
                 ("src/pipeline/vector_retrieval.py", "reopens the store and searches it")],
        "learn": [("learn/02-embeddings-and-vectors/", "Module 02 · Embeddings & Vectors"),
                  ("learn/05-vector-databases/", "Module 05 · Vector Databases")],
        "steps": [
            ("Chunks of text", "All 21 chunks from the 10 lessons, coloured by source lesson. Hover any card to read it."),
            ("Text becomes numbers", "Each chunk is turned into a long vector of numbers. The bars show each chunk's strongest dimensions; a real embedding model like <code>gemini-embedding-001</code> outputs thousands."),
            ("Numbers become a position", "Reduced to three dimensions, each vector becomes a point in space. Drag to orbit and scroll to zoom."),
            ("Similar topics land close together", "Chunks about related ideas sit near each other. This is what lets a question find relevant text even when the wording differs."),
            ("Nearest neighbours", "For one chunk, these are its three most similar chunks by cosine similarity: the same measure used to rank search results."),
            ("The index: links between neighbours", "Vector databases connect each point to its neighbours. A search hops along these links toward the target (the yellow path) instead of comparing against every stored vector. That's the idea behind HNSW."),
            ("Stored with metadata", "Each point is saved with its text and filename in a Chroma collection on disk. Next: asking a question."),
        ],
    },
    {
        "slug": "retrieval", "scene": "retrieval", "stage": "Stage 3 · Retrieval",
        "title": "Hybrid Retrieval, Visualized in 3D",
        "seo_title": "Hybrid Search Visualized: Vector + BM25 + Reciprocal Rank Fusion",
        "description": "Watch a question become a point, find its nearest chunks by vector search and keyword (BM25) search, and merge both rankings with Reciprocal Rank Fusion.",
        "lede": "Pick one of the pipeline's evaluation questions and watch hybrid search work: two retrievers each pick their top 3, and Reciprocal Rank Fusion merges them into one ranked list.",
        "badge": "Real rankings from the real chunks, using BM25 and weighted Reciprocal Rank Fusion (50/50, c=60) as in the pipeline. The vector side uses the same TF-IDF stand-in as the previous page, and a single query: the real pipeline also asks Gemini for 3 rephrasings.",
        "picker": "query",
        "code": [("src/pipeline/hybrid_search.py", "BM25Retriever + vector retriever fused by EnsembleRetriever"),
                  ("src/pipeline/evaluation.py", "the five questions and their expected source files")],
        "learn": [("learn/06-retrieval-mechanics/", "Module 06 · Retrieval Mechanics"),
                  ("learn/09-rag-evaluation/", "Module 09 · RAG Evaluation")],
        "steps": [
            ("The stored space", "All 21 stored chunks, waiting. Choose a question above to change the search."),
            ("The question becomes a point", "The question is embedded with the same model as the chunks, so it lands in the same space."),
            ("Vector search", "The three chunks most similar to the question by cosine similarity. Good at matching meaning."),
            ("Keyword search (BM25)", "BM25 scores exact shared words, weighting rare words more. Good at names, codes and precise terms that vectors can blur."),
            ("Reciprocal Rank Fusion", "Each retriever's ranking is converted to a score of 0.5/(60 + rank) and the scores are added. Chunks both retrievers like rise to the top, and the two scoring scales never need comparing."),
            ("Handed to the prompt", "The fused chunks go into the prompt, each tagged with its source filename so the answer can cite it. The panel shows whether the expected file was retrieved."),
        ],
    },
]


PLAYGROUND = {
    "slug": "playground", "stage": "Playground · The whole pipeline",
    "title": "RAG Pipeline Playground",
    "seo_title": "RAG Pipeline Playground: From Documents to a Grounded Prompt, Step by Step",
    "description": "Follow the full RAG pipeline in one page: load, clean, chunk, vectorize and store a document, then type a question and watch tokens, vectors, hybrid search, fusion and the final prompt.",
    "examples": [
        "What is the difference between chunk size and overlap?",
        "Which vector database is best for prototyping?",
        "Why should retrieval and generation quality be evaluated separately?",
        "What is the capital of France?",
    ],
    "part_a": [
        ("load", "Load the document", "A raw lesson file is read from <code>data/raw/</code>. Pick any of the 10 lessons above."),
        ("clean", "Clean the text", "Normalise whitespace and strip invisible characters so chunks and vectors aren't polluted by noise. "
         "This repo's <code>ingest.py</code> skips this step because the lessons are already clean."),
        ("chunk", "Split into chunks", "The text is cut into pieces of at most 1,000 characters. Shaded colours are chunks; amber text appears in two chunks (overlap). Hover to see which."),
        ("vectorize", "Turn each chunk into a vector", "Every chunk becomes a list of numbers, one per vocabulary word, weighted so rare, distinctive words count most (TF-IDF here; a real embedding model in production)."),
        ("store", "Store with metadata", "Each chunk is saved with an id, its text, its vector and its filename. The filename is what makes citations possible later."),
    ],
    "part_b": [
        ("sentence", "Your question", "Exactly what a user would type."),
        ("tokens", "Split into tokens",
         "Lowercased and split into pieces. Very common words like <em>how</em> and <em>the</em> carry little meaning, so this representation drops them. "
         "Real LLM tokenizers go further and split rare words into sub-word pieces such as <code>retriev</code> + <code>al</code>."),
        ("ids", "Look up token IDs", "Each word is replaced by its position in the vocabulary. Words the vocabulary has never seen can't be represented."),
        ("vector", "Build the query vector",
         "One slot per vocabulary word, the same layout as the chunk vectors in Part A. Brighter cells are rarer, more distinctive words. "
         "Real embedding models produce a few hundred to a few thousand dense numbers instead."),
        ("space", "Place it in space", "Projected to 3D next to the 21 stored chunks. Drag to orbit, hover a point to read its chunk."),
        ("vsearch", "Vector search", "Rank every stored chunk by cosine similarity to the query vector and keep the top 3."),
        ("bm25", "Keyword search (BM25)", "Independently, rank chunks by exact shared words. This catches names and codes that vectors can blur."),
        ("fusion", "Merge with Reciprocal Rank Fusion", "Combine both rankings into one list, as LangChain's <code>EnsembleRetriever</code> does in <code>hybrid_search.py</code>."),
        ("prompt", "Build the prompt", "The fused chunks are tagged with <code>[Source: filename]</code> and placed under the instructions from <code>generation.py</code>. This is the exact text that would be sent to Gemini."),
        ("ground", "Grounding preview", "No model runs here, so there is no generated answer. These are the context sentences that share the most words with your question: the evidence a grounded answer would cite."),
        ("check", "Check the result", "Before trusting an answer, check the evidence: is it relevant enough, can every claim be cited, and for test questions, was the expected source retrieved?"),
    ],
}


def _step(i, key, title, text, extra=""):
    return (f'<li class="pg-step" data-stage="{key}"><div class="pg-step-head"><span class="pg-num">{i}</span>'
            f'<div><h2>{html.escape(title)}</h2><p>{text}</p></div></div><div class="pg-out">{extra}</div></li>')


def playground_html(data, rel, repo_url):
    p = PLAYGROUND
    lesson_opts = "".join(
        f'<option value="{l["id"]}"{" selected" if l["filename"].startswith("03_") else ""}>{html.escape(l["filename"])}</option>'
        for l in data["lessons"])
    part_a = "\n".join(_step(i, *s) for i, s in enumerate(p["part_a"], start=1))
    part_b = []
    for i, (key, title, text) in enumerate(p["part_b"], start=len(p["part_a"]) + 1):
        extra = ""
        if key == "vector":
            extra = '<div class="pg-grid" role="img" aria-label="One cell per vocabulary word; lit cells are words in your question"></div><p class="pg-vector-meta pg-meta"></p>'
        if key == "space":
            extra = ('<div class="pg-stage viz-stage"><div class="viz-tooltip" hidden></div>'
                     '<p class="viz-fallback" hidden>3D needs WebGL, which this browser does not support. The position is listed below.</p></div>'
                     '<p class="pg-space-meta pg-meta"></p>')
        part_b.append(_step(i, key, title, text, extra))
    examples = "".join(f'<button type="button" class="pg-ex" data-example="{html.escape(e)}">{html.escape(e)}</button>' for e in p["examples"])
    return f"""
<header class="viz-head">
  <p class="eyebrow">{html.escape(p["stage"])}</p>
  <h1>{html.escape(p["title"])}</h1>
  <p class="lede">Every step of Retrieval-Augmented Generation on one page. Part A prepares the documents once. Part B runs for every question:
  from your sentence to the exact prompt an LLM would receive.</p>
</header>
<div class="pg">
  <p class="viz-badge">No AI model runs here. Vectors are TF-IDF keyword representations over this repo's {data["settings"]["vocab"]}-word vocabulary,
  computed in your browser with the same math as the other visualizations. Chunks, BM25, rank fusion and the prompt template are the real ones.</p>
  <nav class="pg-parts" aria-label="Parts"><a href="#part-a">A · Indexing (once)</a><a href="#part-b">B · Answering (per question)</a></nav>

  <section class="pg-part pg-part-a" id="part-a">
    <div class="pg-part-head"><span class="pg-part-tag">Part A</span><h2>Indexing: prepare the documents</h2>
      <label class="viz-pick">Lesson <select class="pg-lesson viz-picker">{lesson_opts}</select></label></div>
    <ol class="pg-flow">
{part_a}
    </ol>
  </section>

  <section class="pg-part pg-part-b" id="part-b">
    <div class="pg-part-head"><span class="pg-part-tag">Part B</span><h2>Answering: one question, end to end</h2></div>
    <form class="pg-input" role="search">
      <label for="pg-text">Your question</label>
      <div class="pg-input-row"><input id="pg-text" class="pg-text" type="text" maxlength="200" autocomplete="off" value="{html.escape(p["examples"][0])}"><button type="submit">Run ▶</button></div>
      <div class="pg-examples"><span>Try:</span>{examples}</div>
    </form>
    <ol class="pg-flow">
{chr(10).join(part_b)}
    </ol>
  </section>
</div>
<section class="viz-links">
  <div><h2>Where this happens in the code</h2><ul>
    <li><a href="{repo_url}/blob/master/src/pipeline/ingest.py" rel="noopener"><code>ingest.py</code></a>: load, chunk, embed, store</li>
    <li><a href="{repo_url}/blob/master/src/pipeline/hybrid_search.py" rel="noopener"><code>hybrid_search.py</code></a>: BM25 + vector + fusion</li>
    <li><a href="{repo_url}/blob/master/src/pipeline/generation.py" rel="noopener"><code>generation.py</code></a>: prompt and answer</li>
    <li><a href="{repo_url}/blob/master/src/pipeline/evaluation.py" rel="noopener"><code>evaluation.py</code></a>: the retrieval and generation checks</li></ul></div>
  <div><h2>See the 3D stages</h2><ul>{"".join(f'<li><a href="{rel}visualize/{x["slug"]}/">{html.escape(x["title"])}</a></li>' for x in PAGES)}</ul></div>
</section>
"""


def page_html(p, data, rel, repo_url, idx):
    picker = ""
    if p["picker"] == "lesson":
        options = "".join(
            f'<option value="{l["id"]}"{" selected" if l["filename"].startswith("03_") else ""}>{html.escape(l["filename"])}</option>'
            for l in data["lessons"])
        picker = f'<label class="viz-pick">Lesson <select class="viz-picker">{options}</select></label>'
    elif p["picker"] == "query":
        options = "".join(f'<option value="{i}">{html.escape(q["question"])}</option>' for i, q in enumerate(data["queries"]))
        picker = f'<label class="viz-pick">Question <select class="viz-picker">{options}</select></label>'

    steps = "\n".join(
        f'<li data-step="{i}"><h3>{html.escape(t)}</h3><p>{d}</p></li>' for i, (t, d) in enumerate(p["steps"]))
    code = "".join(f'<li><a href="{repo_url}/blob/master/{path}" rel="noopener"><code>{path}</code></a>: {html.escape(why)}</li>'
                   for path, why in p["code"])
    learn = "".join(f'<li><a href="{rel}{href}">{html.escape(label)}</a></li>' for href, label in p["learn"])
    nxt = PAGES[idx + 1] if idx + 1 < len(PAGES) else None
    prev = PAGES[idx - 1] if idx > 0 else None
    pager = '<nav class="pager" aria-label="Visualization navigation">'
    pager += (f'<a class="pager-link" href="{rel}visualize/{prev["slug"]}/" rel="prev"><small>Previous stage</small>{html.escape(prev["title"])}</a>'
              if prev else f'<a class="pager-link" href="{rel}visualize/"><small>Back to</small>All visualizations</a>')
    pager += (f'<a class="pager-link next" href="{rel}visualize/{nxt["slug"]}/" rel="next"><small>Next stage</small>{html.escape(nxt["title"])}</a>'
              if nxt else f'<a class="pager-link next" href="{rel}implementation/"><small>Next</small>See the full pipeline code</a>')
    pager += "</nav>"

    return f"""
<header class="viz-head">
  <p class="eyebrow">{html.escape(p["stage"])}</p>
  <h1>{html.escape(p["title"])}</h1>
  <p class="lede">{p["lede"]}</p>
</header>
<div class="viz" data-scene="{p["scene"]}" tabindex="0" aria-label="Interactive 3D visualization. Use the left and right arrow keys to change step.">
  <div class="viz-toolbar">{picker}<p class="viz-badge">{html.escape(p["badge"])}</p></div>
  <div class="viz-stage">
    <div class="viz-hud" aria-live="polite"></div>
    <div class="viz-caption" aria-live="polite"></div>
    <div class="viz-tooltip" hidden></div>
    <p class="viz-fallback" hidden>3D needs WebGL, which this browser doesn't support. The steps below describe each stage of the animation.</p>
    <span class="viz-hint">Drag to orbit · scroll to zoom · hover to read</span>
  </div>
  <div class="viz-controls">
    <button type="button" data-action="prev" aria-label="Previous step">←</button>
    <button type="button" data-action="play" class="viz-play">▶ Play</button>
    <button type="button" data-action="next" aria-label="Next step">→</button>
    <div class="viz-dots"></div>
  </div>
  <ol class="viz-steps">
{steps}
  </ol>
</div>
<section class="viz-links">
  <div><h2>Where this happens in the code</h2><ul>{code}</ul></div>
  <div><h2>Learn the concepts</h2><ul>{learn}</ul></div>
</section>
{pager}
"""


def hub_html(data, rel):
    cards = "".join(
        f'<a class="viz-card" href="{rel}visualize/{p["slug"]}/"><span class="eyebrow">{html.escape(p["stage"])}</span>'
        f'<h3>{html.escape(p["title"])}</h3><p>{html.escape(p["description"])}</p><span class="viz-card-go">Open the 3D view →</span></a>'
        for p in PAGES)
    pg = PLAYGROUND
    cards = (f'<a class="viz-card viz-card-feature" href="{rel}visualize/{pg["slug"]}/"><span class="eyebrow">{html.escape(pg["stage"])}</span>'
             f'<h3>{html.escape(pg["title"])}</h3><p>{html.escape(pg["description"])}</p><span class="viz-card-go">Try it →</span></a>') + cards
    s = data["settings"]
    return f"""
<header class="viz-head">
  <p class="eyebrow">Interactive · 3D</p>
  <h1>RAG, Visualized</h1>
  <p class="lede">See what actually happens to the data at each stage of a RAG pipeline: how a document is cut into chunks, how chunks become points in space,
  how they're stored, and how a question finds the right ones. Every scene runs on this repo's real corpus: {len(data["lessons"])} lessons,
  {len(data["chunks"])} chunks, {s["chunkSize"]:,}-character chunks with {s["chunkOverlap"]} overlap, top {s["k"]} per retriever.</p>
</header>
<div class="viz-grid">{cards}</div>
<p class="viz-badge viz-badge-wide">Positions in 3D use TF-IDF keyword vectors reduced with PCA, computed offline so no API key is needed. Chunks, split points, BM25 scores
and rank fusion are the real computations. Coming next: loading &amp; cleaning, generation, and evaluation.</p>
"""


def scripts_html(rel, scene):
    importmap = {"imports": {
        "three": f"https://cdn.jsdelivr.net/npm/three@{THREE_VERSION}/build/three.module.js",
        "three/addons/": f"https://cdn.jsdelivr.net/npm/three@{THREE_VERSION}/examples/jsm/",
    }}
    return (f'<script type="importmap">{json.dumps(importmap)}</script>\n'
            f'<script src="{rel}assets/viz/data.js"></script>\n'
            f'<script type="module" src="{rel}assets/viz/{scene}.js"></script>')


def callout(note_stem, rel):
    slug = NOTE_LINKS.get(note_stem)
    if not slug:
        return ""
    p = next(x for x in PAGES + [PLAYGROUND] if x["slug"] == slug)
    return (f'<a class="viz-callout" href="{rel}visualize/{slug}/"><span>▶ See it in 3D</span>'
            f'<strong>{html.escape(p["title"])}</strong></a>')


def render(*, page, breadcrumbs_html, breadcrumb_ld, out_dir: Path, lessons_dir: Path, site_url, site_name, repo_url, today):
    """Write data, scripts and pages. Returns sitemap entries [(path, lastmod, priority)]."""
    data = build_data(lessons_dir)
    assets = out_dir / "assets" / "viz"
    assets.mkdir(parents=True, exist_ok=True)
    (assets / "data.js").write_text("window.RAG_VIZ = " + json.dumps(data, ensure_ascii=False, separators=(",", ":")) + ";\n",
                                    encoding="utf-8", newline="\n")
    for src in ASSETS_SRC.glob("*.js"):
        shutil.copyfile(src, assets / src.name)

    entries = []
    rel = "../"
    crumbs = [("Home", ""), ("Visualize", "visualize/")]
    hub_ld = [{"@context": "https://schema.org", "@type": "CollectionPage", "name": "RAG, Visualized",
               "description": "Interactive 3D visualizations of each stage of a RAG pipeline.", "url": site_url + "visualize/",
               "hasPart": [{"@type": "LearningResource", "name": p["title"], "url": f"{site_url}visualize/{p['slug']}/"} for p in PAGES]},
              breadcrumb_ld(crumbs)]
    out = out_dir / "visualize" / "index.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page(path="visualize/", title=f"RAG Visualized in 3D: See Each Pipeline Stage | {site_name}",
                        description="Interactive 3D animations of a RAG pipeline on real data: chunking, embeddings, vector storage and hybrid retrieval, step by step.",
                        body=breadcrumbs_html(crumbs, rel) + hub_html(data, rel), rel=rel, ld=hub_ld, active="visualize/", wide=True),
                   encoding="utf-8", newline="\n")
    entries.append(("visualize/", today, "0.9"))

    rel = "../../"
    pg = PLAYGROUND
    path = f"visualize/{pg['slug']}/"
    crumbs = [("Home", ""), ("Visualize", "visualize/"), (pg["title"], path)]
    ld = [{"@context": "https://schema.org", "@type": "LearningResource", "name": pg["title"], "description": pg["description"],
           "url": site_url + path, "learningResourceType": "Interactive simulation", "educationalLevel": "Beginner",
           "inLanguage": "en", "isPartOf": {"@type": "Course", "name": site_name, "url": site_url}},
          breadcrumb_ld(crumbs)]
    out = out_dir / path / "index.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page(path=path, title=f"{pg['seo_title']} | {site_name}", description=pg["description"],
                        body=breadcrumbs_html(crumbs, rel) + playground_html(data, rel, repo_url), rel=rel, og_type="article",
                        ld=ld, active="visualize/", wide=True, scripts=scripts_html(rel, "playground")),
                   encoding="utf-8", newline="\n")
    entries.append((path, today, "0.8"))

    for i, p in enumerate(PAGES):
        path = f"visualize/{p['slug']}/"
        crumbs = [("Home", ""), ("Visualize", "visualize/"), (p["title"], path)]
        ld = [{"@context": "https://schema.org", "@type": "LearningResource", "name": p["title"], "description": p["description"],
               "url": site_url + path, "learningResourceType": "Interactive visualization", "educationalLevel": "Beginner",
               "inLanguage": "en", "isPartOf": {"@type": "Course", "name": site_name, "url": site_url}},
              breadcrumb_ld(crumbs)]
        body = breadcrumbs_html(crumbs, rel) + page_html(p, data, rel, repo_url, i)
        out = out_dir / path / "index.html"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(page(path=path, title=f"{p['seo_title']} | {site_name}", description=p["description"], body=body, rel=rel,
                            og_type="article", ld=ld, active="visualize/", wide=True, scripts=scripts_html(rel, p["scene"])),
                       encoding="utf-8", newline="\n")
        entries.append((path, today, "0.8"))
    return entries, data
