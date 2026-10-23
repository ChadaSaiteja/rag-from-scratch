"""The /visualize/ section: the scroll-driven "How RAG works" story (the section's
front page) and the hands-on playground. build_site.py calls render()."""
import html
import json
import shutil
from pathlib import Path

from viz_data import build as build_data

THREE_VERSION = "0.160.0"
ASSETS_SRC = Path(__file__).resolve().parent / "site_assets" / "viz"
STORY_SCRIPTS = ("story.js",)
PLAYGROUND_SCRIPTS = ("core.js", "space-common.js", "playground.js")

# Example output recorded in the repository README (python generation.py / evaluation.py).
README_ANSWER = ("Chunk size controls how large each split of a document is, while overlap controls how much "
                 "text is repeated between consecutive chunks so context isn't lost at the boundary "
                 "(Source: 03_chunking.txt).")
README_EVAL = [
    ("What is the difference between chunk size and overlap?", True, True),
    ("How does a vector database find similar embeddings?", True, True),
    ("What does the retriever stage do in a RAG pipeline?", True, True),
    ("How should a prompt be constructed for the generation stage?", True, True),
    ("Why should retrieval and generation quality be evaluated separately?", True, True),
]

# note stem -> story step, for the "See it step by step" callouts on module pages
NOTE_LINKS = {
    "01_foundations_transformers": "embedq",
    "02_embeddings_and_vectors": "embed",
    "03_prompting_techniques": "prompt",
    "04_chunking_strategies": "chunk",
    "05_vector_databases": "store",
    "06_retrieval_mechanics": "vsearch",
    "09_rag_evaluation": "evaluate",
    "10_document_ingestion_and_parsing": "load",
}

# (key, lane, title, what happens, why it matters, in this repo)
STORY = [
    ("overview", "", "RAG has two halves",
     "<b>Indexing</b> runs once: documents are prepared and stored. <b>Answering</b> runs for every question: find the right pieces, hand them to the model, check the result.",
     "Keeping the two apart is what makes RAG cheap to update (add a document, re-index only that) and easy to debug (which half failed?).",
     ""),
    ("documents", "index", "Start with documents",
     "This pipeline's knowledge base is 10 short lesson files about RAG itself, in <code>data/raw/</code>.",
     "Everything the system will ever answer comes from here. If a fact isn't in these files, the right answer is “I don't know”.",
     "data/raw/*.txt · 10 files"),
    ("load", "index", "Load and tag each file",
     "Each file is read into memory and tagged with metadata: its <code>filename</code>. Nothing else changes yet.",
     "The tag travels with every piece of the document from here on. It is what lets the final answer say <em>which file</em> a fact came from.",
     "ingest.py · ingest()"),
    ("chunk", "index", "Cut into chunks",
     "Each document is split into pieces of about 1,000 characters, cutting at paragraph breaks where possible. 200 characters are repeated between neighbours (amber) so no sentence is lost at a cut.",
     "A model can only read so much at once, and search works best on focused pieces. Chunk size is the single biggest lever on answer quality.",
     "ingest.py · RecursiveCharacterTextSplitter · 10 files → 21 chunks"),
    ("embed", "index", "Turn each chunk into numbers",
     "An embedding model converts each chunk into a long list of numbers (a vector). Chunks with similar meaning get similar numbers.",
     "Numbers can be compared mathematically; words can't. This is what lets a question find a chunk that says the same thing in different words.",
     "ingest.py · gemini-embedding-001 (shown here: a keyword stand-in)"),
    ("store", "index", "Store the vectors",
     "Every vector is saved in a vector database together with its text and its filename tag. Nearby vectors are chunks about similar things.",
     "Indexing is done. From now on the documents are never read again; every question is answered from this store.",
     "ingest.py · Chroma, collection gemini_collection · 21 vectors"),
    ("question", "answer", "A question arrives",
     "A user asks something. The system has no idea yet which of the 21 chunks matter.",
     "Everything that follows happens live, in well under a second for a store this size.",
     "generation.py · input()"),
    ("embedq", "answer", "Embed the question the same way",
     "The question goes through the <em>same</em> embedding model and lands in the same space as the chunks (the white dot).",
     "Same model, same space: that's the rule. Embed documents and questions differently and nothing is comparable.",
     "hybrid_search.py · GoogleGenerativeAIEmbeddings"),
    ("vsearch", "answer", "Vector search: find the nearest meanings",
     "The 3 chunks closest to the question are picked by cosine similarity (how aligned two vectors are).",
     "This catches chunks that answer the question in different words, which keyword search would miss.",
     "hybrid_search.py · vector_store.as_retriever(k=3) + MultiQueryRetriever"),
    ("bm25", "answer", "Keyword search: find exact words",
     "Independently, BM25 ranks chunks by the exact words they share with the question, weighting rare words more. Its top 3 get amber rings.",
     "Vectors blur exact terms like error codes, names and IDs. Keywords catch them. Each method covers the other's blind spot.",
     "hybrid_search.py · BM25Retriever(k=3)"),
    ("fuse", "answer", "Merge the two rankings",
     "Reciprocal Rank Fusion combines both lists by rank, not by score: each chunk scores 0.5 ÷ (60 + rank) per list, summed.",
     "The two methods score on different scales, so ranks are the only fair thing to add. Chunks both methods like rise to the top.",
     "hybrid_search.py · EnsembleRetriever(weights=[0.5, 0.5])"),
    ("prompt", "answer", "Build the prompt",
     "The winning chunks are pasted under a fixed instruction, each tagged <code>[Source: filename]</code>, followed by the question. This exact text is sent to the model.",
     "The model is told to use only this context, cite the file after every claim, and say “I don't know” otherwise. That instruction is the main defence against made-up answers.",
     "generation.py · format_context() + call_google_api()"),
    ("generate", "answer", "Generate a cited answer",
     "The language model (Gemini) writes the answer from the context, citing the file each claim came from.",
     "Because the citation points at a real chunk, anyone can check the answer against the source.",
     "generation.py · gemini-3.5-flash"),
    ("evaluate", "answer", "Check both halves separately",
     "For 5 test questions the pipeline checks two things: did retrieval find the expected file, and does the answer contain the expected idea?",
     "If an answer is wrong, this tells you which half to fix: the search or the writing. Judging only the final answer can't.",
     "evaluation.py · retrieval 5/5 · generation 5/5 (as recorded in the README)"),
    ("done", "", "That's the whole pipeline",
     "Documents in, cited answers out, with a check at the end. Every number you saw came from this repository's real corpus and settings.",
     "Now try it with your own question, or read the code that does each step.",
     ""),
]
STEP_INDEX = {key: i for i, (key, *_) in enumerate(STORY)}


def story_overlays(data, rel, repo_url):
    q = data["queries"][0]
    chunks = data["chunks"]
    lessons = data["lessons"]
    short = lambda c: f'{c["filename"].replace(".txt", "")} · chunk {c["index"] + 1}'
    rank_of = lambda lst, cid: next((i + 1 for i, (c, _) in enumerate(lst) if c == cid), "–")
    focus = next(l for l in lessons if l["filename"].startswith("03_"))
    fchunks = [chunks[i] for i in focus["chunkIds"]]

    overview = """
<div class="st-map" data-panel="overview">
  <div class="st-lane"><span class="st-lane-tag">Indexing · once</span>
    <div class="st-nodes"><span>Documents</span><i>→</i><span>Load &amp; tag</span><i>→</i><span>Chunk</span><i>→</i><span>Embed</span><i>→</i><span>Store</span></div></div>
  <div class="st-lane answer"><span class="st-lane-tag">Answering · every question</span>
    <div class="st-nodes"><span>Question</span><i>→</i><span>Embed</span><i>→</i><span>Search ×2</span><i>→</i><span>Merge</span><i>→</i><span>Prompt</span><i>→</i><span>Answer</span><i>→</i><span>Check</span></div></div>
</div>"""

    chunk_stats = f"""
<div class="st-note st-note-br" data-panel="chunk">
  <b>{html.escape(focus["filename"])}</b> → {len(fchunks)} chunks
  {"".join(f'<span>chunk {c["index"] + 1}: {c["chars"]} chars ≈ {c["tokens"]} tokens</span>' for c in fchunks)}
  <small>All 10 files → {len(chunks)} chunks · chunk_size={data["settings"]["chunkSize"]} · overlap={data["settings"]["chunkOverlap"]}</small>
</div>"""

    embed_note = f"""
<div class="st-note st-note-br" data-panel="embed">
  <b>Each chunk → one vector</b>
  <span>Bars: a chunk's strongest dimensions</span>
  <small>Real embeddings have thousands of dimensions; this page uses a {data["settings"]["vocab"]}-word keyword stand-in so it runs with no API key.</small>
</div>"""

    legend = f"""
<div class="st-note st-note-bl" data-panel="store embedq">
  <b>Colour = source file</b>
  {"".join(f'<span><i style="background:{l["color"]}"></i>{html.escape(l["filename"].replace(".txt", ""))}</span>' for l in lessons)}
</div>"""

    question = f"""
<div class="st-qcard" data-panel="question embedq vsearch bm25 fuse">
  <small>Question</small>
  <p>{html.escape(q["question"])}</p>
  <div class="st-words" data-panel-sub="bm25">{"".join(f"<span>{html.escape(t)}</span>" for t in q["terms"])}</div>
</div>"""

    vlist = "".join(
        f'<li><i style="background:{lessons[chunks[cid]["lesson"]]["color"]}"></i>{html.escape(short(chunks[cid]))}<code>cos {s:.2f}</code></li>'
        for cid, s in q["vector"])
    blist = "".join(
        f'<li><i style="background:{lessons[chunks[cid]["lesson"]]["color"]}"></i>{html.escape(short(chunks[cid]))}<code>{s:.1f}</code></li>'
        for cid, s in q["bm25"])
    vpanel = f'<div class="st-list" data-panel="vsearch"><b>Vector search · top 3</b><ol>{vlist}</ol></div>'
    bpanel = f'<div class="st-list amber" data-panel="bm25"><b>BM25 keyword search · top 3</b><ol>{blist}</ol></div>'

    fusion_rows = "".join(
        f'<tr><td><i style="background:{lessons[chunks[cid]["lesson"]]["color"]}"></i>{html.escape(short(chunks[cid]))}</td>'
        f'<td>{rank_of(q["bm25"], cid)}</td><td>{rank_of(q["vector"], cid)}</td><td><code>{s:.4f}</code></td></tr>'
        for cid, s in q["fused"])
    fusion = f"""
<div class="st-table" data-panel="fuse">
  <b>Reciprocal Rank Fusion</b>
  <table><tr><th>chunk</th><th>BM25 rank</th><th>vector rank</th><th>score</th></tr>{fusion_rows}</table>
  <small>score = 0.5/(60 + BM25 rank) + 0.5/(60 + vector rank)</small>
</div>"""

    ctx_cards = "".join(
        f'<div class="st-ctx" style="border-color:{lessons[chunks[cid]["lesson"]]["color"]}"><mark>[Source: {html.escape(chunks[cid]["filename"])}]</mark>'
        f'<span>{html.escape(chunks[cid]["text"][:150])}…</span></div>'
        for cid, _ in q["fused"])
    prompt = f"""
<div class="st-prompt" data-panel="prompt">
  <pre class="st-instr">{html.escape(data["promptHead"].strip())}</pre>
  <div class="st-ctxs"><small>Context:</small>{ctx_cards}</div>
  <pre class="st-q">Question: {html.escape(q["question"])}</pre>
</div>"""

    answer = f"""
<div class="st-chat" data-panel="generate">
  <div class="st-msg user"><small>Question</small>{html.escape(q["question"])}</div>
  <div class="st-msg bot"><small>Gemini, answering only from the context</small>
    <p data-answer="{html.escape(README_ANSWER.replace("(Source: 03_chunking.txt)", "<mark>(Source: 03_chunking.txt)</mark>"))}"></p>
    <em>Answer recorded in the README when the pipeline was run.</em></div>
</div>"""

    rows = "".join(
        f'<tr><td>{html.escape(qq)}</td><td class="{ "ok" if r else "bad" }">{"✓" if r else "✗"}</td><td class="{ "ok" if g else "bad" }">{"✓" if g else "✗"}</td></tr>'
        for qq, r, g in README_EVAL)
    score = f"""
<div class="st-score" data-panel="evaluate">
  <table><tr><th>Test question</th><th>Retrieval<br><small>expected file found?</small></th><th>Generation<br><small>expected idea in answer?</small></th></tr>{rows}</table>
  <div class="st-totals"><span>Retrieval <b>{sum(r for _, r, _ in README_EVAL)}/{len(README_EVAL)}</b></span><span>Generation <b>{sum(g for _, _, g in README_EVAL)}/{len(README_EVAL)}</b></span></div>
</div>"""

    done = f"""
<div class="st-done" data-panel="done">
  <h3>Documents in. Cited answers out.</h3>
  <div class="st-cta">
    <a class="btn primary" href="{rel}visualize/playground/">Try it with your own question →</a>
    <a class="btn" href="{rel}implementation/">Read the code, stage by stage</a>
    <a class="btn" href="{rel}learn/">Start the course</a>
  </div>
</div>"""
    return overview + chunk_stats + embed_note + legend + question + vpanel + bpanel + fusion + prompt + answer + score + done


def story_html(data, rel, repo_url):
    lane_name = {"index": "Indexing · once", "answer": "Answering · every question", "": ""}
    steps = []
    for i, (key, lane, title, what, why, where) in enumerate(STORY):
        repo = f'<p class="story-where"><span>In this repo</span>{html.escape(where)}</p>' if where else ""
        steps.append(
            f'<section class="story-step" id="{key}" data-step="{i}">'
            f'{f"<p class=eyebrow>{lane_name[lane]}</p>" if lane else ""}'
            f'<h2><span class="story-num">{i + 1}</span>{html.escape(title)}</h2>'
            f'<p class="story-what">{what}</p><p class="story-why"><span>Why it matters</span>{why}</p>{repo}</section>')
    rail = "".join(
        f'<button type="button" class="{lane or "plain"}" title="{html.escape(title)}" aria-label="Step {i + 1}: {html.escape(title)}"></button>'
        for i, (key, lane, title, *_) in enumerate(STORY))
    s = data["settings"]
    return f"""
<header class="viz-head">
  <p class="eyebrow">Step by step · {len(STORY)} steps</p>
  <h1>How RAG works, from documents to a cited answer</h1>
  <p class="lede">Scroll to follow this repository's real pipeline: {len(data["lessons"])} lesson files become {len(data["chunks"])} chunks, the chunks become vectors,
  and a question finds its evidence through two searches, a merge, a prompt and a check. Every number comes from the real code and corpus.</p>
</header>
<div class="story">
  <div class="story-stage-wrap">
    <div class="story-stage" aria-label="Animated stage">
      <svg class="story-svg" viewBox="0 0 1000 620" role="img" aria-hidden="true"></svg>
      <div class="story-overlay">{story_overlays(data, rel, repo_url)}</div>
      <div class="story-bar">
        <div class="story-rail" aria-label="Steps">{rail}</div>
        <div class="story-nav">
          <button type="button" data-nav="prev" aria-label="Previous step">←</button>
          <button type="button" data-nav="play" class="story-play">▶ Play</button>
          <button type="button" data-nav="next" aria-label="Next step">→</button>
          <span class="story-counter">1 / {len(STORY)}</span>
        </div>
      </div>
    </div>
  </div>
  <div class="story-steps">
{chr(10).join(steps)}
  </div>
</div>
<p class="viz-badge viz-badge-wide">What is real: the files, the 21 chunks and split points, the question, the BM25 scores, the rank fusion, the prompt text and the
recorded answer and scores. What is a stand-in: vector positions use TF-IDF keyword vectors reduced to 3D (computed offline, no API key) instead of Gemini embeddings,
so the layout of the point cloud would differ with real embeddings.</p>
"""


# ------------------------------------------------------------------ playground (unchanged)

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
  <div class="pg-tabs" role="tablist" aria-label="Pipeline part">
    <button type="button" role="tab" class="pg-tab" id="tab-part-a" data-part="a" aria-controls="part-a" aria-selected="false"><b>A · Indexing</b><small>steps 1–5 · once per document</small></button>
    <button type="button" role="tab" class="pg-tab" id="tab-part-b" data-part="b" aria-controls="part-b" aria-selected="true"><b>B · Answering</b><small>steps 6–16 · every question</small></button>
  </div>

  <section class="pg-part pg-part-a" id="part-a" role="tabpanel" aria-labelledby="tab-part-a" hidden>
    <div class="pg-part-head"><span class="pg-part-tag">Part A</span><h2>Indexing: prepare the documents</h2>
      <label class="viz-pick">Lesson <select class="pg-lesson viz-picker">{lesson_opts}</select></label></div>
    <ol class="pg-flow">
{part_a}
    </ol>
  </section>

  <section class="pg-part pg-part-b" id="part-b" role="tabpanel" aria-labelledby="tab-part-b">
    <div class="pg-part-head"><span class="pg-part-tag">Part B</span><h2>Answering: one question, end to end</h2>
      <span class="pg-part-note">Uses the 21 chunks prepared in Part A</span></div>
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
  <div><h2>Keep going</h2><ul>
    <li><a href="{rel}visualize/">How RAG works, step by step (the guided story)</a></li>
    <li><a href="{rel}learn/">The course: 12 modules</a></li></ul></div>
</section>
"""


# ------------------------------------------------------------------ shared

def scripts_html(rel, scripts, importmap=False):
    out = ""
    if importmap:
        im = {"imports": {
            "three": f"https://cdn.jsdelivr.net/npm/three@{THREE_VERSION}/build/three.module.js",
            "three/addons/": f"https://cdn.jsdelivr.net/npm/three@{THREE_VERSION}/examples/jsm/",
        }}
        out += f'<script type="importmap">{json.dumps(im)}</script>\n'
    out += f'<script src="{rel}assets/viz/data.js"></script>\n'
    out += "".join(f'<script type="module" src="{rel}assets/viz/{s}"></script>\n' for s in scripts)
    return out


def callout(note_stem, rel):
    key = NOTE_LINKS.get(note_stem)
    if not key:
        return ""
    title = next(t for k, _, t, *_ in STORY if k == key)
    return (f'<a class="viz-callout" href="{rel}visualize/#{key}"><span>▶ See it step by step</span>'
            f'<strong>{html.escape(title)}</strong></a>')


def render(*, page, breadcrumbs_html, breadcrumb_ld, out_dir: Path, lessons_dir: Path, site_url, site_name, repo_url, today):
    """Write data, scripts and pages. Returns sitemap entries [(path, lastmod, priority)]."""
    data = build_data(lessons_dir)
    assets = out_dir / "assets" / "viz"
    assets.mkdir(parents=True, exist_ok=True)
    (assets / "data.js").write_text("window.RAG_VIZ = " + json.dumps(data, ensure_ascii=False, separators=(",", ":")) + ";\n",
                                    encoding="utf-8", newline="\n")
    for name in STORY_SCRIPTS + PLAYGROUND_SCRIPTS:
        shutil.copyfile(ASSETS_SRC / name, assets / name)

    entries = []
    rel = "../"
    crumbs = [("Home", ""), ("How RAG works", "visualize/")]
    story_ld = [{"@context": "https://schema.org", "@type": "LearningResource", "name": "How RAG works, step by step",
                 "description": "A scroll-driven visual walkthrough of a real RAG pipeline: documents, chunks, embeddings, vector store, hybrid search, fusion, prompt, cited answer and evaluation.",
                 "url": site_url + "visualize/", "learningResourceType": "Interactive visualization", "educationalLevel": "Beginner",
                 "inLanguage": "en", "isPartOf": {"@type": "Course", "name": site_name, "url": site_url},
                 "hasPart": [{"@type": "LearningResource", "name": "RAG Pipeline Playground", "url": site_url + "visualize/playground/"}]},
                breadcrumb_ld(crumbs)]
    out = out_dir / "visualize" / "index.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page(path="visualize/", title=f"How RAG Works, Step by Step: Visual Walkthrough of a Real Pipeline | {site_name}",
                        description="Scroll through a real RAG pipeline: documents become chunks and vectors, a question finds its evidence by vector and keyword search, and the model writes a cited answer.",
                        body=breadcrumbs_html(crumbs, rel) + story_html(data, rel, repo_url), rel=rel, ld=story_ld, active="visualize/",
                        wide=True, scripts=scripts_html(rel, STORY_SCRIPTS)),
                   encoding="utf-8", newline="\n")
    entries.append(("visualize/", today, "0.9"))

    rel = "../../"
    pg = PLAYGROUND
    path = f"visualize/{pg['slug']}/"
    crumbs = [("Home", ""), ("How RAG works", "visualize/"), (pg["title"], path)]
    ld = [{"@context": "https://schema.org", "@type": "LearningResource", "name": pg["title"], "description": pg["description"],
           "url": site_url + path, "learningResourceType": "Interactive simulation", "educationalLevel": "Beginner",
           "inLanguage": "en", "isPartOf": {"@type": "Course", "name": site_name, "url": site_url}},
          breadcrumb_ld(crumbs)]
    out = out_dir / path / "index.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page(path=path, title=f"{pg['seo_title']} | {site_name}", description=pg["description"],
                        body=breadcrumbs_html(crumbs, rel) + playground_html(data, rel, repo_url), rel=rel, og_type="article",
                        ld=ld, active="visualize/", wide=True, scripts=scripts_html(rel, ("playground.js",), importmap=True)),
                   encoding="utf-8", newline="\n")
    entries.append((path, today, "0.8"))
    return entries, data
