"""Build docs/index.html: a single, self-contained page that explains this repo
and renders every study note in notes/ so it can be read in a browser.

Run from anywhere:  python scripts/build_site.py
Re-run it whenever notes/ or src/pipeline/ change; the page is fully regenerated.
"""
import html
import re
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parents[1]
NOTES_DIR = ROOT / "notes"
PIPELINE_DIR = ROOT / "src" / "pipeline"
LESSONS_DIR = ROOT / "data" / "raw"
OUT_FILE = ROOT / "docs" / "index.html"

# Pipeline files in the order data flows through them, with a one-line role each.
PIPELINE_FILES = [
    ("config.py", "Shared settings", "Loads the API key, model names, chunk size and top-k from <code>.env</code>, so every other script is retargeted by editing <code>.env</code> only."),
    ("ingest.py", "Stages 1–4 · Load → chunk → embed → store", "Reads every lesson in <code>data/raw/</code>, splits it into ~1000-character chunks, tags each chunk with its source <code>filename</code>, embeds it with Gemini and persists it to Chroma."),
    ("vector_retrieval.py", "Stage 5a · Semantic retrieval", "The simplest retriever: top-k nearest neighbours by embedding similarity. Useful as a baseline to compare hybrid search against."),
    ("hybrid_search.py", "Stage 5b · Hybrid retrieval", "BM25 (exact keywords) and a multi-query-expanded vector retriever, fused 50/50 by an <code>EnsembleRetriever</code>."),
    ("generation.py", "Stages 6–7 · Prompt + generate", "Wraps each retrieved chunk in a <code>[Source: …]</code> tag and asks Gemini to answer only from that context, cite filenames, and say “I don’t know” otherwise."),
    ("evaluation.py", "Stage 8 · Evaluation", "Runs fixed question → expected-source cases and scores retrieval (right file found?) and generation (expected concept in answer?) separately."),
]

KNOWN_ISSUES = [
    ("Relative persist directory", "<code>./chroma_gemini_db</code> resolves against the working directory, so ingesting and querying from different folders use different stores."),
    ("Re-ingest duplicates data", "<code>ingest_pipeline</code> appends with no reset or deterministic IDs; running it twice doubles the corpus."),
    ("Import-time side effects", "<code>vector_retrieval.py</code> and <code>hybrid_search.py</code> do real work (API calls, index build) when imported."),
    ("No relevance floor", "Retrieval always returns top-k, so off-topic questions still feed chunks to the LLM."),
    ("Flat imports", "Scripts import each other as <code>from config import …</code>, so they only run from <code>src/pipeline/</code>."),
    ("Weak generation check", "A 503 returns an apology string that evaluation still scores, and the keyword check passes easily."),
    ("No automated tests", "<code>evaluation.py</code> checks behaviour, but there is no pytest suite."),
]


def note_id(stem):
    return "notes-guide" if stem.upper() == "README" else f"note-{stem}"


def rewrite_note_links(rendered):
    """Turn links between note files (./02_x.md#frag) into in-page anchors."""
    def repl(m):
        stem = m.group(1)
        return f'href="#{note_id(stem)}"'
    return re.sub(r'href="(?:\./)?([\w-]+)\.md(?:#[^"]*)?"', repl, rendered)


def render_note(path):
    prefix = note_id(path.stem)

    def slugify(value, separator):
        slug = re.sub(r"[^\w\s-]", "", value, flags=re.UNICODE).strip().lower()
        return f"{prefix}--" + re.sub(r"[\s_-]+", separator, slug)

    md = markdown.Markdown(
        extensions=["tables", "fenced_code", "sane_lists", "toc"],
        extension_configs={"toc": {"slugify": slugify}},
    )
    body = md.convert(path.read_text(encoding="utf-8"))
    body = rewrite_note_links(body)
    # Wrap tables so wide ones scroll inside the card instead of the page.
    body = body.replace("<table>", '<div class="table-wrap"><table>').replace("</table>", "</table></div>")
    title_match = re.search(r"^#\s+(.+)$", path.read_text(encoding="utf-8"), flags=re.MULTILINE)
    title = title_match.group(1).strip() if title_match else path.stem
    return prefix, title, body


def clean_title(title):
    """Drop leading emoji/symbols for the sidebar label."""
    title = re.sub(r"^[^\w(]+", "", title).strip()
    # The sidebar already shows the module number, so drop "Module N:" and any "(…)" expansion.
    title = re.sub(r"^Module\s+\d+:\s*", "", title)
    return re.sub(r"\s*\(.*\)$", "", title)


def build():
    note_paths = sorted(p for p in NOTES_DIR.glob("*.md") if p.stem.upper() != "README")
    readme_path = NOTES_DIR / "README.md"
    notes = [render_note(readme_path)] if readme_path.exists() else []
    notes += [render_note(p) for p in note_paths]

    lessons = sorted(p.name for p in LESSONS_DIR.glob("*.txt"))

    nav_notes = "\n".join(
        f'<li><a href="#{nid}"><span class="num">{"★" if nid == "notes-guide" else nid[5:7]}</span>{html.escape(clean_title(title))}</a></li>'
        for nid, title, _ in notes
    )

    pipeline_cards = []
    for i, (fname, stage, role) in enumerate(PIPELINE_FILES):
        src_path = PIPELINE_DIR / fname
        source = html.escape(src_path.read_text(encoding="utf-8")) if src_path.exists() else "(file not found)"
        pipeline_cards.append(f"""
        <article class="file-card">
          <div class="file-head"><span class="step">{i}</span><div><h3><code>src/pipeline/{fname}</code></h3><p class="stage">{stage}</p></div></div>
          <p>{role}</p>
          <details><summary>View source</summary><pre><code>{source}</code></pre></details>
        </article>""")

    issues = "\n".join(f"<li><strong>{t}.</strong> {d}</li>" for t, d in KNOWN_ISSUES)
    lesson_items = "\n".join(f"<li><code>{html.escape(n)}</code></li>" for n in lessons)

    note_sections = "\n".join(
        f'<section class="note" id="{nid}"><div class="note-body">{body}</div>'
        f'<p class="back"><a href="#top">↑ Back to top</a></p></section>'
        for nid, _, body in notes
    )

    page = TEMPLATE.format(
        nav_notes=nav_notes,
        pipeline_cards="\n".join(pipeline_cards),
        issues=issues,
        lesson_items=lesson_items,
        lesson_count=len(lessons),
        note_count=len(note_paths),
        note_sections=note_sections,
    )
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(page, encoding="utf-8")
    print(f"Wrote {OUT_FILE.relative_to(ROOT)} ({len(notes)} notes, {len(PIPELINE_FILES)} pipeline files)")


TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>RAG from Scratch</title>
<meta name="description" content="A from-scratch Retrieval-Augmented Generation pipeline with LangChain, ChromaDB and Gemini, plus study notes covering every stage.">
<style>
:root {{
  --bg: #f7f6f2; --surface: #ffffff; --surface-2: #efede6; --text: #1d1d1b; --muted: #5f5e58;
  --border: #dcd9cf; --accent: #2f5d50; --accent-soft: #e2ece8; --code-bg: #f1efe8; --code-text: #2a2a27;
  --sidebar-w: 280px;
}}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{
    --bg: #161714; --surface: #1f201d; --surface-2: #2a2b27; --text: #e8e6df; --muted: #a3a197;
    --border: #383935; --accent: #8cc4b0; --accent-soft: #233630; --code-bg: #262723; --code-text: #e2e0d8;
  }}
}}
:root[data-theme="dark"] {{
  --bg: #161714; --surface: #1f201d; --surface-2: #2a2b27; --text: #e8e6df; --muted: #a3a197;
  --border: #383935; --accent: #8cc4b0; --accent-soft: #233630; --code-bg: #262723; --code-text: #e2e0d8;
}}
* {{ box-sizing: border-box; }}
html {{ scroll-behavior: smooth; scroll-padding-top: 16px; }}
body {{ margin: 0; background: var(--bg); color: var(--text);
  font: 16px/1.65 system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; }}
a {{ color: var(--accent); }}
code, pre {{ font-family: ui-monospace, "Cascadia Code", Consolas, monospace; font-size: 0.88em; }}
:not(pre) > code {{ background: var(--code-bg); padding: 0.1em 0.35em; border-radius: 4px; }}
pre {{ background: var(--code-bg); color: var(--code-text); padding: 14px 16px; border-radius: 8px;
  overflow-x: auto; line-height: 1.5; border: 1px solid var(--border); }}

.layout {{ display: flex; min-height: 100vh; }}
.sidebar {{ width: var(--sidebar-w); flex-shrink: 0; position: sticky; top: 0; height: 100vh; overflow-y: auto;
  background: var(--surface); border-right: 1px solid var(--border); padding: 24px 16px; }}
.brand {{ font-weight: 700; font-size: 1.1rem; margin: 0 8px 4px; }}
.brand-sub {{ color: var(--muted); font-size: 0.85rem; margin: 0 8px 20px; }}
.sidebar h2 {{ font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.08em; color: var(--muted);
  margin: 20px 8px 6px; }}
.sidebar ul {{ list-style: none; margin: 0; padding: 0; }}
.sidebar a {{ display: flex; gap: 10px; align-items: baseline; padding: 6px 8px; border-radius: 6px;
  color: var(--text); text-decoration: none; font-size: 0.92rem; }}
.sidebar a:hover {{ background: var(--surface-2); }}
.sidebar a.active {{ background: var(--accent-soft); color: var(--accent); font-weight: 600; }}
.num {{ color: var(--muted); font-variant-numeric: tabular-nums; font-size: 0.8rem; min-width: 1.4em; }}
.theme-toggle {{ margin: 20px 8px 0; background: var(--surface-2); color: var(--text); border: 1px solid var(--border);
  border-radius: 6px; padding: 6px 10px; cursor: pointer; font: inherit; font-size: 0.85rem; }}

main {{ flex: 1; min-width: 0; padding: 40px 48px 80px; max-width: 980px; }}
.hero h1 {{ font-size: 2.2rem; line-height: 1.2; margin: 0 0 12px; }}
.hero p {{ color: var(--muted); font-size: 1.08rem; max-width: 720px; }}
.stats {{ display: flex; flex-wrap: wrap; gap: 12px; margin: 24px 0 8px; }}
.stat {{ background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 10px 16px; }}
.stat b {{ display: block; font-size: 1.3rem; }}
.stat span {{ color: var(--muted); font-size: 0.85rem; }}
section {{ margin-top: 56px; }}
section > h2 {{ font-size: 1.5rem; border-bottom: 1px solid var(--border); padding-bottom: 8px; }}

.flow {{ display: flex; flex-wrap: wrap; align-items: center; gap: 8px; margin: 16px 0; }}
.flow .box {{ background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 8px 12px;
  font-size: 0.9rem; }}
.flow .box small {{ display: block; color: var(--muted); font-size: 0.75rem; }}
.flow .arrow {{ color: var(--muted); }}

.file-card {{ background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 16px 20px;
  margin: 14px 0; }}
.file-head {{ display: flex; gap: 14px; align-items: center; }}
.file-head h3 {{ margin: 0; font-size: 1rem; }}
.stage {{ margin: 2px 0 0; color: var(--muted); font-size: 0.85rem; }}
.step {{ display: grid; place-items: center; width: 30px; height: 30px; border-radius: 50%; flex-shrink: 0;
  background: var(--accent-soft); color: var(--accent); font-weight: 700; }}
details summary {{ cursor: pointer; color: var(--accent); font-size: 0.9rem; }}
details pre {{ max-height: 480px; }}

.cols {{ display: grid; grid-template-columns: 1fr 1fr; gap: 24px; }}
.panel {{ background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 4px 20px 12px; }}

.note {{ background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 8px 32px 20px; }}
.note-body h1 {{ font-size: 1.7rem; line-height: 1.25; margin-top: 24px; }}
.note-body h2 {{ font-size: 1.3rem; margin-top: 32px; border-bottom: 1px solid var(--border); padding-bottom: 6px; }}
.note-body h3 {{ font-size: 1.1rem; margin-top: 24px; }}
.note-body blockquote {{ margin: 16px 0; padding: 8px 16px; border-left: 4px solid var(--accent);
  background: var(--accent-soft); border-radius: 0 8px 8px 0; }}
.note-body blockquote p {{ margin: 6px 0; }}
.note-body hr {{ border: 0; border-top: 1px solid var(--border); margin: 28px 0; }}
.note-body img {{ max-width: 100%; }}
.table-wrap {{ overflow-x: auto; margin: 16px 0; }}
table {{ border-collapse: collapse; width: 100%; font-size: 0.92rem; }}
th, td {{ border: 1px solid var(--border); padding: 8px 10px; text-align: left; vertical-align: top; }}
th {{ background: var(--surface-2); }}
.back {{ text-align: right; font-size: 0.85rem; margin: 20px 0 0; }}
footer {{ margin-top: 64px; color: var(--muted); font-size: 0.85rem; }}

.menu-btn {{ display: none; }}
@media (max-width: 860px) {{
  .layout {{ display: block; }}
  .sidebar {{ position: static; width: auto; height: auto; border-right: 0; border-bottom: 1px solid var(--border);
    padding: 12px 16px; }}
  .sidebar nav {{ display: none; }}
  .sidebar.open nav {{ display: block; }}
  .menu-btn {{ display: inline-block; float: right; background: var(--surface-2); color: var(--text);
    border: 1px solid var(--border); border-radius: 6px; padding: 4px 10px; font: inherit; }}
  .brand-sub {{ margin-bottom: 4px; }}
  main {{ padding: 24px 16px 60px; }}
  .hero h1 {{ font-size: 1.7rem; }}
  .cols {{ grid-template-columns: 1fr; }}
  .note {{ padding: 4px 16px 16px; }}
}}
</style>
</head>
<body id="top">
<div class="layout">
  <aside class="sidebar" id="sidebar">
    <button class="menu-btn" id="menu-btn" aria-expanded="false" aria-controls="sidebar">Menu</button>
    <p class="brand">RAG from Scratch</p>
    <p class="brand-sub">Pipeline + study notes</p>
    <nav>
      <h2>The repo</h2>
      <ul>
        <li><a href="#overview">Overview</a></li>
        <li><a href="#architecture">Architecture</a></li>
        <li><a href="#pipeline">Pipeline files</a></li>
        <li><a href="#run">Setup &amp; usage</a></li>
        <li><a href="#corpus">Corpus &amp; known issues</a></li>
      </ul>
      <h2>Study notes</h2>
      <ul>
{nav_notes}
      </ul>
      <button class="theme-toggle" id="theme-toggle" type="button">Toggle theme</button>
    </nav>
  </aside>

  <main>
    <section class="hero" id="overview" style="margin-top:0">
      <h1>RAG from Scratch</h1>
      <p>A Retrieval-Augmented Generation pipeline built stage by stage with LangChain, ChromaDB and Google Gemini —
      implemented by hand to learn how each part works, not wired together from one high-level helper.
      The corpus it searches is a set of short lessons about RAG itself, so the pipeline can answer questions about its own subject.</p>
      <div class="stats">
        <div class="stat"><b>6</b><span>pipeline scripts</span></div>
        <div class="stat"><b>{lesson_count}</b><span>corpus lessons</span></div>
        <div class="stat"><b>{note_count}</b><span>study notes</span></div>
        <div class="stat"><b>2</b><span>retrieval strategies</span></div>
      </div>
      <p>What it demonstrates:</p>
      <ul>
        <li><strong>Ingestion with provenance</strong> — every chunk carries its source filename.</li>
        <li><strong>Two retrievers to compare</strong> — pure vector search vs. hybrid BM25 + vector.</li>
        <li><strong>Query expansion</strong> — <code>MultiQueryRetriever</code> rephrases the question to close the wording gap.</li>
        <li><strong>Cited, grounded answers</strong> — the model must cite filenames or say it doesn’t know.</li>
        <li><strong>Separate evaluation</strong> — retrieval accuracy and generation accuracy scored independently.</li>
      </ul>
    </section>

    <section id="architecture">
      <h2>Architecture</h2>
      <p>Data flows left to right. Ingestion runs once to build the vector store; every question then goes through retrieval and generation.</p>
      <div class="flow" aria-label="Pipeline flow">
        <div class="box">data/raw/*.txt<small>lesson files</small></div><span class="arrow">→</span>
        <div class="box">ingest.py<small>chunk + tag filename</small></div><span class="arrow">→</span>
        <div class="box">Chroma<small>vector store</small></div><span class="arrow">→</span>
        <div class="box">hybrid_search.py<small>BM25 + multi-query vector</small></div><span class="arrow">→</span>
        <div class="box">generation.py<small>[Source] prompt → Gemini</small></div><span class="arrow">→</span>
        <div class="box">Cited answer</div>
      </div>
      <p><code>evaluation.py</code> sits beside the flow: it scores what hybrid search retrieved and what Gemini answered, separately.</p>
    </section>

    <section id="pipeline">
      <h2>Pipeline files</h2>
      <p>Read these in order alongside the matching lesson in <code>data/raw/</code>. Expand a card to read its source.</p>
{pipeline_cards}
    </section>

    <section id="run">
      <h2>Setup &amp; usage</h2>
<pre><code>python -m venv .venv
.venv\\Scripts\\activate          # Windows
pip install -r requirements.txt
copy .env.example .env          # then paste your GOOGLE_API_KEY

cd src/pipeline
python ingest.py                # build the vector store
python hybrid_search.py         # try retrieval
python generation.py            # ask a question, get a cited answer
python evaluation.py            # retrieval vs. generation scorecard</code></pre>
      <p>Settings such as <code>GENERATION_MODEL</code>, <code>CHUNK_SIZE</code> and <code>RETRIEVAL_K</code> live in <code>.env</code>; defaults are in <code>config.py</code>.</p>
    </section>

    <section id="corpus">
      <h2>Corpus &amp; known issues</h2>
      <div class="cols">
        <div class="panel"><h3>Lessons in <code>data/raw/</code></h3><ul>
{lesson_items}
        </ul></div>
        <div class="panel"><h3>Known issues</h3><ul>
{issues}
        </ul></div>
      </div>
    </section>

    <section id="notes">
      <h2>Study notes</h2>
      <p>The full contents of <code>notes/</code>, from “what is RAG?” to production evaluation and ingestion. Start with the guide, then read 00 → 10.</p>
    </section>
{note_sections}

    <footer>Generated from <code>notes/</code> and <code>src/pipeline/</code> by <code>scripts/build_site.py</code>.</footer>
  </main>
</div>
<script>
(function () {{
  var root = document.documentElement;
  try {{ var saved = localStorage.getItem("theme"); if (saved) root.setAttribute("data-theme", saved); }} catch (e) {{}}
  document.getElementById("theme-toggle").addEventListener("click", function () {{
    var current = root.getAttribute("data-theme") ||
      (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
    var next = current === "dark" ? "light" : "dark";
    root.setAttribute("data-theme", next);
    try {{ localStorage.setItem("theme", next); }} catch (e) {{}}
  }});

  var sidebar = document.getElementById("sidebar"), btn = document.getElementById("menu-btn");
  btn.addEventListener("click", function () {{
    var open = sidebar.classList.toggle("open");
    btn.setAttribute("aria-expanded", open);
  }});
  sidebar.addEventListener("click", function (e) {{
    if (e.target.closest("nav a")) {{ sidebar.classList.remove("open"); btn.setAttribute("aria-expanded", false); }}
  }});

  // Highlight the sidebar link for the section currently in view.
  var links = Array.prototype.slice.call(document.querySelectorAll(".sidebar nav a"));
  var targets = links.map(function (a) {{ return document.querySelector(a.getAttribute("href")); }});
  var observer = new IntersectionObserver(function (entries) {{
    entries.forEach(function (entry) {{
      if (!entry.isIntersecting) return;
      var i = targets.indexOf(entry.target);
      links.forEach(function (l, j) {{ l.classList.toggle("active", j === i); }});
    }});
  }}, {{ rootMargin: "0px 0px -70% 0px" }});
  targets.forEach(function (t) {{ if (t) observer.observe(t); }});
}})();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    build()
