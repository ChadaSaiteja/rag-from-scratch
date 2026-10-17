"""Build the GitHub Pages site in docs/ from notes/, data/raw/ and src/pipeline/.

The site is multi-page so every topic gets its own URL, <title>, description and
structured data, which is what search engines rank:

  /                     landing page + learning path
  /learn/               how to use the notes (notes/README.md)
  /learn/<module>/      one page per study note
  /learn/glossary/, /learn/faq/  reference pages (FAQ gets FAQPage structured data)
  /implementation/      how this repo implements the pipeline, with source
  /lessons/             the plain-text corpus the pipeline retrieves from
  /lessons/<lesson>/
  sitemap.xml, robots.txt, 404.html, .nojekyll, assets/

Run from anywhere:  python scripts/build_site.py
Preview locally:    python -m http.server -d docs   (then open http://localhost:8000)
Re-run whenever notes/, data/raw/ or src/pipeline/ change; docs/ is fully regenerated.
"""
import html
import json
import re
import shutil
import subprocess
from datetime import date
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parents[1]
NOTES_DIR = ROOT / "notes"
LESSONS_DIR = ROOT / "data" / "raw"
PIPELINE_DIR = ROOT / "src" / "pipeline"
OUT_DIR = ROOT / "docs"

SITE_URL = "https://chadasaiteja.github.io/rag-from-scratch/"
REPO_URL = "https://github.com/ChadaSaiteja/rag-from-scratch"
SITE_NAME = "RAG from Scratch"
AUTHOR = {"@type": "Person", "name": "ChadaSaiteja", "url": "https://github.com/ChadaSaiteja"}
# Paste the content value from Google Search Console's "HTML tag" verification method.
GOOGLE_SITE_VERIFICATION = ""

HOME_TITLE = "RAG from Scratch — Learn Retrieval-Augmented Generation Step by Step"
HOME_DESCRIPTION = (
    "Learn RAG (Retrieval-Augmented Generation) from scratch: a free beginner course on embeddings, "
    "chunking, vector databases, retrieval and evaluation, with Python code."
)

# Pipeline files in the order data flows through them.
PIPELINE_FILES = [
    ("config.py", "Shared settings", "Loads the API key, model names, chunk size and top-k from <code>.env</code>, so every other script is retargeted by editing <code>.env</code> only."),
    ("ingest.py", "Stages 1–4 · Load → chunk → embed → store", "Reads every lesson in <code>data/raw/</code>, splits it into ~1000-character chunks, tags each chunk with its source <code>filename</code>, embeds it with Gemini and persists it to Chroma."),
    ("vector_retrieval.py", "Stage 5a · Semantic retrieval", "The simplest retriever: top-k nearest neighbours by embedding similarity. A baseline to compare hybrid search against."),
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


# ---------------------------------------------------------------- helpers

def slug_for(stem):
    return re.sub(r"[_\s]+", "-", stem.lower())


def clean_title(title):
    """Drop leading emoji, "Module N:" and trailing "(…)" for short labels."""
    title = re.sub(r"^[^\w(]+", "", title).strip()
    title = re.sub(r"^Module\s+\d+:\s*", "", title)
    return re.sub(r"\s*\(.*\)$", "", title)


SMALL_WORDS = {"a", "an", "and", "as", "at", "by", "for", "in", "of", "on", "or", "the", "to", "vs"}
ACRONYMS = {"rag": "RAG", "llm": "LLM", "api": "API", "bm25": "BM25"}


def title_case(text):
    words = text.lower().split()
    out = []
    for i, w in enumerate(words):
        if w in ACRONYMS:
            out.append(ACRONYMS[w])
        elif i and w in SMALL_WORDS:
            out.append(w)
        else:
            out.append(w[:1].upper() + w[1:])
    return " ".join(out)


def strip_markdown(text):
    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"[*_>#]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def truncate(text, limit=158):
    if len(text) <= limit:
        return text
    cut = text[:limit].rsplit(" ", 1)[0].rstrip(",;:—-")
    return cut + "…"


def first_paragraph(md_text):
    """First prose paragraph (skipping headings, quotes, lists, code) as plain text."""
    in_code = False
    for block in re.split(r"\n\s*\n", md_text):
        stripped = block.strip()
        if stripped.startswith("```"):
            in_code = not in_code if stripped.count("```") % 2 else in_code
            continue
        if in_code or not stripped or stripped[0] in "#>|-*!`" or stripped.startswith("---"):
            continue
        return strip_markdown(stripped)
    return ""


def git_lastmod(path):
    try:
        out = subprocess.run(["git", "log", "-1", "--format=%cs", "--", str(path)],
                             cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
        return out or date.today().isoformat()
    except (OSError, subprocess.CalledProcessError):
        return date.today().isoformat()


def ld_json(data):
    return f'<script type="application/ld+json">{json.dumps(data, ensure_ascii=False)}</script>'


def breadcrumb_ld(crumbs):
    return {
        "@context": "https://schema.org", "@type": "BreadcrumbList",
        "itemListElement": [
            {"@type": "ListItem", "position": i + 1, "name": name, "item": SITE_URL + path}
            for i, (name, path) in enumerate(crumbs)
        ],
    }


# ---------------------------------------------------------------- content model

def load_notes():
    """Return (guide, modules): the notes README and each numbered note, in order."""
    def load(path, url_path):
        text = path.read_text(encoding="utf-8")
        m = re.search(r"^#\s+(.+)$", text, flags=re.MULTILINE)
        h1 = m.group(1).strip() if m else path.stem
        return {"path": path, "stem": path.stem, "text": text, "h1": h1, "title": clean_title(h1),
                "url": url_path, "description": truncate(first_paragraph(text))}

    modules, extras = [], []
    for p in sorted(NOTES_DIR.glob("*.md")):
        if p.stem.upper() == "README":
            continue
        n = load(p, f"learn/{slug_for(p.stem)}/")
        if p.stem[:2].isdigit():
            n["num"] = p.stem[:2]
            modules.append(n)
        else:
            n["num"] = {"glossary": "📖", "faq": "❓"}.get(p.stem, "•")
            extras.append(n)
    readme = NOTES_DIR / "README.md"
    guide = load(readme, "learn/") if readme.exists() else None
    return guide, modules, extras


def load_lessons():
    lessons = []
    for p in sorted(LESSONS_DIR.glob("*.txt")):
        text = p.read_text(encoding="utf-8")
        first = text.strip().splitlines()[0].strip()
        body = text.strip().split("\n", 1)[1] if "\n" in text.strip() else ""
        lessons.append({"path": p, "stem": p.stem, "num": p.stem[:2], "text": text,
                        "title": title_case(first),
                        "url": f"lessons/{slug_for(p.stem)}/",
                        "description": truncate(re.sub(r"\s+", " ", body).strip())})
    return lessons


LIST_ITEM = re.compile(r"^\s*(?:[-*+]|\d+\.)\s")


def space_lists(text):
    """Insert a blank line before a list that directly follows a paragraph line.

    GitHub renders such lists, but Python-Markdown needs the blank line, so the notes
    would otherwise show "- item" as plain text on the site.
    """
    out, in_fence = [], False
    for line in text.splitlines():
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
        elif not in_fence and LIST_ITEM.match(line) and out:
            prev = out[-1]
            is_paragraph = (prev.strip() and not LIST_ITEM.match(prev)
                            and not prev.lstrip().startswith(("|", ">", "<"))
                            and not prev.startswith((" ", "\t")))
            if is_paragraph:
                out.append("")
        out.append(line)
    return "\n".join(out) + "\n"


def render_markdown(note, url_for_stem, rel):
    def slugify(value, separator):
        slug = re.sub(r"[^\w\s-]", "", value, flags=re.UNICODE).strip().lower()
        return re.sub(r"[\s_-]+", separator, slug)

    md = markdown.Markdown(extensions=["tables", "fenced_code", "sane_lists", "toc", "md_in_html"],
                           extension_configs={"toc": {"slugify": slugify}})
    body = md.convert(space_lists(note["text"]))

    def repl(m):
        target = url_for_stem.get(m.group(1))
        return f'href="{rel}{target}{m.group(2) or ""}"' if target is not None else m.group(0)
    body = re.sub(r'href="(?:\./)?([\w-]+)\.md(#[^"]*)?"', repl, body)
    # Links that leave notes/ (../README.md, ../src/...) point at the repository on GitHub.
    body = re.sub(r'href="\.\./([^"]+)"', lambda m: f'href="{REPO_URL}/blob/master/{m.group(1)}"', body)
    body = body.replace("<table>", '<div class="table-wrap"><table>').replace("</table>", "</table></div>")
    return body


def render_lesson_text(text):
    """Plain-text lesson -> simple HTML: short standalone lines become headings, '- ' runs become lists."""
    blocks = [b.strip("\n") for b in re.split(r"\n\s*\n", text.strip())][1:]  # first line is the title
    out = []
    for block in blocks:
        lines = [l.rstrip() for l in block.splitlines() if l.strip()]
        if all(l.lstrip().startswith(("- ", "* ")) for l in lines):
            items = "".join(f"<li>{html.escape(l.lstrip()[2:])}</li>" for l in lines)
            out.append(f"<ul>{items}</ul>")
            continue
        if re.match(r"^\d+\.\s", lines[0]) and all(re.match(r"^\d+\.\s", l) for l in lines):
            items = "".join(f"<li>{html.escape(re.sub(r'^\d+\.\s', '', l))}</li>" for l in lines)
            out.append(f"<ol>{items}</ol>")
            continue
        first = lines[0]
        if len(lines) > 1 and len(first) < 60 and not first.endswith((".", ",")):
            level = "h3" if first.endswith(":") else "h2"  # "Small chunks:" labels a sub-point
            out.append(f"<{level}>{html.escape(first.rstrip(':'))}</{level}>")
            lines = lines[1:]
        elif len(lines) == 1 and len(first) < 50 and not first.endswith((".", ",", ":")):
            out.append(f"<h2>{html.escape(first)}</h2>")
            continue
        if all(l.lstrip().startswith(("- ", "* ")) for l in lines):
            out.append("<ul>" + "".join(f"<li>{html.escape(l.lstrip()[2:])}</li>" for l in lines) + "</ul>")
        else:
            out.append("<p>" + "<br>".join(html.escape(l) for l in lines) + "</p>")
    return "\n".join(out)


# ---------------------------------------------------------------- page shell

def page(*, path, title, description, body, rel, og_type="website", ld=(), active="", sidebar=""):
    canonical = SITE_URL + path
    verify = (f'<meta name="google-site-verification" content="{html.escape(GOOGLE_SITE_VERIFICATION)}">\n'
              if GOOGLE_SITE_VERIFICATION else "")
    nav = [("", "Home"), ("learn/", "Learn"), ("implementation/", "Implementation"), ("lessons/", "Lessons"),
           ("learn/faq/", "FAQ")]
    current = ' aria-current="page"'
    nav_html = "".join(
        f'<a href="{rel}{href}"{current if active == href else ""}>{label}</a>' for href, label in nav
    )
    lds = "\n".join(ld_json(x) for x in ld)
    layout_cls = "with-sidebar" if sidebar else "no-sidebar"
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<meta name="description" content="{html.escape(description)}">
<link rel="canonical" href="{canonical}">
<meta name="robots" content="index, follow">
{verify}<meta property="og:site_name" content="{SITE_NAME}">
<meta property="og:type" content="{og_type}">
<meta property="og:title" content="{html.escape(title)}">
<meta property="og:description" content="{html.escape(description)}">
<meta property="og:url" content="{canonical}">
<meta name="twitter:card" content="summary">
<meta name="twitter:title" content="{html.escape(title)}">
<meta name="twitter:description" content="{html.escape(description)}">
<link rel="icon" href="{rel}assets/favicon.svg" type="image/svg+xml">
<link rel="stylesheet" href="{rel}assets/style.css">
<script>try{{var t=localStorage.getItem("theme");if(t)document.documentElement.setAttribute("data-theme",t)}}catch(e){{}}</script>
{lds}
</head>
<body>
<a class="skip" href="#content">Skip to content</a>
<header class="topbar">
  <a class="brand" href="{rel}">RAG <span>from Scratch</span></a>
  <nav class="topnav" aria-label="Main">{nav_html}<a href="{REPO_URL}" rel="noopener">GitHub</a></nav>
  <button class="theme-toggle" id="theme-toggle" type="button" aria-label="Toggle dark mode">◐</button>
</header>
<div class="layout {layout_cls}">
{sidebar}
<main id="content">
{body}
</main>
</div>
<footer class="site-footer">
  <p>{SITE_NAME} · free and open source on <a href="{REPO_URL}" rel="noopener">GitHub</a> · built from <code>notes/</code>, <code>data/raw/</code> and <code>src/pipeline/</code></p>
</footer>
<script src="{rel}assets/site.js" defer></script>
</body>
</html>
"""


def learn_sidebar(guide, modules, extras, rel, current_url):
    def item(url, num, label):
        cur = ' aria-current="page"' if url == current_url else ""
        return f'<li><a href="{rel}{url}"{cur}><span class="num">{num}</span>{html.escape(label)}</a></li>'
    items = [item(guide["url"], "★", "How to use these notes")] if guide else []
    items += [item(m["url"], m["num"], m["title"]) for m in modules]
    refs = "".join(item(x["url"], x["num"], x["title"]) for x in extras)
    ref_block = f'<p class="sidebar-label">Reference</p><ul>{refs}</ul>' if refs else ""
    return f"""<aside class="sidebar" aria-label="Course modules">
  <details class="sidebar-box" open>
    <summary>Course modules</summary>
    <ul>{"".join(items)}</ul>
    {ref_block}
  </details>
</aside>"""


def breadcrumbs_html(crumbs, rel):
    parts = []
    for i, (name, path) in enumerate(crumbs):
        if i == len(crumbs) - 1:
            parts.append(f'<span aria-current="page">{html.escape(name)}</span>')
        else:
            parts.append(f'<a href="{rel}{path}">{html.escape(name)}</a>')
    return f'<nav class="crumbs" aria-label="Breadcrumb">{" / ".join(parts)}</nav>'


def pager(prev, nxt, rel):
    left = (f'<a class="pager-link" href="{rel}{prev["url"]}" rel="prev"><small>Previous</small>{html.escape(prev["title"])}</a>'
            if prev else "<span></span>")
    right = (f'<a class="pager-link next" href="{rel}{nxt["url"]}" rel="next"><small>Next</small>{html.escape(nxt["title"])}</a>'
             if nxt else "<span></span>")
    return f'<nav class="pager" aria-label="Module navigation">{left}{right}</nav>'


def faq_ld(note):
    """FAQPage data from '### Question' headings and the text under each."""
    entries = []
    for block in re.split(r"^###\s+", note["text"], flags=re.MULTILINE)[1:]:
        question, _, answer = block.partition("\n")
        answer = strip_markdown(answer.split("\n#", 1)[0])
        if question.strip() and answer:
            entries.append({"@type": "Question", "name": question.strip(),
                            "acceptedAnswer": {"@type": "Answer", "text": answer}})
    return {"@context": "https://schema.org", "@type": "FAQPage", "mainEntity": entries}


def glossary_ld(note):
    terms = []
    for block in re.split(r"^###\s+", note["text"], flags=re.MULTILINE)[1:]:
        term, _, definition = block.partition("\n")
        definition = strip_markdown(definition.split("→")[0])
        terms.append({"@type": "DefinedTerm", "name": term.strip(), "description": definition,
                      "inDefinedTermSet": SITE_URL + note["url"]})
    return {"@context": "https://schema.org", "@type": "DefinedTermSet", "name": "RAG Glossary",
            "url": SITE_URL + note["url"], "hasDefinedTerm": terms}


def course_ld(modules):
    return {
        "@context": "https://schema.org", "@type": "Course",
        "name": SITE_NAME, "description": HOME_DESCRIPTION, "url": SITE_URL,
        "provider": {"@type": "Organization", "name": SITE_NAME, "sameAs": REPO_URL},
        "author": AUTHOR, "inLanguage": "en", "educationalLevel": "Beginner",
        "isAccessibleForFree": True,
        "teaches": ["Retrieval-Augmented Generation", "Embeddings", "Chunking", "Vector databases",
                    "Hybrid search", "Prompt engineering", "RAG evaluation"],
        "hasCourseInstance": {"@type": "CourseInstance", "courseMode": "Online",
                              "courseWorkload": "PT8H"},
        "offers": {"@type": "Offer", "category": "Free", "price": 0, "priceCurrency": "USD"},
        "hasPart": [{"@type": "LearningResource", "name": m["title"], "url": SITE_URL + m["url"]} for m in modules],
    }


# ---------------------------------------------------------------- pages

def build_home(guide, modules, extras, lessons):
    rel = ""
    cards = "".join(
        f'<a class="module-card" href="{m["url"]}"><span class="num">{m["num"]}</span>'
        f'<h3>{html.escape(m["title"])}</h3><p>{html.escape(m["description"])}</p></a>'
        for m in modules
    )
    ref_cards = "".join(
        f'<a class="module-card" href="{x["url"]}"><span class="num">{x["num"]}</span>'
        f'<h3>{html.escape(x["title"])}</h3><p>{html.escape(x["description"])}</p></a>'
        for x in extras
    )
    body = f"""
<section class="hero">
  <p class="eyebrow">Free course · Python · open source</p>
  <h1>RAG from Scratch</h1>
  <p class="lede">Learn Retrieval-Augmented Generation step by step — what it is, why it works, and how to build every stage yourself:
  ingestion, chunking, embeddings, vector databases, hybrid retrieval, prompting and evaluation.</p>
  <div class="cta">
    <a class="btn primary" href="{modules[0]['url'] if modules else 'learn/'}">Start module 00</a>
    <a class="btn" href="implementation/">See the working pipeline</a>
  </div>
  <div class="stats">
    <div class="stat"><b>{len(modules)}</b><span>study modules</span></div>
    <div class="stat"><b>{len(lessons)}</b><span>corpus lessons</span></div>
    <div class="stat"><b>6</b><span>pipeline scripts</span></div>
    <div class="stat"><b>Free</b><span>and open source</span></div>
  </div>
</section>

<section>
  <h2>What is RAG?</h2>
  <p><strong>Retrieval-Augmented Generation (RAG)</strong> gives a large language model an open book. Before the model answers, a retriever
  searches your own documents for the passages most relevant to the question and puts them in the prompt. The model then answers from
  that evidence instead of guessing — which means fresher knowledge, fewer hallucinations and answers you can trace back to a source.</p>
  <ol class="pipeline-steps">
    <li><b>Ingest</b> documents</li><li><b>Chunk</b> them</li><li><b>Embed</b> each chunk</li>
    <li><b>Store</b> vectors</li><li><b>Retrieve</b> the best matches</li><li><b>Generate</b> a cited answer</li>
  </ol>
</section>

<section>
  <h2>The learning path</h2>
  <p>Each module owns one stage of the pipeline and builds on the one before. Read them in order; budget 30–45 minutes each.
  New here? Read <a href="learn/">how to use these notes</a> first.</p>
  <div class="module-grid">{cards}</div>
</section>

<section>
  <h2>Quick reference</h2>
  <div class="module-grid">{ref_cards}</div>
</section>

<section class="split">
  <div>
    <h2>Then see it implemented</h2>
    <p>The <a href="implementation/">implementation guide</a> walks through a real pipeline built with LangChain, ChromaDB and Google Gemini:
    hybrid BM25 + vector search, multi-query expansion, citation-grounded answers and separate retrieval/generation evaluation —
    with the full source of every script.</p>
  </div>
  <div>
    <h2>And the corpus it searches</h2>
    <p>The pipeline answers questions over <a href="lessons/">{len(lessons)} short lessons about RAG itself</a>, so you can ask it about
    the very concepts you are learning.</p>
  </div>
</section>
"""
    ld = [
        {"@context": "https://schema.org", "@type": "WebSite", "name": SITE_NAME, "url": SITE_URL,
         "description": HOME_DESCRIPTION, "inLanguage": "en"},
        course_ld(modules),
    ]
    return page(path="", title=HOME_TITLE, description=HOME_DESCRIPTION, body=body, rel=rel, ld=ld)


def build_note_page(note, guide, modules, extras, url_for_stem, prev, nxt, is_guide=False):
    depth = note["url"].count("/")
    rel = "../" * depth
    crumbs = [("Home", ""), ("Learn", "learn/")]
    if not is_guide:
        crumbs.append((note["title"], note["url"]))
    content = render_markdown(note, url_for_stem, rel)
    body = f"""{breadcrumbs_html(crumbs, rel)}
<article class="prose">
{content}
</article>
{pager(prev, nxt, rel)}
<p class="edit"><a href="{REPO_URL}/blob/master/notes/{note['path'].name}" rel="noopener">View this page’s source on GitHub</a></p>
"""
    title = (f"Learn RAG from Scratch — Course Guide | {SITE_NAME}" if is_guide
             else f"{note['title']} | {SITE_NAME}")
    description = (truncate("How to work through the RAG from Scratch course: reading order, study loop, "
                            "paths for beginners and practitioners, and the full module index.")
                   if is_guide else note["description"])
    if note["stem"] == "faq":
        title = f"RAG FAQ: Common Questions Answered | {SITE_NAME}"
    elif note["stem"] == "glossary":
        title = f"RAG Glossary: Key Terms Explained Simply | {SITE_NAME}"
    article_ld = {
        "@context": "https://schema.org", "@type": "LearningResource" if is_guide else "TechArticle",
        "headline": title.split(" | ")[0], "name": title.split(" | ")[0], "description": description,
        "url": SITE_URL + note["url"], "inLanguage": "en", "author": AUTHOR,
        "dateModified": git_lastmod(note["path"]), "educationalLevel": "Beginner",
        "isPartOf": {"@type": "Course", "name": SITE_NAME, "url": SITE_URL},
    }
    lds = [article_ld, breadcrumb_ld(crumbs)]
    if note["stem"] == "faq":
        lds.append(faq_ld(note))
    elif note["stem"] == "glossary":
        lds.append(glossary_ld(note))
    return page(path=note["url"], title=title, description=description, body=body, rel=rel,
                og_type="article", ld=lds, active="learn/faq/" if note["stem"] == "faq" else "learn/",
                sidebar=learn_sidebar(guide, modules, extras, rel, note["url"]))


def build_implementation():
    path = "implementation/"
    rel = "../"
    cards = []
    for i, (fname, stage, role) in enumerate(PIPELINE_FILES):
        src = PIPELINE_DIR / fname
        source = html.escape(src.read_text(encoding="utf-8")) if src.exists() else "(file not found)"
        cards.append(f"""
<article class="file-card" id="{slug_for(fname.replace('.py', ''))}">
  <div class="file-head"><span class="step">{i}</span><div><h3><code>src/pipeline/{fname}</code></h3><p class="stage">{stage}</p></div></div>
  <p>{role}</p>
  <details><summary>View source</summary><pre><code class="language-python">{source}</code></pre></details>
</article>""")
    issues = "".join(f"<li><strong>{t}.</strong> {d}</li>" for t, d in KNOWN_ISSUES)
    crumbs = [("Home", ""), ("Implementation", path)]
    body = f"""{breadcrumbs_html(crumbs, rel)}
<article class="prose wide">
<h1>How this RAG pipeline is implemented</h1>
<p class="lede">A Retrieval-Augmented Generation pipeline built stage by stage with LangChain, ChromaDB and Google Gemini —
implemented by hand to learn how each part works, not wired together from one high-level helper.</p>

<h2>What it demonstrates</h2>
<ul>
  <li><strong>Ingestion with provenance</strong> — every chunk carries its source filename.</li>
  <li><strong>Two retrievers to compare</strong> — pure vector search vs. hybrid BM25 + vector.</li>
  <li><strong>Query expansion</strong> — <code>MultiQueryRetriever</code> rephrases the question to close the wording gap.</li>
  <li><strong>Cited, grounded answers</strong> — the model must cite filenames or say it doesn’t know.</li>
  <li><strong>Separate evaluation</strong> — retrieval accuracy and generation accuracy scored independently.</li>
</ul>

<h2>Architecture</h2>
<p>Ingestion runs once to build the vector store; every question then flows through retrieval and generation.</p>
<div class="flow" role="img" aria-label="data/raw lessons to ingest.py to Chroma to hybrid_search.py to generation.py to a cited answer">
  <div class="box">data/raw/*.txt<small>lesson files</small></div><span class="arrow">→</span>
  <div class="box">ingest.py<small>chunk + tag filename</small></div><span class="arrow">→</span>
  <div class="box">Chroma<small>vector store</small></div><span class="arrow">→</span>
  <div class="box">hybrid_search.py<small>BM25 + multi-query vector</small></div><span class="arrow">→</span>
  <div class="box">generation.py<small>[Source] prompt → Gemini</small></div><span class="arrow">→</span>
  <div class="box">Cited answer</div>
</div>
<p><code>evaluation.py</code> sits beside the flow and scores what hybrid search retrieved and what Gemini answered, separately.</p>

<h2>Pipeline files</h2>
<p>Read them in order alongside the matching <a href="{rel}lessons/">lesson</a>. Expand a card to read its source.</p>
{"".join(cards)}

<h2 id="setup">Setup and usage</h2>
<pre><code>git clone {REPO_URL}.git
cd rag-from-scratch
python -m venv .venv
.venv\\Scripts\\activate          # Windows (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
copy .env.example .env          # then paste your GOOGLE_API_KEY

cd src/pipeline
python ingest.py                # build the vector store
python hybrid_search.py         # try retrieval
python generation.py            # ask a question, get a cited answer
python evaluation.py            # retrieval vs. generation scorecard</code></pre>
<p>Settings such as <code>GENERATION_MODEL</code>, <code>CHUNK_SIZE</code> and <code>RETRIEVAL_K</code> live in <code>.env</code>; defaults are in <code>config.py</code>.</p>

<h2 id="known-issues">Known issues</h2>
<p>Real limitations of the current code, kept visible as next exercises:</p>
<ul>{issues}</ul>
</article>
"""
    title = f"Build a RAG Pipeline in Python: LangChain + Chroma | {SITE_NAME}"
    description = truncate("Step-by-step walkthrough of a from-scratch RAG pipeline in Python: ingestion, chunking, "
                           "Chroma vector store, hybrid BM25 + vector retrieval, cited Gemini answers and evaluation.")
    ld = [{
        "@context": "https://schema.org", "@type": "TechArticle", "headline": title.split(" | ")[0],
        "description": description, "url": SITE_URL + path, "author": AUTHOR, "inLanguage": "en",
        "dateModified": max(git_lastmod(PIPELINE_DIR / f) for f, _, _ in PIPELINE_FILES),
        "proficiencyLevel": "Beginner", "isPartOf": {"@type": "Course", "name": SITE_NAME, "url": SITE_URL},
    }, {
        "@context": "https://schema.org", "@type": "SoftwareSourceCode", "name": SITE_NAME,
        "codeRepository": REPO_URL, "programmingLanguage": "Python", "description": description,
    }, breadcrumb_ld(crumbs)]
    return page(path=path, title=title, description=description, body=body, rel=rel,
                og_type="article", ld=ld, active="implementation/")


def build_lessons_index(lessons):
    path, rel = "lessons/", "../"
    items = "".join(
        f'<a class="module-card" href="{rel}{l["url"]}"><span class="num">{l["num"]}</span>'
        f'<h3>{html.escape(l["title"])}</h3><p>{html.escape(l["description"])}</p></a>'
        for l in lessons
    )
    crumbs = [("Home", ""), ("Lessons", path)]
    body = f"""{breadcrumbs_html(crumbs, rel)}
<article class="prose wide">
<h1>RAG lessons: the corpus the pipeline searches</h1>
<p class="lede">{len(lessons)} short plain-text lessons that explain RAG end to end. The pipeline ingests exactly these files, so every
answer it gives cites one of them. They also make a fast, compact refresher.</p>
</article>
<div class="module-grid">{items}</div>
"""
    title = f"RAG Lessons — Short Explanations of Every Stage | {SITE_NAME}"
    description = truncate("Ten short lessons covering RAG fundamentals, data ingestion, chunking, embeddings, vector databases, "
                           "retrieval, prompting, evaluation and metadata filtering.")
    ld = [{"@context": "https://schema.org", "@type": "CollectionPage", "name": title.split(" | ")[0],
           "description": description, "url": SITE_URL + path}, breadcrumb_ld(crumbs)]
    return page(path=path, title=title, description=description, body=body, rel=rel, ld=ld, active="lessons/")


def build_lesson(lesson, prev, nxt):
    rel = "../../"
    crumbs = [("Home", ""), ("Lessons", "lessons/"), (lesson["title"], lesson["url"])]
    body = f"""{breadcrumbs_html(crumbs, rel)}
<article class="prose">
<p class="eyebrow">Lesson {lesson['num']}</p>
<h1>{html.escape(lesson['title'])}</h1>
{render_lesson_text(lesson['text'])}
</article>
{pager(prev, nxt, rel)}
<p class="edit"><a href="{REPO_URL}/blob/master/data/raw/{lesson['path'].name}" rel="noopener">View the raw lesson file on GitHub</a></p>
"""
    title = f"{lesson['title']} — RAG Lesson {lesson['num']} | {SITE_NAME}"
    ld = [{"@context": "https://schema.org", "@type": "LearningResource", "name": lesson["title"],
           "description": lesson["description"], "url": SITE_URL + lesson["url"], "inLanguage": "en",
           "educationalLevel": "Beginner", "dateModified": git_lastmod(lesson["path"]),
           "isPartOf": {"@type": "Course", "name": SITE_NAME, "url": SITE_URL}}, breadcrumb_ld(crumbs)]
    return page(path=lesson["url"], title=title, description=lesson["description"], body=body, rel=rel,
                og_type="article", ld=ld, active="lessons/")


def build_404():
    body = """
<section class="hero">
  <h1>Page not found</h1>
  <p class="lede">That page doesn’t exist. Try the <a href="/rag-from-scratch/">course home</a> or start with
  <a href="/rag-from-scratch/learn/">how to use these notes</a>.</p>
</section>"""
    # 404.html is served for any missing path, so it uses absolute site paths instead of relative ones.
    out = page(path="404.html", title=f"Page not found | {SITE_NAME}", description="Page not found.",
               body=body, rel="/rag-from-scratch/")
    return out.replace('<meta name="robots" content="index, follow">', '<meta name="robots" content="noindex">')


# ---------------------------------------------------------------- assets

FAVICON = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64"><rect width="64" height="64" rx="14" fill="#2f5d50"/><text x="32" y="42" font-family="system-ui,sans-serif" font-size="26" font-weight="700" text-anchor="middle" fill="#fff">RAG</text></svg>
"""

STYLE = """:root {
  --bg: #f7f6f2; --surface: #ffffff; --surface-2: #efede6; --text: #1d1d1b; --muted: #5f5e58;
  --border: #dcd9cf; --accent: #2f5d50; --accent-soft: #e2ece8; --code-bg: #f1efe8; --code-text: #2a2a27;
  --sidebar-w: 270px; --content-w: 760px;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --bg: #161714; --surface: #1f201d; --surface-2: #2a2b27; --text: #e8e6df; --muted: #a3a197;
    --border: #383935; --accent: #8cc4b0; --accent-soft: #233630; --code-bg: #262723; --code-text: #e2e0d8;
  }
}
:root[data-theme="dark"] {
  --bg: #161714; --surface: #1f201d; --surface-2: #2a2b27; --text: #e8e6df; --muted: #a3a197;
  --border: #383935; --accent: #8cc4b0; --accent-soft: #233630; --code-bg: #262723; --code-text: #e2e0d8;
}
* { box-sizing: border-box; }
html { scroll-padding-top: 72px; }
body { margin: 0; background: var(--bg); color: var(--text);
  font: 17px/1.7 system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; }
a { color: var(--accent); }
code, pre { font-family: ui-monospace, "Cascadia Code", Consolas, monospace; font-size: 0.86em; }
:not(pre) > code { background: var(--code-bg); padding: 0.1em 0.35em; border-radius: 4px; }
pre { background: var(--code-bg); color: var(--code-text); padding: 14px 16px; border-radius: 8px;
  overflow-x: auto; line-height: 1.5; border: 1px solid var(--border); }
.skip { position: absolute; left: -999px; }
.skip:focus { left: 16px; top: 8px; z-index: 10; background: var(--surface); padding: 6px 10px; }

.topbar { position: sticky; top: 0; z-index: 5; display: flex; align-items: center; gap: 24px;
  padding: 12px 24px; background: color-mix(in srgb, var(--bg) 92%, transparent);
  backdrop-filter: blur(8px); border-bottom: 1px solid var(--border); }
.brand { font-weight: 800; font-size: 1.05rem; color: var(--text); text-decoration: none; white-space: nowrap; }
.brand span { color: var(--accent); }
.topnav { display: flex; gap: 4px; flex: 1; overflow-x: auto; }
.topnav a { color: var(--muted); text-decoration: none; padding: 6px 10px; border-radius: 6px; font-size: 0.93rem; white-space: nowrap; }
.topnav a:hover { background: var(--surface-2); color: var(--text); }
.topnav a[aria-current] { color: var(--accent); background: var(--accent-soft); font-weight: 600; }
.theme-toggle { background: var(--surface-2); color: var(--text); border: 1px solid var(--border);
  border-radius: 6px; width: 34px; height: 34px; cursor: pointer; font-size: 1rem; }

.layout { max-width: 1240px; margin: 0 auto; padding: 0 24px; }
.layout.with-sidebar { display: grid; grid-template-columns: var(--sidebar-w) minmax(0, 1fr); gap: 40px; }
main { min-width: 0; padding: 32px 0 64px; }
.no-sidebar main { max-width: 1000px; margin: 0 auto; }

.sidebar { padding-top: 32px; }
.sidebar-box { position: sticky; top: 76px; max-height: calc(100vh - 96px); overflow-y: auto; }
.sidebar-box summary { font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.08em; color: var(--muted);
  font-weight: 600; cursor: pointer; padding: 4px 8px; list-style: none; }
.sidebar ul { list-style: none; margin: 8px 0 0; padding: 0; }
.sidebar a { display: flex; gap: 10px; align-items: baseline; padding: 6px 8px; border-radius: 6px;
  color: var(--text); text-decoration: none; font-size: 0.9rem; line-height: 1.4; }
.sidebar a:hover { background: var(--surface-2); }
.sidebar a[aria-current] { background: var(--accent-soft); color: var(--accent); font-weight: 600; }
.num { color: var(--muted); font-variant-numeric: tabular-nums; font-size: 0.8rem; min-width: 1.5em; }

.crumbs { font-size: 0.85rem; color: var(--muted); margin-bottom: 16px; }
.crumbs a { color: var(--muted); }
.prose { max-width: var(--content-w); }
.prose.wide { max-width: 900px; }
.prose h1 { font-size: 2.1rem; line-height: 1.2; margin: 0 0 16px; }
.prose h2 { font-size: 1.4rem; margin-top: 40px; border-bottom: 1px solid var(--border); padding-bottom: 6px; }
.prose h3 { font-size: 1.12rem; margin-top: 28px; }
.prose blockquote { margin: 16px 0; padding: 8px 16px; border-left: 4px solid var(--accent);
  background: var(--accent-soft); border-radius: 0 8px 8px 0; }
.prose blockquote p { margin: 6px 0; }
.prose hr { border: 0; border-top: 1px solid var(--border); margin: 32px 0; }
.prose img { max-width: 100%; }
.table-wrap { overflow-x: auto; margin: 16px 0; }
table { border-collapse: collapse; width: 100%; font-size: 0.92rem; }
th, td { border: 1px solid var(--border); padding: 8px 10px; text-align: left; vertical-align: top; }
th { background: var(--surface-2); }
.eyebrow { color: var(--accent); font-weight: 600; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.06em; margin: 0 0 8px; }
.lede { color: var(--muted); font-size: 1.12rem; }

.hero { padding: 24px 0 8px; }
.hero h1 { font-size: 3rem; line-height: 1.1; margin: 0 0 16px; }
.hero .lede { max-width: 720px; }
.cta { display: flex; flex-wrap: wrap; gap: 12px; margin: 24px 0; }
.btn { display: inline-block; padding: 10px 18px; border-radius: 8px; border: 1px solid var(--border);
  background: var(--surface); color: var(--text); text-decoration: none; font-weight: 600; }
.btn.primary { background: var(--accent); border-color: var(--accent); color: var(--bg); }
.stats { display: flex; flex-wrap: wrap; gap: 12px; margin: 8px 0; }
.stat { background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 10px 16px; }
.stat b { display: block; font-size: 1.3rem; }
.stat span { color: var(--muted); font-size: 0.85rem; }
main > section { margin-top: 48px; }
main > section > h2 { font-size: 1.5rem; }
.pipeline-steps { display: flex; flex-wrap: wrap; gap: 8px; padding: 0; list-style: none; counter-reset: s; }
.pipeline-steps li { counter-increment: s; background: var(--surface); border: 1px solid var(--border);
  border-radius: 8px; padding: 6px 12px; font-size: 0.93rem; }
.pipeline-steps li::before { content: counter(s) ". "; color: var(--muted); }
.module-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); gap: 14px; margin-top: 16px; }
.module-card { display: block; background: var(--surface); border: 1px solid var(--border); border-radius: 10px;
  padding: 16px 18px; color: var(--text); text-decoration: none; transition: border-color .15s; }
.module-card:hover { border-color: var(--accent); }
.module-card h3 { margin: 4px 0 6px; font-size: 1.02rem; line-height: 1.35; }
.module-card p { margin: 0; color: var(--muted); font-size: 0.88rem; line-height: 1.5; }
.split { display: grid; grid-template-columns: 1fr 1fr; gap: 32px; }

.flow { display: flex; flex-wrap: wrap; align-items: center; gap: 8px; margin: 16px 0; }
.flow .box { background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 8px 12px; font-size: 0.9rem; }
.flow .box small { display: block; color: var(--muted); font-size: 0.75rem; }
.flow .arrow { color: var(--muted); }
.file-card { background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 16px 20px; margin: 14px 0; }
.file-head { display: flex; gap: 14px; align-items: center; }
.file-head h3 { margin: 0; font-size: 1rem; }
.stage { margin: 2px 0 0; color: var(--muted); font-size: 0.85rem; }
.step { display: grid; place-items: center; width: 30px; height: 30px; border-radius: 50%; flex-shrink: 0;
  background: var(--accent-soft); color: var(--accent); font-weight: 700; }
details summary { cursor: pointer; color: var(--accent); font-size: 0.9rem; }
details pre { max-height: 520px; }

.prose details { background: var(--surface); border: 1px solid var(--border); border-radius: 8px;
  padding: 10px 14px; margin: 10px 0; }
.prose details summary { color: var(--text); font-weight: 600; font-size: 0.97rem; }
.prose details[open] summary { margin-bottom: 6px; }
.prose details > p:last-child { margin-bottom: 0; }
.sidebar-label { font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.08em; color: var(--muted);
  font-weight: 600; margin: 18px 8px 0; }
.pager { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; max-width: var(--content-w); margin-top: 48px; }
.pager-link { display: block; border: 1px solid var(--border); border-radius: 10px; padding: 12px 16px;
  background: var(--surface); text-decoration: none; color: var(--text); font-weight: 600; }
.pager-link:hover { border-color: var(--accent); }
.pager-link small { display: block; color: var(--muted); font-weight: 400; font-size: 0.8rem; }
.pager-link.next { text-align: right; }
.edit { font-size: 0.85rem; max-width: var(--content-w); }
.site-footer { border-top: 1px solid var(--border); color: var(--muted); font-size: 0.85rem; text-align: center; padding: 20px 16px; }

@media (max-width: 900px) {
  .layout { padding: 0 16px; }
  .layout.with-sidebar { display: block; }
  .sidebar { padding-top: 16px; }
  .sidebar-box { position: static; max-height: none; background: var(--surface); border: 1px solid var(--border);
    border-radius: 10px; padding: 8px; }
  .topbar { padding: 10px 16px; gap: 12px; flex-wrap: wrap; }
  .topnav { order: 3; flex-basis: 100%; }
  .brand { flex: 1; }
  main { padding: 20px 0 48px; }
  .hero h1 { font-size: 2.2rem; }
  .prose h1 { font-size: 1.7rem; }
  .split { grid-template-columns: 1fr; gap: 0; }
  .pager { grid-template-columns: 1fr; }
}
"""

SCRIPT = """(function () {
  var root = document.documentElement;
  var btn = document.getElementById("theme-toggle");
  if (btn) btn.addEventListener("click", function () {
    var current = root.getAttribute("data-theme") ||
      (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
    var next = current === "dark" ? "light" : "dark";
    root.setAttribute("data-theme", next);
    try { localStorage.setItem("theme", next); } catch (e) {}
  });
  // Collapse the module list by default on small screens so the article comes first.
  var box = document.querySelector(".sidebar-box");
  if (box && matchMedia("(max-width: 900px)").matches) box.removeAttribute("open");
})();
"""


# ---------------------------------------------------------------- build

def write(rel_path, content):
    out = OUT_DIR / rel_path
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(content, encoding="utf-8", newline="\n")


def build():
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    guide, modules, extras = load_notes()
    lessons = load_lessons()

    url_for_stem = {m["stem"]: m["url"] for m in modules + extras}
    if guide:
        url_for_stem["README"] = guide["url"]

    sitemap = [("", date.today().isoformat(), "1.0")]
    write("index.html", build_home(guide, modules, extras, lessons))

    if guide:
        write("learn/index.html", build_note_page(guide, guide, modules, extras, url_for_stem, None,
                                                  modules[0] if modules else None, is_guide=True))
        sitemap.append((guide["url"], git_lastmod(guide["path"]), "0.9"))
    for i, m in enumerate(modules):
        prev = modules[i - 1] if i > 0 else guide
        nxt = modules[i + 1] if i + 1 < len(modules) else None
        write(m["url"] + "index.html", build_note_page(m, guide, modules, extras, url_for_stem, prev, nxt))
        sitemap.append((m["url"], git_lastmod(m["path"]), "0.8"))
    for x in extras:
        write(x["url"] + "index.html", build_note_page(x, guide, modules, extras, url_for_stem, None, None))
        sitemap.append((x["url"], git_lastmod(x["path"]), "0.8"))

    write("implementation/index.html", build_implementation())
    sitemap.append(("implementation/", max(git_lastmod(PIPELINE_DIR / f) for f, _, _ in PIPELINE_FILES), "0.8"))

    write("lessons/index.html", build_lessons_index(lessons))
    sitemap.append(("lessons/", date.today().isoformat(), "0.6"))
    for i, l in enumerate(lessons):
        prev = lessons[i - 1] if i > 0 else None
        nxt = lessons[i + 1] if i + 1 < len(lessons) else None
        write(l["url"] + "index.html", build_lesson(l, prev, nxt))
        sitemap.append((l["url"], git_lastmod(l["path"]), "0.5"))

    write("404.html", build_404())
    write("assets/style.css", STYLE)
    write("assets/site.js", SCRIPT)
    write("assets/favicon.svg", FAVICON)
    write(".nojekyll", "")
    write("robots.txt", f"User-agent: *\nAllow: /\n\nSitemap: {SITE_URL}sitemap.xml\n")
    urls = "\n".join(
        f"  <url><loc>{SITE_URL}{p}</loc><lastmod>{d}</lastmod><priority>{pr}</priority></url>"
        for p, d, pr in sitemap
    )
    write("sitemap.xml", f'<?xml version="1.0" encoding="UTF-8"?>\n'
                         f'<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n{urls}\n</urlset>\n')
    print(f"Built {len(sitemap)} pages into {OUT_DIR.relative_to(ROOT)}/ "
          f"({len(modules)} modules, {len(extras)} reference pages, {len(lessons)} lessons)")


if __name__ == "__main__":
    build()
