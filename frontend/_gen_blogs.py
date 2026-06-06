"""
Run from anywhere:  python3 frontend/_gen_blogs.py
Incremental build:  only rebuilds HTML files whose .md source is newer.
Force full rebuild: python3 frontend/_gen_blogs.py --force

Reads every .md source, embeds it inline, writes a styled .html to frontend/.
No server needed — works on file://.

──────────────────────────────────────────────────────────────────────────────
AUTO-SYNC ANSWER
──────────────────────────────────────────────────────────────────────────────
If you edit a .md file (with AI or manually), re-run this script:
  python3 frontend/_gen_blogs.py
It compares modification times and only regenerates changed files.
Use --force to rebuild everything regardless.
──────────────────────────────────────────────────────────────────────────────
"""
import os, sys, base64

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))   # frontend/
PROJECT_ROOT = os.path.dirname(BASE_DIR)                    # repo root
FORCE        = '--force' in sys.argv

# (source subdir relative to PROJECT_ROOT, md filename, output html, page title, back anchor)
ALL_FILES = [
    # ── Topics ───────────────────────────────────────────────────────────────
    ("01-sql",                "blog-sql.md",                    "blog-sql.html",                    "SQL for Data Science",                 "index.html#topics"),
    ("02-statistics",         "blog-statistics.md",             "blog-statistics.html",             "Statistics & Probability",             "index.html#topics"),
    ("03-dsa",                "blog-dsa.md",                    "blog-dsa.html",                    "Data Structures & Algorithms",         "index.html#topics"),
    ("04-python-pandas",      "blog-python-pandas.md",          "blog-python-pandas.html",          "Python, NumPy & Pandas",               "index.html#topics"),
    ("05-pytorch",            "blog-pytorch.md",                "blog-pytorch.html",                "PyTorch",                              "index.html#topics"),
    ("06-git",                "blog-git.md",                    "blog-git.html",                    "Git & CI/CD for ML",                   "index.html#topics"),
    ("07-machine-learning",   "blog-machine-learning.md",       "blog-machine-learning.html",       "Classical Machine Learning",           "index.html#topics"),
    ("08-deep-learning",      "blog-deep-learning.md",          "blog-deep-learning.html",          "Deep Learning",                        "index.html#topics"),
    ("09-recommenders",       "blog-recommenders.md",           "blog-recommenders.html",           "Recommender Systems",                  "index.html#topics"),
    ("10-nlp-foundations",    "blog-nlp-foundations.md",        "blog-nlp-foundations.html",        "NLP Foundations",                      "index.html#topics"),
    ("11-embedding-models",   "blog-embedding.md",              "blog-embedding.html",              "Embedding Models",                     "index.html#topics"),
    ("12-rag",                "blog-rag.md",                    "blog-rag.html",                    "Retrieval-Augmented Generation",       "index.html#topics"),
    ("13-agents",             "blog-agents.md",                 "blog-agents.html",                 "AI Agents",                            "index.html#topics"),
    ("14-llm",                "blog-llm.md",                    "blog-llm.html",                    "Large Language Models",                "index.html#topics"),
    ("15-llm-inferencing",    "blog-llm-inferencing.md",        "blog-llm-inferencing.html",        "LLM Inferencing & Optimization",       "index.html#topics"),
    ("16-llm-evaluation",     "blog-llm-evaluation.md",         "blog-llm-evaluation.html",         "LLM Evaluation",                       "index.html#topics"),
    ("17-reinforcement-learning", "blog-reinforcement-learning.md", "blog-reinforcement-learning.html", "Reinforcement Learning",           "index.html#topics"),
    ("18-speech",             "blog-speech.md",                 "blog-speech.html",                 "Speech: ASR, TTS & Voice Agents",      "index.html#topics"),
    ("19-vision-ocr",         "blog-vision-ocr.md",             "blog-vision-ocr.html",             "Vision & OCR",                         "index.html#topics"),
    # ── Interview ─────────────────────────────────────────────────────────────
    ("20-interview",          "blog-ds-ml-ai-interviews.md",    "interview-overview.html",          "DS/ML/AI Interview Guide",             "index.html#interview"),
    ("20-interview/Target",   "1.deep-learning-interview.md",   "interview-deep-learning.html",     "Deep Learning — Interview Q&A",        "index.html#interview"),
    ("20-interview/Target",   "2.ml-algorithms-interview.md",   "interview-ml-algorithms.html",     "ML Algorithms — Interview Q&A",        "index.html#interview"),
    ("20-interview/Target",   "3.statistics-interview.md",      "interview-statistics.html",        "Statistics — Interview Q&A",           "index.html#interview"),
    ("20-interview/Target",   "4.model-evaluation-interview.md","interview-model-eval.html",        "Model Evaluation — Interview Q&A",     "index.html#interview"),
    ("20-interview/Target",   "5.feature-engineering-interview.md", "interview-feature-eng.html",  "Feature Engineering — Interview Q&A",  "index.html#interview"),
    # ── References ───────────────────────────────────────────────────────────
    (".",                     "references.md",                  "references.html",                  "LearningLab — Reference URLs",         "index.html#references"),
]

# ─────────────────────────────────────────────────────────────────────────────
# STYLE
# ─────────────────────────────────────────────────────────────────────────────
STYLE = r"""
  :root {
    --bg:       #f6f1e7;
    --bg-mid:   #ede6d6;
    --bg-code:  #1e1c18;
    --text:     #1c1814;
    --muted:    #6e6259;
    --faint:    #a89f94;
    --accent:   #c85e35;
    --border:   #d9cfc0;
    --f-serif:  'DM Serif Display', Georgia, serif;
    --f-body:   'Syne', sans-serif;
    --f-mono:   'JetBrains Mono', 'Fira Code', monospace;
    --nav-h:    52px;
  }
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  html { font-size: 16px; scroll-behavior: smooth; }
  body { font-family: var(--f-body); background: var(--bg); color: var(--text); -webkit-font-smoothing: antialiased; line-height: 1.7; }

  /* ── nav ── */
  nav {
    position: sticky; top: 0; z-index: 100;
    background: var(--bg); border-bottom: 1px solid var(--border);
    padding: 0 48px; height: var(--nav-h);
    display: flex; align-items: center; justify-content: space-between;
  }
  .nav-back { font-family: var(--f-mono); font-size: 11px; letter-spacing: 0.8px; text-transform: uppercase; color: var(--muted); text-decoration: none; display: flex; align-items: center; gap: 6px; transition: color .15s; }
  .nav-back:hover { color: var(--accent); }
  .nav-title { font-family: var(--f-serif); font-size: 15px; color: var(--muted); }

  /* ── two-column layout ── */
  .page-layout {
    display: grid;
    grid-template-columns: minmax(0, 1fr) 220px;
    gap: 0 48px;
    max-width: 1120px;
    margin: 0 auto;
    padding: 0 48px;
    align-items: start;
  }

  /* ── article ── */
  article { padding: 56px 0 120px; min-width: 0; }

  article h1 { font-family: var(--f-serif); font-size: clamp(28px, 4.5vw, 46px); font-weight: 400; line-height: 1.1; letter-spacing: -0.5px; color: var(--text); margin-bottom: 22px; }
  article h2 { font-family: var(--f-serif); font-size: 24px; font-weight: 400; line-height: 1.2; color: var(--text); margin: 52px 0 14px; padding-bottom: 8px; border-bottom: 1px solid var(--border); }
  article h3 { font-family: var(--f-serif); font-size: 18px; font-weight: 400; color: var(--text); margin: 32px 0 10px; }
  article h4 { font-family: var(--f-body); font-size: 13px; font-weight: 600; letter-spacing: 0.5px; text-transform: uppercase; color: var(--muted); margin: 24px 0 8px; }

  article p { margin-bottom: 16px; font-size: 15.5px; }
  article a { color: var(--accent); text-decoration: underline; text-underline-offset: 3px; }
  article a:hover { opacity: 0.8; }
  article strong { font-weight: 600; }
  article ul, article ol { margin: 0 0 16px 24px; font-size: 15.5px; }
  article li { margin-bottom: 5px; }
  article li > ul, article li > ol { margin-top: 5px; margin-bottom: 0; }

  article code { font-family: var(--f-mono); font-size: 13px; background: var(--bg-mid); padding: 1px 6px; border-radius: 3px; color: #9b3f1f; }
  article pre { background: var(--bg-code); padding: 20px 24px; margin: 20px 0; overflow-x: auto; border-left: 3px solid var(--accent); }
  article pre code { font-family: var(--f-mono); font-size: 13px; background: transparent; color: #d4c9b8; padding: 0; border-radius: 0; }

  .hljs { background: transparent; color: #d4c9b8; }
  .hljs-keyword, .hljs-built_in { color: #e8956d; }
  .hljs-string  { color: #a8c97a; }
  .hljs-comment { color: #7a7060; font-style: italic; }
  .hljs-number, .hljs-literal { color: #7ec8e3; }
  .hljs-title, .hljs-function { color: #c4a3e0; }
  .hljs-attr, .hljs-variable  { color: #f0c879; }

  article blockquote { border-left: 3px solid var(--accent); padding: 12px 20px; margin: 20px 0; background: var(--bg-mid); font-size: 15px; color: var(--muted); }
  article blockquote p { margin: 0; }

  .table-wrap { overflow-x: auto; margin: 20px 0; }
  article table { width: 100%; border-collapse: collapse; font-size: 13.5px; }
  article th { background: var(--bg-mid); font-weight: 600; padding: 10px 14px; text-align: left; border-bottom: 2px solid var(--border); font-family: var(--f-body); font-size: 12px; letter-spacing: 0.3px; text-transform: uppercase; }
  article td { padding: 9px 14px; border-bottom: 1px solid var(--border); vertical-align: top; }
  article tr:hover td { background: var(--bg-mid); }

  article hr { border: none; border-top: 1px solid var(--border); margin: 36px 0; }
  article img { max-width: 100%; margin: 14px 0; }

  /* ── math ── */
  .math-block  { overflow-x: auto; padding: 12px 0; text-align: center; margin: 20px 0; }
  .math-inline { display: inline; }
  .katex       { font-size: 1em; }
  .katex-display { overflow-x: auto; overflow-y: hidden; }

  /* ── sticky TOC sidebar ── */
  .toc-sidebar {
    position: sticky;
    top: calc(var(--nav-h) + 20px);
    padding-top: 56px;
    padding-bottom: 40px;
    max-height: calc(100vh - var(--nav-h) - 20px);
    overflow-y: auto;
    scrollbar-width: thin;
  }
  .toc-sidebar::-webkit-scrollbar { width: 3px; }
  .toc-sidebar::-webkit-scrollbar-thumb { background: var(--border); }

  .toc-label { font-family: var(--f-mono); font-size: 9px; letter-spacing: 1.8px; text-transform: uppercase; color: var(--faint); margin-bottom: 14px; display: block; }
  .toc-list { list-style: none; padding: 0; margin: 0; }

  /* h2 group row */
  .toc-group { margin-bottom: 2px; }
  .toc-group-hd {
    display: flex;
    align-items: baseline;
    gap: 5px;
    cursor: pointer;
    padding: 5px 0 5px 12px;
    border-left: 2px solid transparent;
    margin-left: 0;
    transition: border-color .15s;
  }
  .toc-group-hd:hover { border-left-color: var(--border); }
  .toc-group-hd a {
    font-family: var(--f-body);
    font-size: 12px;
    font-weight: 500;
    line-height: 1.4;
    color: var(--muted);
    text-decoration: none;
    flex: 1;
    transition: color .15s;
  }
  .toc-group-hd a:hover { color: var(--text); }
  .toc-group-hd.toc-active { border-left-color: var(--accent); }
  .toc-group-hd.toc-active a { color: var(--accent); }

  /* collapse toggle arrow */
  .toc-arrow {
    font-size: 8px;
    color: var(--faint);
    flex-shrink: 0;
    transform: rotate(0deg);
    transition: transform .18s ease;
    user-select: none;
    line-height: 1;
  }
  .toc-group.open > .toc-group-hd .toc-arrow { transform: rotate(90deg); }

  /* h3 sub-list */
  .toc-sub {
    list-style: none;
    padding: 0;
    margin: 0 0 4px 0;
    border-left: 1px solid var(--border);
    margin-left: 18px;
    overflow: hidden;
    max-height: 0;
    transition: max-height .22s ease;
  }
  .toc-group.open > .toc-sub { max-height: 1200px; }
  .toc-sub li a {
    display: block;
    font-family: var(--f-body);
    font-size: 11px;
    line-height: 1.4;
    color: var(--faint);
    text-decoration: none;
    padding: 3px 0 3px 12px;
    border-left: 2px solid transparent;
    margin-left: -1px;
    transition: color .15s, border-color .15s;
  }
  .toc-sub li a:hover { color: var(--text); }
  .toc-sub li a.toc-active { color: var(--accent); border-left-color: var(--accent); }

  /* ── scroll to top ── */
  .scroll-top { position: fixed; bottom: 32px; right: 40px; width: 38px; height: 38px; background: var(--accent); color: white; border: none; cursor: pointer; font-size: 17px; display: flex; align-items: center; justify-content: center; opacity: 0; transition: opacity .2s; }
  .scroll-top.show { opacity: 1; }
  .scroll-top:hover { background: #a84d28; }

  /* ── print ── */
  @media print {
    nav, .toc-sidebar, .scroll-top { display: none; }
    .page-layout { display: block; }
    article { padding: 0; }
    article pre { break-inside: avoid; }
    body { background: white; }
  }

  /* ── responsive ── */
  @media (max-width: 900px) {
    .page-layout { grid-template-columns: 1fr; padding: 0 24px; }
    .toc-sidebar { display: none; }
    nav { padding: 0 24px; }
    article { padding: 40px 0 80px; }
  }
"""

# ─────────────────────────────────────────────────────────────────────────────
# HTML TEMPLATE
# ─────────────────────────────────────────────────────────────────────────────
HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>{title} — LearningLab</title>
  <link rel="preconnect" href="https://fonts.googleapis.com" />
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
  <link href="https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=Syne:wght@400;500;600&family=JetBrains+Mono:wght@300;400;500&display=swap" rel="stylesheet" />
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css" />
  <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js"></script>
  <script defer src="https://cdn.jsdelivr.net/npm/marked@9/marked.min.js"></script>
  <script defer src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js"></script>
  <style>{style}</style>
</head>
<body>

<nav>
  <a href="{back_link}" class="nav-back">← Back to LearningLab</a>
  <span class="nav-title">{title}</span>
</nav>

<div class="page-layout">
  <article id="content"><p style="color:var(--faint);padding:40px 0;font-family:var(--f-mono);font-size:12px">Loading…</p></article>
  <aside class="toc-sidebar" id="toc"></aside>
</div>

<button class="scroll-top" id="scrollTop" onclick="window.scrollTo({{top:0,behavior:'smooth'}})">↑</button>

<script>
/* Content is Base64-encoded so no escaping issues regardless of chars in md */
const _B64 = "{b64}";

function decodeB64(b64) {{
  const bytes = Uint8Array.from(atob(b64), c => c.charCodeAt(0));
  return new TextDecoder('utf-8').decode(bytes);
}}

/* ── Math: protect $..$ and $$...$$ before marked processes them ── */
function extractMath(src) {{
  const store = [];

  /* block math  $$...$$ */
  src = src.replace(/[$][$]([\\s\\S]+?)[$][$]/g, function(_, math) {{
    store.push({{ math: math, block: true }});
    return '\\n\\nMATHBLK' + (store.length - 1) + '\\n\\n';
  }});

  /* inline math  $...$  — simple pattern, no lookbehind needed */
  src = src.replace(/[$]([^\\n$]+?)[$]/g, function(_, math) {{
    store.push({{ math: math, block: false }});
    return 'MATHINL' + (store.length - 1);
  }});

  return {{ src: src, store: store }};
}}

function restoreMath(html, store) {{
  store.forEach(function(entry, i) {{
    var rendered;
    try {{
      rendered = katex.renderToString(entry.math, {{
        displayMode:  entry.block,
        throwOnError: false,
      }});
    }} catch(e) {{
      rendered = '<code>' + entry.math + '</code>';
    }}
    if (entry.block) {{
      /* marked wraps lone paragraphs with <p> */
      html = html.replace(new RegExp('<p>\\\\s*MATHBLK' + i + '\\\\s*</p>', 'g'),
                          '<div class="math-block">' + rendered + '</div>');
      /* fallback if not wrapped */
      html = html.split('MATHBLK' + i).join('<div class="math-block">' + rendered + '</div>');
    }} else {{
      html = html.split('MATHINL' + i).join('<span class="math-inline">' + rendered + '</span>');
    }}
  }});
  return html;
}}

/* ── Wait for all deferred scripts, then render ── */
window.addEventListener('load', function() {{
  var contentEl = document.getElementById('content');

  if (typeof marked === 'undefined') {{
    contentEl.innerHTML = '<p style="color:#c85e35;padding:40px 0;">Could not load marked.js — check your internet connection and reload.</p>';
    return;
  }}

  try {{
    var MD = decodeB64(_B64);

    /* GitHub-style slug — matches what the markdown's own TOC links use */
    function slugify(text) {{
      return text
        .replace(/<[^>]+>/g, '')          /* strip any HTML tags */
        .toLowerCase()
        .replace(/[^\\w\\s-]/g, '')       /* drop punctuation except word chars, spaces, hyphens */
        .replace(/[\\s]+/g, '-')          /* spaces → hyphens */
        .replace(/-+/g, '-')              /* collapse runs of hyphens */
        .replace(/^-|-$/g, '');           /* trim leading/trailing hyphens */
    }}

    /* track slugs to deduplicate (GitHub appends -1, -2 … on collision) */
    var slugCount = {{}};
    function uniqueSlug(base) {{
      if (!(base in slugCount)) {{ slugCount[base] = 0; return base; }}
      slugCount[base]++;
      return base + '-' + slugCount[base];
    }}

    /* configure marked v9 with slug-based heading IDs */
    marked.use({{
      gfm: true,
      breaks: false,
      renderer: {{
        heading: function(text, level) {{
          var id = uniqueSlug(slugify(text));
          return '<h' + level + ' id="' + id + '">' + text + '</h' + level + '>';
        }},
        table: function(header, body) {{
          return '<div class="table-wrap"><table><thead>' + header + '</thead><tbody>' + body + '</tbody></table></div>';
        }}
      }}
    }});

    var extracted = extractMath(MD);
    var html      = marked.parse(extracted.src);
    html          = restoreMath(html, extracted.store);

    contentEl.innerHTML = html;

    /* syntax highlight */
    if (typeof hljs !== 'undefined') {{
      document.querySelectorAll('pre code').forEach(function(el) {{ hljs.highlightElement(el); }});
    }}

    /* ── TOC sidebar: grouped by h2, h3s nested + collapsible ── */
    var allHeadings = Array.from(document.querySelectorAll('#content h2, #content h3'));
    var tocEl       = document.getElementById('toc');

    /* skip the in-page "Table of Contents" heading itself */
    var headings = allHeadings.filter(function(h) {{
      return !/table of contents/i.test(h.textContent);
    }});

    if (headings.length > 1) {{
      /* build group structure: each h2 starts a new group; h3s attach to last h2 */
      var groups = [];
      var cur    = null;
      headings.forEach(function(h) {{
        if (h.tagName === 'H2') {{
          cur = {{ h: h, children: [] }};
          groups.push(cur);
        }} else if (cur) {{
          cur.children.push(h);
        }}
      }});

      /* render grouped list */
      var html = '<span class="toc-label">On this page</span><ul class="toc-list">';
      groups.forEach(function(g, gi) {{
        var hasSubs = g.children.length > 0;
        html += '<li class="toc-group" data-gi="' + gi + '">';
        html += '<div class="toc-group-hd">';
        if (hasSubs) html += '<span class="toc-arrow">&#9658;</span>';
        html += '<a href="#' + g.h.id + '">' + g.h.textContent.trim().replace(/</g,'&lt;') + '</a>';
        html += '</div>';
        if (hasSubs) {{
          html += '<ul class="toc-sub">';
          g.children.forEach(function(h) {{
            html += '<li><a href="#' + h.id + '">' + h.textContent.trim().replace(/</g,'&lt;') + '</a></li>';
          }});
          html += '</ul>';
        }}
        html += '</li>';
      }});
      html += '</ul>';
      tocEl.innerHTML = html;

      /* click on group header: toggle open/closed; navigate on link click */
      tocEl.querySelectorAll('.toc-group-hd').forEach(function(hd) {{
        hd.addEventListener('click', function(e) {{
          /* if they clicked the link itself, let navigation happen normally */
          if (e.target.tagName === 'A') return;
          var group = hd.closest('.toc-group');
          group.classList.toggle('open');
        }});
      }});

      /* ── scroll tracking: highlight active h2 group and h3 sub-item ── */
      var flatLinks = [];   /* {{a, h, groupEl, isH3}} for every toc entry */
      groups.forEach(function(g) {{
        var groupEl = tocEl.querySelector('.toc-group[data-gi="' + groups.indexOf(g) + '"]');
        var hd = groupEl.querySelector('.toc-group-hd');
        flatLinks.push({{ h: g.h, hd: hd, groupEl: groupEl, isH3: false }});
        g.children.forEach(function(child) {{
          var a = groupEl.querySelector('.toc-sub a[href="#' + child.id + '"]');
          if (a) flatLinks.push({{ h: child, a: a, groupEl: groupEl, isH3: true }});
        }});
      }});

      var activeH = null;

      function updateToc() {{
        var newH = null;
        for (var i = flatLinks.length - 1; i >= 0; i--) {{
          if (flatLinks[i].h.getBoundingClientRect().top <= 120) {{
            newH = flatLinks[i];
            break;
          }}
        }}
        if (newH === activeH) return;
        activeH = newH;

        /* clear all active states */
        tocEl.querySelectorAll('.toc-active').forEach(function(el) {{ el.classList.remove('toc-active'); }});

        if (!newH) return;

        /* highlight the active item */
        if (newH.isH3) {{
          if (newH.a) newH.a.classList.add('toc-active');
        }} else {{
          newH.hd.classList.add('toc-active');
        }}

        /* auto-open the group containing the active item */
        groups.forEach(function(g, gi) {{
          var groupEl = tocEl.querySelector('.toc-group[data-gi="' + gi + '"]');
          var isActive = newH.groupEl === groupEl;
          if (isActive) groupEl.classList.add('open');
        }});
      }}

      window.addEventListener('scroll', updateToc, {{ passive: true }});
      updateToc();
    }}

    /* scroll-to-top */
    var btn = document.getElementById('scrollTop');
    window.addEventListener('scroll', function() {{
      btn.classList.toggle('show', window.scrollY > 400);
    }}, {{ passive: true }});

  }} catch(err) {{
    contentEl.innerHTML = '<p style="color:#c85e35;padding:40px 0;font-family:monospace">Render error: ' + err.message + '</p>';
    console.error(err);
  }}
}});
</script>
</body>
</html>
"""

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def encode_md(content):
    """Base64-encode markdown — completely safe, no escaping edge cases."""
    return base64.b64encode(content.encode('utf-8')).decode('ascii')

# ─────────────────────────────────────────────────────────────────────────────
# BUILD
# ─────────────────────────────────────────────────────────────────────────────
generated, skipped, uptodate = [], [], []

for subdir, mdfile, out_html, title, back_link in ALL_FILES:
    src_dir  = os.path.join(PROJECT_ROOT, subdir)
    md_path  = os.path.join(src_dir, mdfile)
    out_path = os.path.join(BASE_DIR, out_html)

    if not os.path.exists(md_path):
        skipped.append(out_html)
        print(f"  ✗  SKIP  {out_html}  (source not found: {os.path.relpath(md_path)})")
        continue

    # incremental: skip if output is newer than source
    if not FORCE and os.path.exists(out_path):
        if os.path.getmtime(md_path) <= os.path.getmtime(out_path):
            uptodate.append(out_html)
            print(f"  –  OK    {out_html}  (up to date)")
            continue

    with open(md_path, 'r', encoding='utf-8') as f:
        raw = f.read()

    html = HTML_TEMPLATE.format(
        title     = title,
        style     = STYLE,
        b64       = encode_md(raw),
        back_link = back_link,
    )

    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(html)

    generated.append(out_html)
    print(f"  ✓  BUILD {out_html}")

print(f"\nDone — {len(generated)} built, {len(uptodate)} up to date, {len(skipped)} skipped.")
if skipped:
    print("Skipped files:", ', '.join(skipped))
print("\nTip: edit any .md file and re-run this script — only changed files are rebuilt.")
print("     python3 frontend/_gen_blogs.py --force   ← rebuild everything")
