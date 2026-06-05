#!/usr/bin/env python3
"""
build_html.py — converts pandoc slide markdown (##-based) to a styled, self-contained HTML presentation.
Usage: python3 build_html.py <input.md> [output.html]
"""

import sys
import re
from pathlib import Path

# ── Accent palette per domain ─────────────────────────────────────────────────

DOMAIN_COLORS = {
    "dl":    {"accent": "#818cf8", "bg": "rgba(129,140,248,0.10)", "border": "rgba(129,140,248,0.35)"},
    "ml":    {"accent": "#34d399", "bg": "rgba(52,211,153,0.10)",  "border": "rgba(52,211,153,0.35)"},
    "stats": {"accent": "#fbbf24", "bg": "rgba(251,191,36,0.10)",  "border": "rgba(251,191,36,0.35)"},
    "fe":    {"accent": "#fb7185", "bg": "rgba(251,113,133,0.10)", "border": "rgba(251,113,133,0.35)"},
    "intro": {"accent": "#a78bfa", "bg": "rgba(167,139,250,0.10)", "border": "rgba(167,139,250,0.35)"},
    "app":   {"accent": "#94a3b8", "bg": "rgba(148,163,184,0.08)", "border": "rgba(148,163,184,0.25)"},
}

DL_KEYWORDS   = ["deep learning", "sigmoid", "relu", "gelu", "gradient", "adam", "initialization",
                  "vanishing", "backprop", "cnn", "rnn", "lstm", "transformer", "derivation chain"]
ML_KEYWORDS   = ["algorithm", "naive bayes", "regulariz", "boosting", "decision tree", "random forest",
                  "svm", "xgboost", "lightgbm", "bias-variance", "ridge", "lasso"]
STATS_KEYWORDS = ["statistic", "evaluation", "clt", "accuracy", "auc", "f1", "precision", "recall",
                   "macro", "micro", "covariance", "pearson", "skew", "central limit"]
FE_KEYWORDS   = ["feature engineering", "scaling", "smote", "imbalance", "cross-valid", "min-max",
                  "standardiz", "cosine", "euclidean", "k-fold", "close", "framework", "chain"]
APP_KEYWORDS  = ["appendix", "a1 —", "a2 —", "quick reference", "selector"]


def classify(title: str) -> str:
    t = title.lower()
    if any(k in t for k in APP_KEYWORDS):   return "app"
    if any(k in t for k in DL_KEYWORDS):    return "dl"
    if any(k in t for k in ML_KEYWORDS):    return "ml"
    if any(k in t for k in STATS_KEYWORDS): return "stats"
    if any(k in t for k in FE_KEYWORDS):    return "fe"
    return "intro"


# ── Inline markdown renderer ──────────────────────────────────────────────────

def md_inline(text: str) -> str:
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"\*([^*]+?)\*",  r"<em>\1</em>", text)
    text = re.sub(r"`(.+?)`",        r"<code>\1</code>", text)
    text = re.sub(r"\[\^(\d+)\]",   r'<sup class="fn-ref">[\1]</sup>', text)
    return text


# ── Block renderers ───────────────────────────────────────────────────────────

def render_table(rows: list[str]) -> str:
    html = '<div class="table-wrap"><table>'
    is_header = True
    for row in rows:
        if re.match(r"^[\|\s\-:]+$", row):   # separator row
            is_header = False
            continue
        cells = [c.strip() for c in re.split(r"\|", row.strip("|"))]
        tag = "th" if is_header else "td"
        html += "<tr>" + "".join(f"<{tag}>{md_inline(c)}</{tag}>" for c in cells) + "</tr>"
        if is_header:
            is_header = False
    return html + "</table></div>"


def render_bullets(items: list[str]) -> str:
    return "<ul>" + "".join(f"<li>{md_inline(item)}</li>" for item in items) + "</ul>"


def render_callout(lines: list[str]) -> str:
    content = " ".join(re.sub(r"^>\s*", "", line) for line in lines)
    return f'<div class="callout">{md_inline(content)}</div>'


def render_footnotes(notes: dict) -> str:
    if not notes:
        return ""
    items = " &nbsp;·&nbsp; ".join(
        f'<span id="fn{n}">[{n}] {md_inline(t)}</span>' for n, t in sorted(notes.items())
    )
    return f'<div class="footnotes">{items}</div>'


# ── Slide parser ──────────────────────────────────────────────────────────────

def parse_front_matter(text: str) -> tuple[dict, str]:
    meta = {"title": "Presentation", "subtitle": "", "author": "", "date": ""}
    m = re.match(r"^---\n(.*?)\n---\n", text, re.DOTALL)
    if m:
        for line in m.group(1).splitlines():
            kv = re.match(r'^(\w+):\s*"?(.+?)"?\s*$', line)
            if kv:
                meta[kv.group(1)] = kv.group(2)
        text = text[m.end():]
    return meta, text


def split_into_raw_slides(text: str) -> list[str]:
    lines = text.splitlines()
    chunks, current = [], []

    def flush():
        s = "\n".join(current).strip()
        if s:
            chunks.append(s)
        current.clear()

    for line in lines:
        if line.strip() == "---":
            flush()
        elif re.match(r"^# [^#]", line):    # section-level heading = new slide
            flush()
            current.append(line)
        elif re.match(r"^## ", line):        # content heading = new slide
            flush()
            current.append(line)
        else:
            current.append(line)
    flush()
    return chunks


def parse_slide(raw: str) -> dict:
    slide = {"type": "content", "title": "", "body_blocks": [], "notes": "", "footnotes": {}}

    # Extract ::: notes ::: block
    notes_m = re.search(r":::\s*notes\s*\n(.*?)\n:::", raw, re.DOTALL)
    if notes_m:
        slide["notes"] = notes_m.group(1).strip()
        raw = raw[:notes_m.start()] + raw[notes_m.end():]

    # Extract footnote definitions
    for fn_m in re.finditer(r"^\[\^(\d+)\]:\s*(.+)$", raw, re.MULTILINE):
        slide["footnotes"][fn_m.group(1)] = fn_m.group(2).strip()
    raw = re.sub(r"^\[\^\d+\]:.+\n?", "", raw, flags=re.MULTILINE)

    lines = raw.strip().splitlines()
    if not lines:
        return slide

    # Detect slide type from first heading
    if re.match(r"^# [^#]", lines[0]):
        slide["type"] = "section"
        slide["title"] = lines[0][2:].strip()
        lines = lines[1:]
    elif re.match(r"^## ", lines[0]):
        slide["type"] = "content"
        slide["title"] = lines[0][3:].strip()
        lines = lines[1:]

    # Parse body blocks
    i = 0
    while i < len(lines):
        line = lines[i]
        if not line.strip():
            i += 1
            continue

        # Table block
        if "|" in line:
            table_rows = []
            while i < len(lines) and ("|" in lines[i] or re.match(r"^[\|\s\-:]+$", lines[i])):
                table_rows.append(lines[i])
                i += 1
            slide["body_blocks"].append(render_table(table_rows))
            continue

        # Blockquote callout
        if line.startswith(">"):
            bq = []
            while i < len(lines) and lines[i].startswith(">"):
                bq.append(lines[i])
                i += 1
            slide["body_blocks"].append(render_callout(bq))
            continue

        # Bullet list
        if line.startswith("- "):
            bullets = []
            while i < len(lines) and lines[i].startswith("- "):
                bullets.append(lines[i][2:].strip())
                i += 1
            slide["body_blocks"].append(render_bullets(bullets))
            continue

        # Paragraph
        slide["body_blocks"].append(f"<p>{md_inline(line.strip())}</p>")
        i += 1

    return slide


# ── HTML template ─────────────────────────────────────────────────────────────

CSS = """\
:root {
  --bg:      #0b0d14;
  --surface: #12151f;
  --border:  rgba(255,255,255,0.06);
  --text:    #e2e8f0;
  --muted:   #64748b;
  --accent:  #a78bfa;
  --acc-bg:  rgba(167,139,250,0.10);
  --acc-bdr: rgba(167,139,250,0.30);
}
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
html, body { width: 100%; height: 100%; overflow: hidden; }
body {
  font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
  background: var(--bg);
  color: var(--text);
}

/* ── Deck ── */
#deck { position: relative; width: 100vw; height: 100vh; }

/* ── Slides ── */
.slide {
  position: absolute; inset: 0;
  display: none;
  flex-direction: column;
  padding: 52px 68px 44px;
  opacity: 0;
  transition: opacity 0.3s ease;
}
.slide.active { display: flex; opacity: 1; }

/* ── Title slide ── */
.slide-title {
  justify-content: center; align-items: center; text-align: center;
  background: radial-gradient(ellipse at 60% 40%, rgba(129,140,248,0.12) 0%, transparent 70%),
              radial-gradient(ellipse at 30% 70%, rgba(167,139,250,0.08) 0%, transparent 60%);
}
.slide-title .deck-title {
  font-size: clamp(1.9rem, 3.2vw, 2.8rem);
  font-weight: 700;
  letter-spacing: -0.035em;
  line-height: 1.2;
  background: linear-gradient(135deg, #c4b5fd 0%, #818cf8 45%, #38bdf8 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  margin-bottom: 18px;
}
.slide-title .deck-subtitle { font-size: 1rem; color: #94a3b8; margin-bottom: 8px; }
.slide-title .deck-meta     { font-size: 0.82rem; color: #334155; font-family: 'SF Mono', monospace; }

/* ── Section divider ── */
.slide-section {
  justify-content: center; align-items: flex-start;
  padding-left: 72px;
  border-left: 3px solid var(--accent);
}
.slide-section .sec-label {
  font-size: 0.7rem; font-weight: 700; letter-spacing: 0.14em;
  text-transform: uppercase; color: var(--accent); margin-bottom: 14px;
}
.slide-section h2 {
  font-size: clamp(1.8rem, 2.8vw, 2.6rem);
  font-weight: 700; letter-spacing: -0.03em; color: #f8fafc;
}

/* ── Content slide header ── */
.slide-head {
  display: flex; align-items: flex-start; gap: 12px; margin-bottom: 26px;
}
.domain-tag {
  flex-shrink: 0;
  font-size: 0.62rem; font-weight: 700; letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--accent); background: var(--acc-bg);
  border: 1px solid var(--acc-bdr);
  border-radius: 4px; padding: 3px 9px; margin-top: 5px; white-space: nowrap;
}
.slide h2 {
  font-size: clamp(1.1rem, 1.7vw, 1.45rem);
  font-weight: 600; line-height: 1.35;
  color: #f1f5f9; letter-spacing: -0.015em;
}

/* ── Slide body ── */
.slide-body {
  flex: 1; display: flex; flex-direction: column;
  gap: 18px; overflow: hidden; min-height: 0;
}

/* ── Bullets ── */
ul { list-style: none; display: flex; flex-direction: column; gap: 13px; }
li {
  position: relative; padding-left: 22px;
  font-size: clamp(0.88rem, 1.25vw, 1.05rem);
  line-height: 1.6; color: #cbd5e1;
}
li::before {
  content: ''; position: absolute; left: 0; top: 9px;
  width: 6px; height: 6px; border-radius: 50%;
  background: var(--accent);
}

/* ── Data callout ── */
.callout {
  background: var(--acc-bg);
  border: 1px solid var(--acc-bdr);
  border-left: 3px solid var(--accent);
  border-radius: 6px;
  padding: 13px 18px;
  font-size: clamp(0.8rem, 1.1vw, 0.92rem);
  line-height: 1.65; color: #e2e8f0;
  font-family: 'SF Mono', 'Fira Code', 'Cascadia Code', monospace;
}

/* ── Tables ── */
.table-wrap { overflow-x: auto; border-radius: 6px; border: 1px solid var(--border); }
table { width: 100%; border-collapse: collapse; font-size: clamp(0.76rem, 1vw, 0.88rem); }
th {
  background: var(--acc-bg); color: var(--accent);
  font-weight: 600; font-size: 0.75rem;
  letter-spacing: 0.06em; text-transform: uppercase;
  padding: 9px 14px; text-align: left;
  border-bottom: 1px solid var(--acc-bdr);
}
td {
  padding: 8px 14px; color: #cbd5e1;
  border-bottom: 1px solid var(--border);
}
tr:last-child td { border-bottom: none; }
tr:hover td { background: rgba(255,255,255,0.025); }

/* ── Inline code ── */
code {
  font-family: 'SF Mono', 'Fira Code', 'Cascadia Code', monospace;
  font-size: 0.84em;
  background: rgba(255,255,255,0.08);
  padding: 1px 5px; border-radius: 3px; color: #c4b5fd;
}

/* ── Footnotes ── */
.footnotes {
  margin-top: auto; padding-top: 10px;
  border-top: 1px solid var(--border);
  font-size: 0.68rem; color: #475569; line-height: 1.55;
}
.fn-ref { font-size: 0.65em; color: var(--accent); vertical-align: super; }

/* ── Progress bar ── */
#progress {
  position: fixed; bottom: 0; left: 0; height: 2px; z-index: 50;
  background: var(--accent); transition: width 0.28s ease;
}

/* ── Counter ── */
#counter {
  position: fixed; bottom: 10px; right: 18px;
  font-size: 0.68rem; color: var(--muted);
  font-family: 'SF Mono', monospace; font-variant-numeric: tabular-nums;
  z-index: 50;
}

/* ── Key hint ── */
#keyhint {
  position: fixed; bottom: 10px; left: 18px;
  font-size: 0.64rem; color: #1e293b;
  font-family: 'SF Mono', monospace; z-index: 50;
}

/* ── Notes overlay ── */
#notes-bg {
  display: none; position: fixed; inset: 0;
  background: rgba(5,7,12,0.94); z-index: 100;
  padding: 56px 72px; overflow-y: auto;
}
#notes-bg.open { display: block; }
#notes-close {
  position: fixed; top: 18px; right: 26px;
  font-size: 1.4rem; color: #475569; cursor: pointer; z-index: 101;
  line-height: 1;
}
#notes-close:hover { color: #94a3b8; }
#notes-label {
  font-size: 0.7rem; font-weight: 700; letter-spacing: 0.12em;
  text-transform: uppercase; color: var(--accent); margin-bottom: 16px;
}
#notes-text {
  font-size: 1rem; color: #94a3b8;
  line-height: 1.75; white-space: pre-wrap;
  max-width: 780px;
}

/* ── Appendix badge ── */
.app-badge {
  position: absolute; top: 18px; right: 22px;
  font-size: 0.62rem; color: #334155;
  border: 1px solid #1e293b; padding: 2px 8px;
  border-radius: 3px; text-transform: uppercase; letter-spacing: 0.1em;
}

/* ── Strong / em in slide body ── */
strong { color: #f1f5f9; }
em     { color: #c4b5fd; font-style: italic; }
"""

JS = """\
(function() {
  const slides  = Array.from(document.querySelectorAll('.slide'));
  const bar     = document.getElementById('progress');
  const counter = document.getElementById('counter');
  const notesBg = document.getElementById('notes-bg');
  const notesTx = document.getElementById('notes-text');
  let cur = 0;

  function go(n) {
    slides[cur].classList.remove('active');
    cur = Math.max(0, Math.min(n, slides.length - 1));
    slides[cur].classList.add('active');
    const pct = slides.length > 1 ? (cur / (slides.length - 1)) * 100 : 0;
    bar.style.width = pct + '%';
    counter.textContent = (cur + 1) + ' / ' + slides.length;
    notesBg.classList.remove('open');
  }

  document.addEventListener('keydown', e => {
    if (e.key === 'ArrowRight' || e.key === ' ')  { e.preventDefault(); go(cur + 1); }
    else if (e.key === 'ArrowLeft')               { e.preventDefault(); go(cur - 1); }
    else if (e.key === 'n' || e.key === 'N') {
      const note = slides[cur].dataset.notes || '';
      if (note) { notesTx.textContent = note; notesBg.classList.toggle('open'); }
    }
    else if (e.key === 'Escape') { notesBg.classList.remove('open'); }
    else if (e.key === 'f' || e.key === 'F') {
      if (!document.fullscreenElement) document.documentElement.requestFullscreen?.();
      else document.exitFullscreen?.();
    }
  });

  document.getElementById('notes-close').addEventListener('click',
    () => notesBg.classList.remove('open'));

  document.getElementById('deck').addEventListener('click', e => {
    if (notesBg.classList.contains('open')) return;
    if (e.target.closest('#counter, #keyhint, #progress')) return;
    go(e.clientX > window.innerWidth / 2 ? cur + 1 : cur - 1);
  });

  go(0);
})();
"""


def css_vars(domain: str) -> str:
    c = DOMAIN_COLORS[domain]
    return f'--accent:{c["accent"]};--acc-bg:{c["bg"]};--acc-bdr:{c["border"]};'


def esc_attr(s: str) -> str:
    return s.replace("&", "&amp;").replace('"', "&quot;").replace("<", "&lt;").replace("\n", "&#10;")


def esc_html(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def build_html(meta: dict, slides: list[dict]) -> str:
    title = meta["title"]
    subtitle = meta["subtitle"]
    author = meta["author"]
    date = meta["date"]

    html_slides = []

    # ── Title slide ──
    html_slides.append(
        f'<div class="slide slide-title active" data-notes="">'
        f'<div class="deck-title">{esc_html(title)}</div>'
        f'<div class="deck-subtitle">{esc_html(subtitle)}</div>'
        f'<div class="deck-meta">{esc_html(author)} &nbsp;·&nbsp; {esc_html(date)}</div>'
        f'</div>'
    )

    current_domain = "intro"
    in_appendix = False

    for slide in slides:
        stype = slide["type"]
        stitle = slide["title"]

        if stype == "section":
            if "appendix" in stitle.lower():
                in_appendix = True
                current_domain = "app"
            else:
                current_domain = classify(stitle)
            cvars = css_vars(current_domain)
            notes_attr = esc_attr(slide["notes"])
            html_slides.append(
                f'<div class="slide slide-section" style="{cvars}" data-notes="{notes_attr}">'
                f'<div class="sec-label">Section</div>'
                f'<h2>{md_inline(stitle)}</h2>'
                f'</div>'
            )
            continue

        # Classify content slide domain
        domain = classify(stitle) if not in_appendix else "app"
        cvars = css_vars(domain)
        notes_attr = esc_attr(slide["notes"])

        domain_labels = {
            "dl": "Deep Learning", "ml": "ML Algorithms",
            "stats": "Stats + Eval", "fe": "Feature Engineering",
            "intro": "Framework", "app": "Appendix",
        }
        tag_label = domain_labels.get(domain, domain.upper())
        tag_html = f'<span class="domain-tag">{tag_label}</span>' if stitle else ""
        title_html = f'<h2>{md_inline(stitle)}</h2>' if stitle else ""
        app_badge = '<span class="app-badge">Appendix</span>' if in_appendix else ""
        body = "\n".join(slide["body_blocks"])
        fn_html = render_footnotes(slide["footnotes"])

        html_slides.append(
            f'<div class="slide" style="{cvars}" data-notes="{notes_attr}">'
            f'{app_badge}'
            f'<div class="slide-head">{tag_html}{title_html}</div>'
            f'<div class="slide-body">{body}{fn_html}</div>'
            f'</div>'
        )

    total = len(html_slides)
    slides_joined = "\n".join(html_slides)

    return f"""\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{esc_html(title)}</title>
<style>
{CSS}
</style>
</head>
<body>
<div id="deck">
{slides_joined}
</div>
<div id="progress" style="width:0"></div>
<div id="counter">1 / {total}</div>
<div id="keyhint">← → · N notes · F fullscreen</div>
<div id="notes-bg">
  <span id="notes-close">&#x2715;</span>
  <div id="notes-label">Speaker Notes</div>
  <div id="notes-text"></div>
</div>
<script>
{JS}
</script>
</body>
</html>"""


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 build_html.py <input.md> [output.html]", file=sys.stderr)
        sys.exit(1)

    src = Path(sys.argv[1])
    dst = Path(sys.argv[2]) if len(sys.argv) > 2 else src.with_suffix(".html")

    text = src.read_text(encoding="utf-8")
    meta, body = parse_front_matter(text)
    raw_slides = split_into_raw_slides(body)
    slides = [parse_slide(r) for r in raw_slides if r.strip()]

    html = build_html(meta, slides)
    dst.write_text(html, encoding="utf-8")
    total = 1 + sum(1 for s in slides)   # title + content slides
    print(f"✓ {dst}  ({total} slides, self-contained)")


if __name__ == "__main__":
    main()
