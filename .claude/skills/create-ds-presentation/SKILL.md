---
name: create-ds-presentation
description: Creates structured, persuasive data science presentations using McLuhan & Davies' Think on Your Feet® structures, McKinsey Pyramid Principle / SCQA narrative, and DS-specific slide design rules. Use when asked to "create a presentation", "build slides", "prepare a talk", "make a deck", or "present findings" on any data science, ML, or AI topic. Do NOT use for general writing tasks, blog posts, or non-presentation documents.
license: MIT
metadata:
  author: LearningLab
  version: 1.3.0
  category: communication
  tags: [presentation, data-science, storytelling, think-on-your-feet, slides, communication, html, pptx]
---

# Create Data Science Presentation

You build presentations the way a **principal data scientist pitching to a VP of Product** would — not like someone dumping analysis slides. Every slide earns its place. Every number connects to a decision. The audience leaves knowing what to do next.

## CRITICAL Rules

Read these before doing anything else. They are non-negotiable.

**1. No uncited facts or numbers on any slide.**
Every external statistic, benchmark, or claim requires an inline footnote (`¹`) and a footer citation with URL. Internal data requires a `Source: [system, date]` label. If you cannot find or verify a citation, **omit the claim entirely** and log it as `[OMITTED — no verifiable source: "claim text"]` so the user can supply the source. Never write "studies show" or "research suggests" without a specific named source and link.

**2. All content is gated by audience type.**
Read `references/audience-calibration.md` before writing any slide body. Executive slides never show methodology or stat language. Technical slides include all rigor signals. Mixed rooms get plain-language glosses on every technical term. Client slides use no internal system names.

**3. SCQA always opens the deck.**
Slide 1 = title. Slide 2 = SCQA Answer (the recommendation, stated immediately). Everything else proves it.

---

## Phase 1: Clarify Before Building

Do NOT start writing slides yet. Ask the user these questions in one message:

1. **Audience** — Executive / Technical peer / Mixed room / External client?
2. **Goal** — What must the audience DO after this presentation? (approve, change a decision, understand a finding, learn a concept)
3. **Topic and key data** — What is the main finding or recommendation? What numbers drive it?
4. **Time constraint** — How many minutes / slides? (Rule: 1 content slide per minute)
5. **Blockers** — Sensitive data to hide, stakeholder concerns to pre-empt, prior failures to acknowledge?

After receiving answers, suggest the best Think on Your Feet® structure from `references/toyf-structures.md` and confirm with the user before writing.

Show the deck shape before writing slides:
```
AUDIENCE CONTRACT: [type] — showing [included content] — hiding [appendix content]
DECK SHAPE:
  Opening:  SCQA — [N] slides
  Body:     [Structure name] — [N] slides
  Close:    Call to action — [N] slides
  Appendix: [what goes here]
```

Wait for user confirmation, then proceed.

---

## Phase 2: Write the Conceptual Slide Deck

Read `assets/slide-template.md` for the full slide format.

### Slide rules

**Action titles — every slide:**
- NOT: "Model Results" → YES: "Model reduces churn by 18% vs. baseline in A/B test"
- The title IS the message. The body proves it.

**One insight per slide.** Two bullets = two slides. Exception: appendix.

**Data callout box — every data slide:**
```
┌──────────────────────────────┐
│  KEY NUMBER: 18% lift        │
│  n = 45,000 users            │
│  CI: [15.2%, 20.8%] p<0.01   │
│  Source: [system, date]      │
└──────────────────────────────┘
```

**Citation footnotes — every external fact:**
```
Slide body:  [claim]¹
Slide footer: ¹ Author et al., Year — Title (URL)
              OR: Source: [Internal system — date — n=size]
```

**Audience annotations** (in your output, not on slides):
```
[EXEC]     — business takeaway for executives
[TECH]     — rigor/methodology note for technical peers
[MIXED]    — plain-language gloss needed here
[CLIENT]   — internal term to replace with descriptive label
[QUESTION] — likely challenge from the room
```

**Rule of 3:** Max 3 bullets per slide. Max 3 key messages in the whole deck.

**Jargon rule for non-technical audiences:** Define every acronym on first use. See `references/audience-calibration.md` for metric translation table.

---

## Phase 3: Write the Pandoc Slide Markdown File

After completing the conceptual deck output, write the same deck as a **pandoc-format markdown file saved to disk**. This is the source that pandoc compiles into both HTML and PPTX.

Read `references/pandoc-slide-format.md` for the exact format rules, header conventions, speaker note syntax, blockquote callout style, footnote format, and appendix structure.

### File naming and location
- Derive a `<deck-slug>` from the topic (kebab-case, no spaces)
- Save to the **output directory specified by the user** (default: project root)
- Filename: `<deck-slug>.md`

### Key rules (full details in reference file)
- YAML front matter at top: `title`, `subtitle`, `author`, `date`
- `##` heading = slide title (action sentence). `#` heading = section divider. `---` = forced slide break
- Max 3 bullets per slide — overflow goes to appendix
- `:::notes` / `:::` fenced divs for speaker notes (appear in PPTX presenter view)
- Blockquote for data callout: `> **KEY NUMBER** · context · Source:`
- `[^1]` footnotes for citations; define at the bottom of the slide block
- Do NOT include `[EXEC]`, `[TECH]`, etc. annotation tags — those are for the conceptual output only

---

## Phase 4: Prepare for Questions

For each major claim, write a BRIDGE response using the structure from `references/toyf-structures.md`:
```
QUESTION BANK
Q: [Likely challenge]
Bridge: "That's a fair concern about [X]. In our data, [Y shows Z], which means..."
```

Include 5 bridges in the Speaker Notes section of your output.

---

## Phase 5: Quality Check

**Narrative test:** Read only slide titles in order. Does the story hold? If not, rewrite titles.

**Citation integrity:**
- Every external number has an inline footnote + footer citation with URL
- Internal data has `Source:` label
- No "studies show" without a named source
- All omissions logged as `[OMITTED — no verifiable source]`

**Audience calibration:** Re-read the AUDIENCE CONTRACT line. Confirm gating rules from `references/audience-calibration.md` are applied.

**Think on Your Feet® test:**
- Does the opening follow SCQA?
- Is the recommendation on slide 2?
- Are 5 bridges prepared?
- Does the close state a specific action, owner, and date?

**Delivery checklist** (include at end of output):
```
DELIVERY CHECKLIST
□ Opening SCQA rehearsed cold (60 seconds, no slides)
□ Key number committed to memory
□ 5 hard questions prepared with bridges
□ Backup slide for each major claim in appendix
□ Timing: [N slides × 1 min] = [X min] target
```

---

## Phase 6: Generate HTML and PPTX Files

After the pandoc markdown source file has been written to disk (Phase 3), run both commands below using the Bash tool. Both read from the same `<deck-slug>.md` source file.

### Step 1 — Generate HTML (custom-designed slide presentation)

Use the skill's `build_html.py` script — it produces a fully self-contained HTML file with a professional dark theme, domain-coloured accents, styled callout boxes, and keyboard navigation. No CDN or internet required.

```bash
# Run from the output directory (where <deck-slug>.md lives)
python3 "<skill-base-dir>/scripts/build_html.py" "<deck-slug>.md" "<deck-slug>.html"
```

Where `<skill-base-dir>` is the path to the `create-ds-presentation` skill folder:
`.claude/skills/create-ds-presentation`

**What the script produces:**
- Self-contained `.html` — open in any browser, zero dependencies
- Dark theme (`#0b0d14`) with domain-coloured accents: indigo (DL), emerald (ML), amber (Stats/Eval), rose (FE)
- Styled callout boxes (blockquotes → highlighted data panels), clean table design with accent headers
- `← →` / Space to navigate, `N` for speaker notes overlay, `F` for fullscreen
- Progress bar + slide counter

**Domain colour mapping** (automatic, keyword-based):
- Deep Learning / gradient / adam → indigo `#818cf8`
- ML Algorithms / boosting / regulariz → emerald `#34d399`
- Statistics / evaluation / AUC / F1 → amber `#fbbf24`
- Feature Engineering / SMOTE / scaling → rose `#fb7185`
- Appendix slides → slate `#94a3b8`

### Step 2 — Generate PPTX (PowerPoint)

```bash
pandoc "<deck-slug>.md" \
  -t pptx \
  --slide-level=2 \
  -o "<deck-slug>.pptx"
```

- Pandoc maps `##` headings → slide titles, bullets → content, `:::notes` → presenter view notes, tables → native PPTX tables
- The `.pptx` uses PowerPoint's default Office Theme — apply a custom theme by adding `--reference-doc=<template.pptx>`

### Step 3 — Verify and report

After both commands run, confirm:
```
✓ <deck-slug>.html  — [N] slides, self-contained (open in browser; ← → to navigate, N for notes)
✓ <deck-slug>.pptx  — [N] slides, PowerPoint format (View → Presenter View for speaker notes)
```

If pandoc is not installed (`which pandoc` returns nothing):
```
brew install pandoc   # macOS
sudo apt install pandoc   # Ubuntu/Debian
```

---

## Output Format

Four outputs produced by this skill, in this order:

### 1. Conceptual Slide Deck (in-chat reference)
```
AUDIENCE CONTRACT: [type] — showing [...] — hiding [...]

--- SLIDE N ---
TITLE: [action title]
BODY:
  • [point]¹
  • [point]
CITATION FOOTER: ¹ [Author, Year — Title (URL)] OR [Source: system, date, n=]
DATA CALLOUT: [key number + stat context + source]
VISUAL: [chart type, axes, what to highlight]
[EXEC] [TECH] [MIXED] [CLIENT] [QUESTION] annotations
```

### 2. Speaker Notes
Narrative for each slide (conversational, not a script). SCQA opening written out in full.

### 3. Question Bank + Delivery Checklist
5 bridges for the hardest likely questions. Delivery checklist.

### 4. File Outputs (written to disk)
- `<deck-slug>.md`   — pandoc slide markdown source (edit this to update the deck)
- `<deck-slug>.html` — self-contained reveal.js browser presentation (open in any browser, no dependencies)
- `<deck-slug>.pptx` — PowerPoint file with speaker notes in presenter view

All three files are saved to the output directory specified by the user (default: project root).
The `.md` source is the single source of truth — regenerate `.html` and `.pptx` any time by re-running the Phase 6 pandoc commands.

---

## Examples

**Example 1: Executive churn presentation**
User says: "Create a 10-minute exec presentation on our new churn model results"
- Audience: EXEC → methodology goes to appendix, no stat language in body
- Structure: TRIANGLE (3 supporting findings: accuracy, revenue impact, implementation cost)
- Slide 2 title: "Deploying the churn model will retain $2.1M in annual revenue"
- All metrics translated: AUC 0.87 → "correctly identifies 87% of churners before they leave"

**Example 2: Technical deep-dive**
User says: "Build slides for a technical review of the feature engineering work"
- Audience: TECH → full methodology, confidence intervals, alternatives considered
- Structure: Q&A FUNNEL (broad question → deeper analysis → recommendation)
- Includes failure modes slide and model card in appendix

**Example 3: Mixed stakeholder update**
User says: "Prepare a project update presentation for the steering committee"
- Audience: MIXED → every technical term gets a gloss, 1 methodology slide in body
- Structure: NOSE (Now → Obstacle → Solution → Evaluation)
- `[MIXED]` annotations throughout for where to add plain-language glosses

---

## Common Issues

**User provides no data / numbers yet:**
Ask: "What is the main finding or metric you want to present? I need at least one anchor number to build the SCQA opening." Do not fabricate numbers.

**User asks to include an uncited benchmark:**
Respond: "I can't place that number on a slide without a citation — it opens you to challenges in the room. Do you have the source? If not, I'll log it as [OMITTED] and we can either find the source or rephrase as your internal result."

**User wants methodology on an exec slide:**
Respond: "For an exec audience, methodology weakens the story — it shifts focus from the decision to the process. I'll put it in appendix A3 so it's available if asked, and keep the body focused on the business case. Does that work?"

**User hasn't specified audience:**
Default to MIXED until confirmed. Flag it: "I've defaulted to MIXED audience (both executive and technical content). Confirm or correct so I gate content appropriately."

**HTML file looks blank or unstyled:**
The skill now uses `build_html.py` (not pandoc) for HTML — it generates a fully self-contained file with no CDN dependency. If the script fails, check: (1) Python 3 is available (`python3 --version`), (2) the path to `build_html.py` is correct, (3) the source `.md` file exists.

**User wants a custom PPTX theme:**
```bash
pandoc "<deck-slug>.md" -t pptx --slide-level=2 --reference-doc=<theme.pptx> -o "<deck-slug>.pptx"
```
The `<theme.pptx>` file must be a valid PPTX with at least one blank slide of each layout type. Create it in PowerPoint by saving an empty deck with the desired theme.
