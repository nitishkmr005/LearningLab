# Pandoc Slide Markdown Format

Reference for Phase 3 of the `create-ds-presentation` skill.
This file defines the exact markdown format that pandoc uses to compile `.html` (reveal.js) and `.pptx` (PowerPoint) from a single source file.

---

## File Header (YAML Front Matter)

Every pandoc slide file must start with YAML front matter:

```markdown
---
title: "Full Deck Title"
subtitle: "Audience · Date"
author: "Presenter Name"
date: "YYYY-MM-DD"
---
```

---

## Slide Separators

| Element | Meaning |
|---------|---------|
| `##` heading | Creates a new content slide. The heading text = the slide title. Must be an action title (full sentence stating the insight). |
| `#` heading | Creates a section divider slide (e.g., "Domain 1: Deep Learning"). Use sparingly to signal major topic transitions. |
| `---` horizontal rule | Forces a slide break without a title (use for blank transition slides or when `##` is already used). |

**`--slide-level=2` is always passed to pandoc.** This means `##` headings are the slide title level. Do not use a different heading level for slide titles.

---

## Slide Body Content

### Bullets (max 3 per slide)

```markdown
## Sigmoid max derivative of 0.25 causes vanishing gradients in deep networks

- σ'(x) = σ(x)(1−σ(x)) — max at x=0: 0.5 × 0.5 = 0.25
- Stacking 8 sigmoid layers: gradient scale = 0.25⁸ = 1.5 × 10⁻⁵ (near zero)
- ReLU fix: f'(x) = 1 for x > 0 — gradient passes through intact at any depth
```

### Tables (decision tables, comparison tables)

```markdown
## Initialization rule: He for ReLU, Xavier for Sigmoid and Tanh

| Activation | Initialization | Variance Formula |
|-----------|----------------|-----------------|
| ReLU      | He (Kaiming)   | 2 / n_in        |
| Sigmoid   | Xavier (Glorot)| 2 / (n_in + n_out) |
| Tanh      | Xavier (Glorot)| 2 / (n_in + n_out) |
```

### Data callout box (blockquote)

Use a blockquote for the key number callout. Pandoc renders it as a styled block in both HTML and PPTX:

```markdown
> **KEY METRIC: AUC-PR = 0.35** — Fraud model struggles despite AUC-ROC = 0.98
> n = 2.1M transactions · Source: 19-interview/Target/4.model-evaluation-interview.md
```

### Inline citations (footnotes)

```markdown
Sigmoid's max derivative is 0.25 at x=0[^1], causing exponential gradient decay.

[^1]: Source: 19-interview/Target/1.deep-learning-interview.md — Q1, Q3
```

---

## Speaker Notes

Use fenced divs. Notes appear in PPTX presenter view and reveal.js speaker view (`S` key):

```markdown
## Slide Title Here

- Bullet 1
- Bullet 2

::: notes
The conversational narrative for this slide. Not a script — talk-track level.
For the SCQA opening slide, write the full 60-second cold open here.
:::
```

---

## Appendix Section

Place appendix slides after the main deck. Use a `#` section header to signal the break, then `##` per appendix slide:

```markdown
# Appendix

## A1 — Activation Function Decision Table

| Use Case                  | Activation | Why                              |
|---------------------------|------------|----------------------------------|
| Hidden layers (MLP, CNN)  | ReLU       | No vanishing gradient            |
| Transformer hidden layers | GELU       | Smooth probabilistic gating      |
| Binary output             | Sigmoid    | Outputs P(class=1) ∈ (0,1)      |

::: notes
Reference slide — show if asked about activation choices in Q&A.
:::
```

---

## Complete Minimal Example

```markdown
---
title: "DS Interview Mastery"
subtitle: "TECH Audience · June 2026"
author: "Nitish Harsoor"
date: "2026-06-06"
---

## 5 domains, 103 questions — most candidates fail at derivation depth, not vocabulary

- Every DS interview tests: Deep Learning, ML Algorithms, Statistics, Model Evaluation, Feature Engineering
- Interviewers probe 3 layers: vocabulary → derivation → trade-off
- This deck covers all 3 layers for every high-frequency question

::: notes
SCQA opening cold: "Every DS interview I've seen probes the same five areas.
The failure mode is always the same — candidates know the vocabulary but cannot
go one level deeper. Lead with intuition, walk equations step-by-step with
numbers, then explain why this choice and not another."
:::

---

## Sigmoid's max derivative of 0.25 is the root cause of vanishing gradients

- σ'(x) = σ(x)(1−σ(x)) — max at x=0: 0.5 × 0.5 = 0.25
- 8 sigmoid layers → gradient scale = 0.25⁸ = 1.5 × 10⁻⁵ (near zero)[^1]
- ReLU fix: f'(x) = 1 for x > 0 — passes through intact at any depth

> **GRADIENT AT 8 LAYERS: 1.5 × 10⁻⁵** (sigmoid) vs **1.0** (ReLU)
> Source: 19-interview/Target/1.deep-learning-interview.md — Q3, Q4

[^1]: Source: 19-interview/Target/1.deep-learning-interview.md

::: notes
The question has 3 layers. Layer 1 = vocabulary answer. Layer 2 = show the math:
0.25⁸ = 1.5e-5. Layer 3 = trade-off: ReLU fixes it but dying ReLU is the new
problem, which is why Leaky ReLU and GELU exist.
:::

# Appendix

## A1 — Activation function decision table

| Use Case                  | Activation | Why                         |
|---------------------------|------------|-----------------------------|
| Hidden layers (MLP, CNN)  | ReLU       | No vanishing gradient       |
| Transformer hidden layers | GELU       | Smooth probabilistic gating |
| Binary output             | Sigmoid    | P(class=1) ∈ (0,1)         |
```

---

## Pandoc Commands (Phase 6)

### HTML (reveal.js)
```bash
pandoc "<deck-slug>.md" \
  -t revealjs \
  --standalone \
  --slide-level=2 \
  -V theme=night \
  -V transition=fade \
  -V controls=true \
  -V progress=true \
  -V center=true \
  --embed-resources \
  -o "<deck-slug>.html"
```

### PPTX (PowerPoint)
```bash
pandoc "<deck-slug>.md" \
  -t pptx \
  --slide-level=2 \
  -o "<deck-slug>.pptx"
```

### Custom PPTX theme
```bash
pandoc "<deck-slug>.md" \
  -t pptx \
  --slide-level=2 \
  --reference-doc=<theme.pptx> \
  -o "<deck-slug>.pptx"
```
The `<theme.pptx>` must be a valid PPTX with at least one blank slide per layout type.

---

## Troubleshooting

**`--embed-resources` fails (offline / CDN blocked):**
Omit the flag — the HTML will require internet to load reveal.js when opened.

**Tables don't render in PPTX:**
Ensure the table has a header row (the `|---|` separator line is required by pandoc's table parser).

**Speaker notes missing from PPTX:**
Use the exact `:::notes` / `:::` syntax — no extra spaces before `notes`.

**Slide count differs between HTML and PPTX:**
`#` section headers create a section divider slide in revealjs but a title-only layout in PPTX. This is expected — slide counts can differ by the number of `#` headers used.
