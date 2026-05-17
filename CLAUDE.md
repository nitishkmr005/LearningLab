# LearningLab

An exhaustive, interview-ready learning system covering SQL through advanced LLMs, vision, speech, and RL — targeted at four roles:

- **Data Scientist** — statistics, experimentation, classical ML, SQL, feature engineering
- **ML Engineer** — model training, serving, MLOps, data pipelines, distributed systems
- **AI Engineer** — LLMs, RAG, agents, prompt engineering, evaluation, fine-tuning
- **GenAI Engineer** — multimodal models, inference optimization, tool use, reasoning models, voice/vision agents

Every topic is chosen because it appears in interviews, production systems, or both for at least one of these roles.

## Goal

Three complementary deliverables per topic:

1. **Interview prep** — the blog serves as the one-stop literature survey covering history, theory, trade-offs, and everything an interviewer probes across DS / ML / AI / GenAI roles.
2. **Literature survey blog** — a single-file deep-dive: history, theory, formulas, code, benchmarks, and citations from authoritative sources.
3. **Learning snippets** — byte-sized, minimal scripts under `scripts/` to understand and learn advanced topics hands-on. Each script isolates one concept, runs in the terminal, and teaches through its output.

## Folder structure

```
<NN>-<topic>/
  outline-<topic>.md       # roadmap while the blog doesn't exist yet — delete once blog is written
  blog-<topic>.md          # one-stop literature survey blog
  scripts/
    01-<subtopic-name>.py  # one concept per file, runnable as-is
    02-<subtopic-name>.py
    ...
```

Folder numbers are sequential. Adding a new topic inserts it in order and renumbers downstream folders.

## Workflows

### Add a learning snippet
1. Open `outline-<topic>.md` (or `blog-<topic>.md` if the blog exists) and pick the subtopic number.
2. Create `scripts/<NN>-<subtopic-name>.py` — real imports, one concept, runnable without extra setup.
3. Every line must have an inline comment explaining what it does.
4. Running the script must print useful output to the terminal so the learner sees the concept in action.
5. No placeholder comments. No pseudocode.

### Write or update a blog
Use the `/write-learning-blog` skill. It handles research, structure, and citations automatically.

Blog requirements (enforced by the skill):
- Table of contents at the top.
- Every claim cited inline: `([Author et al., YEAR](https://arxiv.org/abs/XXXX))`.
- All citations must link to authoritative sources — papers on arXiv, or substack/medium articles or official docs/blogs from: Anthropic, Hugging Face, NVIDIA, OpenAI, PyTorch, Google DeepMind, Microsoft Research, Cohere, Unsloth, SBERT, Snowflake, or equivalent orgs.
- References section at the end, organised by category.
- Save as `blog-<topic-slug>.md` in the topic folder.
- Delete `outline-<topic>.md` after the blog is saved.

### Add a new topic folder
1. Create `<NN>-<topic>/outline-<topic>.md` following the format of existing outlines (numbered `## 01 —` sections, 2-3 reference links each).
2. Update `README.md` table.

## Outline quality standard

Every section in an outline should note the **most popular production approach**, the **main alternatives**, and the **SotA model or technique** where they differ. This makes the outline serve as a quick reference for choosing tools in interviews and production.

Format guidance for each section description:
- Lead with the canonical/most-popular approach in industry (e.g., "XGBoost most popular for structured data; LightGBM faster for large datasets; CatBoost best for categoricals").
- Call out the SotA if it differs from what is commonly deployed (e.g., "research SotA: FT-Transformer; production default: LightGBM").
- Link to: arXiv papers, official library docs, and the most-referred developer resources (HuggingFace, PyTorch docs, NVIDIA, Anthropic, Snowflake, authoritative blog posts from practitioners).

This standard applies when writing new outlines and when adding new sections to existing ones. Existing sections should be upgraded when the blog for that topic is written.

## Conventions

- Outline section headers: `## NN — Title` (zero-padded two digits).
- Script filenames: `NN-kebab-case.py` inside `scripts/`, where NN matches the subtopic number.
- Blog filenames: `blog-<topic-slug>.md` in the topic root (no numbers).
- No notebooks. Scripts only.
