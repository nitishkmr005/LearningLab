# LearningLab

An exhaustive, interview-ready learning system for data scientists covering SQL through advanced LLMs, vision, speech, and RL.

## Goal

Two complementary deliverables per topic:

1. **Interview prep** — runnable Python scripts (`XX-<subtopic>.py`) that demonstrate one concept each, minimal and self-contained.
2. **Literature survey blog** — a single-file deep-dive that is the one-stop reference for that topic: history, theory, formulas, code, benchmarks, and citations.

## Folder structure

```
<NN>-<topic>/
  outline-<topic>.md       # exhaustive learning roadmap — source of truth for subtopics
  blog-<topic>.md          # one-stop literature survey blog
  XX-<subtopic-name>.py    # one concept per file, runnable as-is
```

Folder numbers are sequential. Adding a new topic inserts it in order and renumbers downstream folders.

## Workflows

### Add a runnable script
1. Open `outline-<topic>.md` and pick the subtopic number.
2. Create `<NN>-<subtopic-name>.py` — real imports, one concept, runnable without extra setup.
3. No placeholder comments. No pseudocode.

### Write or update a blog
Use the `/write-learning-blog` skill. It handles research, structure, and citations automatically.

Blog requirements (enforced by the skill):
- Table of contents at the top.
- Every claim cited inline: `([Author et al., YEAR](https://arxiv.org/abs/XXXX))`.
- All citations must link to authoritative sources — papers on arXiv, or official docs/blogs from: Anthropic, Hugging Face, NVIDIA, OpenAI, PyTorch, Google DeepMind, Microsoft Research, Cohere, Unsloth, SBERT, Snowflake, or equivalent orgs.
- References section at the end, organised by category.
- Save as `blog-<topic-slug>.md` in the topic folder.

### Add a new topic folder
1. Create `<NN>-<topic>/outline-<topic>.md` following the format of existing outlines (numbered `## 01 —` sections, 2-3 reference links each).
2. Update `README.md` table.

## Conventions

- Outline section headers: `## NN — Title` (zero-padded two digits).
- Script filenames: `XX-kebab-case.py` where XX matches the outline section number.
- Blog filenames: `blog-<topic-slug>.md` (no numbers, just the topic name).
- No notebooks. Scripts only.
