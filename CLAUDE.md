# LearningLab

An exhaustive, interview-ready learning system for data scientists covering SQL through advanced LLMs, vision, speech, and RL.

## Goal

Three complementary deliverables per topic:

1. **Interview prep** — the blog serves as the one-stop literature survey covering history, theory, trade-offs, and everything an interviewer can probe.
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
- All citations must link to authoritative sources — papers on arXiv, or official docs/blogs from: Anthropic, Hugging Face, NVIDIA, OpenAI, PyTorch, Google DeepMind, Microsoft Research, Cohere, Unsloth, SBERT, Snowflake, or equivalent orgs.
- References section at the end, organised by category.
- Save as `blog-<topic-slug>.md` in the topic folder.
- Delete `outline-<topic>.md` after the blog is saved.

### Add a new topic folder
1. Create `<NN>-<topic>/outline-<topic>.md` following the format of existing outlines (numbered `## 01 —` sections, 2-3 reference links each).
2. Update `README.md` table.

## Conventions

- Outline section headers: `## NN — Title` (zero-padded two digits).
- Script filenames: `NN-kebab-case.py` inside `scripts/`, where NN matches the subtopic number.
- Blog filenames: `blog-<topic-slug>.md` in the topic root (no numbers).
- No notebooks. Scripts only.
