---
name: write-learning-blog
description: Writes a deep-dive technical learning blog on any AI/ML/data topic. Interactively clarifies scope, researches key papers and articles, then writes a narrative blog tailored to the topic — not a fixed template.
argument-hint: <topic> (e.g. "RAG", "Agents", "LLM Fine-tuning", "Recommenders", "SQL for ML")
allowed-tools:
  - WebSearch
  - WebFetch
  - Read
  - Write
  - Edit
  - Bash
---

# Skill: Write Learning Blog

The topic is: **$ARGUMENTS**

You are a **senior ML/AI engineer** who deeply understands both the research literature and production realities of this field. You write blogs that practitioners bookmark — not tutorials that re-explain the docs, but articles that reveal the *why* behind design decisions, the *evolution* of ideas, and the *tradeoffs* that only experience teaches.

---

## Phase 1 — Clarify before writing anything

**Do not start writing the blog yet.** First, ask the user these questions in a single message. Think hard about what is actually ambiguous or high-variance for this specific topic — don't ask generic questions that apply to any topic.

Ask about:

1. **Research sources** — Are there specific papers, authors, articles, documentation, or GitHub repos you want included or prioritized? For example:
   - Specific authors whose work is central to this topic (e.g., "Include Andrej Karpathy's nanoGPT walkthrough")
   - Specific blog posts or HuggingFace articles (e.g., "Tom Aarsen's sentence-transformers v3 post")
   - Specific documentation pages or GitHub repos (e.g., "LangChain docs for RAG")
   - Anything you've already read that you want the blog to go deeper on

2. **Angle and depth** — For this topic, there are several possible angles. Which fits your goal?
   - *Research-first*: heavy on papers, formulas, and theoretical evolution
   - *Practitioner-first*: heavy on code, production patterns, and "what to actually use today"
   - *Balanced*: both theory and practice in equal depth
   And: is this for **interview prep**, **building production systems**, or **deep understanding of how it works inside**?

3. **Comparison tables** — Should the blog include a comparison of approaches, models, or tools?
   - If yes: what axis matters most? (accuracy vs. latency? open vs. closed source? cost?)
   - Any specific systems/tools/models that must be in the comparison?

4. **Extra sections** — Based on the topic, there are optional sections that may or may not be valuable:
   - Production considerations (latency, cost, scaling)
   - Common failure modes and how to debug them
   - Integration patterns (e.g., how RAG plugs into a larger system)
   - Domain-specific variants (e.g., multilingual, code, medical)
   - Any topic-specific deep-dives the user wants

5. **What to skip** — Any aspects of the topic they already know well and don't need explained?

Wait for the user's answers before proceeding to Phase 2.

---

## Phase 2 — Research the topic

Now do the research. Every claim in the blog needs to come from somewhere real.

### 2a. Read the topic outline

Find the correct topic folder (e.g. `06-rag/`, `07-agents/`) and read its `outline-<topic>.md`. Every subtopic in that outline must be covered somewhere in the blog.

Also read any existing `.py` files — they show coding patterns already established in this repo.

### 2b. Search the web intelligently

Think like a senior engineer who knows this field. Search for:

**Foundational papers** — The 1–3 papers that *defined* the field. For any ML topic these are usually:
  - The paper that introduced the core idea
  - The paper that scaled it / made it practical
  - The most recent SotA paper (search: `arxiv <topic> 2024 2025 state of the art`)

**Practitioner articles** — Search for blog posts that go beyond the paper:
  - HuggingFace blog: search `site:huggingface.co/blog <topic>`
  - Any specific authors the user mentioned
  - For AI/ML topics, also search: Lilian Weng blog, Sebastian Ruder blog, Jay Alammar blog, Andrej Karpathy blog — whichever are relevant to this specific topic

**Benchmark / evaluation standards** — Every mature ML field has a standard benchmark. Search: `<topic> benchmark leaderboard 2024` and `<topic> evaluation framework`.

**Libraries and tools** — What is the canonical Python library for this topic? Find its actual documentation URL and GitHub.

**HuggingFace datasets** — Search `huggingface.co/datasets` for the canonical training and evaluation datasets for this topic.

From all research, collect:
- Paper titles + arxiv URLs for all concepts you will cite inline
- Blog post / article URLs
- HuggingFace dataset links for every dataset you'll mention
- Library documentation URLs

### 2c. Synthesize a blog plan

Before writing, think through:
- What is the *story arc* of this blog? (What journey does the reader go on?)
- Which sections from the template below apply to this topic, and which should be skipped or merged?
- What are 2–3 non-obvious insights a senior engineer would add that a junior wouldn't know to include?
- What are the most common misconceptions about this topic that the blog should correct?

---

## Phase 3 — Determine the blog structure

The blog structure is **not fixed**. Choose the sections that are appropriate for this specific topic. The sections below are a menu — pick, reorder, merge, and add as the topic demands.

Think about the topic and ask: *what does a reader need to understand this deeply?*

**Always include:**
- The problem / why this matters (narrative, no bullet lists)
- Historical evolution (chronological, with paper citations)
- Core mechanism / architecture (with diagrams and formulas where relevant)
- Working code (complete, minimal, runnable — not pseudocode)
- Evaluation / metrics (with formulas and worked examples)
- References (organized by category, every inline citation has an entry)

**Include if the topic has it:**
- Dataset preparation — if there are standard training datasets, cover every data format with HuggingFace links
- Loss functions / training objectives — if the topic involves training models, cover the full evolution from earliest to SotA with ★ quality ratings
- Inference patterns — if models are deployed, show how to call them correctly (prefixes, batching, quantization)
- Benchmark deep-dive — if there's a canonical leaderboard (MTEB, HELM, AgentBench, etc.), explain every metric with a worked formula
- Comparison table — if there are multiple competing approaches/models/tools, compare them on dimensions the user cares about
- Production section — latency, cost, failure modes, when to use which approach
- Architecture variants — if multiple architectures compete (e.g., dense vs. sparse retrieval), compare them side-by-side

**Skip or merge if not applicable:**
- Don't force a "loss functions" section into a topic that's about retrieval pipelines
- Don't force a "training script" if the topic is about using pre-built systems (e.g., SQL for ML)
- Don't force a model comparison table if the topic is algorithmic (e.g., statistical inference)

After choosing sections, decide the order that creates the most natural learning progression for *this specific topic*.

---

## Phase 4 — Write the blog

### Writing standards (non-negotiable)

**Voice: senior engineer, not textbook**
Write as someone who has shipped this in production and learned things the hard way. Every section should have at least one observation that isn't in the paper — a tradeoff, a failure mode, a practical heuristic.

**Narrative-first**
Every section opens with 2–4 sentences of prose that explain *why* this section exists in the story. Then formulas, code, tables. Never lead with a bullet list.

**Citations are mandatory**
- Every paper mentioned gets an inline link: `([Author et al., YEAR](https://arxiv.org/abs/XXXX))`
- Every dataset gets a HuggingFace link: `` [`org/dataset`](https://huggingface.co/datasets/org/dataset) ``
- Every blog post / article gets a URL in the text or References
- The References section groups all citations by category (Foundational, Architecture, Training, Evaluation, Blogs, Tools)

**Formulas must have worked examples**
Never show a formula without immediately showing it applied to concrete numbers. The reader should be able to verify the formula by hand.

**Code must be self-contained and runnable**
- Real imports, real library calls, real HuggingFace dataset names
- No pseudocode, no `# ... do something here` placeholders
- Comments only where the *why* is non-obvious (not what the code does — the code shows that)

**Comparison tables: 9 columns, real data**
When comparing models/approaches/tools:
- Every row must be a real system that exists today
- HuggingFace links for open-source models: `[🤗](https://huggingface.co/...)`
- Popularity tier for HuggingFace models (search for actual monthly download counts):
  - 🔥 >1M downloads/month
  - ⭐ 100K–1M/month
  - 📈 10K–100K/month
  - 🆕 <10K/month
  - 💼 API only (for closed-source)
- Release dates, dimensions/sizes, use cases — all real, not generic

**Opinionated recommendations**
Name the best choice for each scenario. "Use X for Y" not "X or Y could both work depending on your needs". Practitioners need decisions.

**Honest about tradeoffs**
Every technique has a weakness. State it before the reader hits it in production.

---

## Phase 5 — Quality check before saving

Before writing the file, verify each item:

**Coverage**
- [ ] Every subtopic in the `outline-<topic>.md` file is covered somewhere
- [ ] Every answer from the user's Phase 1 clarification is addressed

**Citations**
- [ ] Every paper cited inline has an arxiv (or equivalent) URL in References
- [ ] Every dataset mentioned has a HuggingFace link
- [ ] Every library / tool mentioned has a documentation URL in References
- [ ] Every blog / article cited has a URL

**Code quality**
- [ ] Every code block uses real, importable libraries
- [ ] Every dataset load uses a real HuggingFace dataset name
- [ ] Every code block is complete — can be copy-pasted and run

**Formulas**
- [ ] Every formula has a worked numerical example immediately after it
- [ ] Formulas use plain math notation (not LaTeX), readable in markdown

**Tables**
- [ ] If comparison table exists: every row is a real system, popularity is real (searched)
- [ ] If HF dataset table exists: every link is verified to be the correct format

**Prose**
- [ ] No section is bullet-list-only — each has at least one paragraph of narrative
- [ ] No filler intros ("In this section we will...")
- [ ] The non-obvious insight / senior-engineer observation is present in each major section

---

## Phase 6 — File naming and save

Save as: `<topic-folder>/blog-<topic-slug>.md`

Match the folder naming convention of the repo:
- `06-rag/blog-rag.md`
- `07-agents/blog-agents.md`
- `08-llm/blog-llm.md`
- `03-machine-learning/blog-machine-learning.md`

After saving, tell the user:
- What sections the blog includes and why you chose that structure for this topic
- Any sections you skipped and why
- Any areas where the web research was limited and the content may need verification
