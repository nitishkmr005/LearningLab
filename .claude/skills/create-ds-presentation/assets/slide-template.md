# Slide Template — Data Science Presentations

Used by the `create-ds-presentation` skill as the structural skeleton for every deck.

---

## Citation Format (apply to every slide with external facts)

```
Inline on slide:  [claim]¹   [claim]²
Footer (10pt):    ¹ Author et al., Year — Title (URL)
                  ² Organization, Year — Report title (URL)

For internal data:
Footer (10pt):    Source: [Dashboard/System name], [Date], n=[sample size]
```

Rule: if a number or claim has no citation ready, write `[CITATION NEEDED]` as a placeholder — do NOT put the number on the slide without it. Log it as `[OMITTED — no verifiable source: "claim"]` in the skill output.

---

## Audience Gate (apply before writing each slide)

Before writing any slide body, check:
```
AUDIENCE: [EXEC / TECH / MIXED / CLIENT]
INCLUDE:  [list of content types allowed per audience table in SKILL.md]
EXCLUDE → APPENDIX: [content to move, not delete]
```

---

## Mandatory Deck Sections

```
[A] TITLE SLIDE         (1 slide)
[B] OPENING: SCQA       (2-3 slides)
[C] BODY                (4-12 slides — varies by Think on Your Feet® structure)
[D] CLOSE / CALL TO ACTION  (1-2 slides)
[E] APPENDIX            (unlimited — methodology, raw data, FAQ)
```

---

## [A] Title Slide

```
--- SLIDE 1 ---
TITLE:   [Topic + Stakes in one line]
         e.g., "Reducing Mobile Churn: 18% Lift Opportunity in Q3"
SUBTITLE: [Audience + Date]
BODY:    [Presenter name, team, date]
NOTE:    No data here. The stakes go in the title so executives know
         why this matters before the deck begins.
```

---

## [B] Opening: SCQA (Slides 2-4)

### Situation slide
```
--- SLIDE 2 ---
TITLE:   [Undisputed fact that sets the stage]
         e.g., "Mobile users now represent 62% of our active base"
BODY:
  • [Fact 1 — recent, agreed-upon, no controversy]¹
  • [Fact 2 — context for scale or trend]²
DATA CALLOUT: [The one number that anchors the situation]
CITATION FOOTER:
  ¹ Source: [Internal dashboard / External report — URL — date]
  ² Source: [Internal dashboard / External report — URL — date]
VISUAL:  Trend line or single bar showing the current state
[EXEC]   Confirms the baseline they already know — creates agreement
[TECH]   Data source and freshness noted in footnote
[AUDIENCE NOTE] All audiences: situation slide always shown. Calibrate depth of facts to audience.
```

### Complication slide
```
--- SLIDE 3 ---
TITLE:   [What changed or threatens — creates urgency]
         e.g., "Mobile 30-day churn has risen 4pp in 90 days"
BODY:
  • [The problem stated with a number]
  • [The cost or risk if unaddressed]
DATA CALLOUT: [The alarming metric — make it big]
VISUAL:  Before/after bar or annotated time series with the inflection point marked
[EXEC]   This is the "why now" — they need to feel the urgency here
[TECH]   Segmentation logic explained in appendix slide A1
[QUESTION] "Is this seasonal?" → BRIDGE: "We controlled for seasonality — appendix A2"
```

### Answer slide (LEAD WITH THE RECOMMENDATION)
```
--- SLIDE 4 ---
TITLE:   [Your recommendation — specific, owned, time-bound]
         e.g., "Deploy churn model to target top-2000 at-risk users weekly"
BODY:
  • Expected impact: [metric + magnitude]
  • Resource needed: [cost / time / team]
  • Confidence: [how we know this works]
DATA CALLOUT: [Projected ROI or key metric]
VISUAL:  None needed — text clarity is paramount here
[EXEC]   This IS the slide. Everything else proves it.
[TECH]   Model details in Body section
[QUESTION] "How confident are you?" → BRIDGE ready (see Question Bank)
```

---

## [C] Body Slides — Pattern Templates

### Finding slide (one insight)
```
--- SLIDE N ---
AUDIENCE CHECK: [EXEC → show business impact only / TECH → show full stat / MIXED → show both / CLIENT → plain language only]
TITLE:   [Insight as a full sentence — not a label]
         e.g., "Users who skip onboarding step 3 are 3× more likely to churn"
BODY:
  • [What the data shows — the fact]¹
  • [Why it matters — the implication]
  • [What we did about it — the action] (only if applicable)
DATA CALLOUT:
  ┌──────────────────────────────┐
  │  KEY: 3× churn rate          │
  │  n = 28,400 users            │
  │  p < 0.001  [TECH/MIXED only]│
  └──────────────────────────────┘
CITATION FOOTER:
  ¹ Source: [Internal — system name, date range, n=size]
    OR: Author et al., Year — [Title](URL)
VISUAL:  [Chart type] — describe: axes, what's highlighted, color coding
         e.g., "Grouped bar chart: churned vs. retained, grouped by onboarding step
                completed. Step 3 bar pair should be highlighted in red."
[EXEC]   One-line business takeaway — no stat language
[TECH]   Statistical test used; effect size; any confounders controlled
[MIXED]  Plain-language gloss of any technical term in parentheses on slide
[CLIENT] Replace all internal system names with descriptive labels
[QUESTION] Anticipated challenge + Bridge answer
```

### Methodology slide (technical credibility — keep brief)
```
AUDIENCE CHECK:
  EXEC    → SKIP THIS SLIDE. Move all content to Appendix A3.
  TECH    → Include in body. Full detail welcome.
  MIXED   → 1 slide maximum. Plain language. Full detail in appendix.
  CLIENT  → SKIP THIS SLIDE. Move to appendix or omit entirely.

--- SLIDE N ---
TITLE:   "How we built it: [2-line summary of approach]"
BODY:
  • Data: [source, date range, size]¹
  • Model: [algorithm, why chosen over alternatives]
  • Validation: [holdout strategy, key metric + value]
DATA CALLOUT: AUC / F1 / RMSE — whichever metric the audience understands
             [EXEC/CLIENT: translate to business terms — see ds-presentation-guide.md]
CITATION FOOTER:
  ¹ Source: [Internal data system — date range — n=size]
    Benchmark comparison: [paper or report cited if claiming "best" or "state of the art"]
VISUAL:  Simple pipeline diagram (3-4 boxes max) or lift curve
[EXEC]   "The model was tested on unseen data — these results are real"
[TECH]   Full methodology in appendix A3
NOTE:    This slide earns trust. Keep it to 60 seconds — details go in appendix.
```

### Before / After slide (CONTRAST structure)
```
--- SLIDE N ---
TITLE:   "[New approach] outperforms [old approach] by [magnitude]"
BODY:
  LEFT COLUMN (Before):
    Method:  [Previous approach]
    Metric:  [Old value]
    Cost:    [Old cost / time]
  RIGHT COLUMN (After):
    Method:  [New approach]
    Metric:  [New value — highlighted]
    Cost:    [New cost / time]
DATA CALLOUT: [Delta — the improvement number, large font]
VISUAL:  Two-column layout; arrow pointing right; improvement in green
```

### Segmentation / drill-down slide
```
--- SLIDE N ---
TITLE:   "[Segment X] drives [Y]% of the total effect"
BODY:
  • [Top segment name]: [metric value]
  • [Second segment]: [metric value]
  • All others: [remainder]
DATA CALLOUT: [Top segment share of impact]
VISUAL:  Pareto chart or treemap — label the top 2 segments explicitly
[EXEC]   "Focus intervention here — 20% of users, 80% of the opportunity"
[TECH]   Segment definitions in appendix
```

---

## [D] Close / Call to Action

```
--- SLIDE (LAST-1) ---
TITLE:   "Recommended next steps"
BODY:
  • Action 1: [Specific, verb-first] — Owner: [Name] — By: [Date]
  • Action 2: [Specific, verb-first] — Owner: [Name] — By: [Date]
  • Action 3: [Specific, verb-first] — Owner: [Name] — By: [Date]
NOTE:    Three max. If you have more, put them in a table in the appendix.
[EXEC]   They need to make a decision or assign an owner today.

--- SLIDE (LAST) ---
TITLE:   "Questions?"
BODY:    [Key finding restated in one line]
         [Contact / email]
NOTE:    Keep the recommendation visible while questions happen.
         Do NOT use a blank "Thank You" slide — it wastes real estate.
```

---

## [E] Appendix Slides

Label every appendix slide with a letter code referenced from the main deck.

```
--- APPENDIX A1 ---
TITLE:   "Segmentation methodology"
[full detail]

--- APPENDIX A2 ---
TITLE:   "Seasonality control — YoY comparison"
[supporting evidence for complication claim]

--- APPENDIX A3 ---
TITLE:   "Model card — feature set, hyperparameters, training data"
[full technical spec]

--- APPENDIX A4 ---
TITLE:   "Alternative models evaluated"
[comparison table: algo, AUC, latency, interpretability]

--- APPENDIX FAQ ---
TITLE:   "Frequently asked questions"
[pre-written Q&A for questions not addressed in main deck]

--- APPENDIX CITATIONS ---
TITLE:   "Full citations"
FORMAT:
  [¹] Author(s), Year. "Title." Source/Journal. URL
  [²] Organization, Year. "Report title." URL
  [Internal-1] System name — date range — owner team
NOTE:    This slide is always present if the deck contains any external citations.
         It is the source of truth — every footnote number in the main deck
         must appear here with a complete reference.
```

---

## Visual Design Rules

| Element | Rule |
|---------|------|
| Slide title | Full sentence, 24-28pt bold, max 2 lines |
| Body text | 18-20pt, max 5 lines, max 3 bullets |
| Data callout | 48-60pt for the key number; 14-16pt for context |
| Chart labels | Always label data points directly — no legend if possible |
| Color | One accent color for "the thing that matters"; grey for everything else |
| White space | Leave 20% of slide empty — resist filling it |
| Source footnote | "Source: [system], [date], n=[size]" — bottom-left, 10pt |

---

## Anti-Patterns to Avoid

| Anti-pattern | Why it fails | Fix |
|---|---|---|
| Slide title = "Results" | Forces audience to read body to understand | "Model lifts revenue 18% in A/B test" |
| 8 bullets per slide | Audience reads ahead, stops listening | 3 bullets max; split if needed |
| Conclusion slide at the end | Executive left before you got there | Recommendation on slide 2 (SCQA Answer) |
| Raw output tables | Signals you didn't interpret the data | Show the insight, offer table in appendix |
| "As you can see..." | Condescending; also admits a bad chart | Make the chart self-evident OR fix it |
| Acronyms unexplained | Alienates non-technical stakeholders | Spell out + define on first use |
| No confidence intervals | Looks overconfident; invites attack | Always add CI or sample size to key claims |
| Number without a source | Invites "where does that come from?" and destroys trust | Add inline footnote or `Source:` footer — or remove the number |
| "Studies show" / "Research suggests" | Vague; signals you don't have the actual source | Name the study, year, and link — or cut the claim |
| Technical metric shown to exec | Forces audience to convert it to meaning themselves | Translate: AUC → "ranks at-risk users correctly X% of the time" |
| Exec detail shown to technical peer | Feels shallow; peer disengages | Add methodology and rigor signals for tech audiences |
