# Data Science Presentation Guide

Supporting reference for `create-ds-presentation`. Covers storytelling principles, audience calibration, and DS-specific communication patterns.

---

## Core Principle: Lead With the Answer

Most data scientists present like detectives — showing clues, building to a reveal. Executives think like judges — they want the verdict first, evidence second.

**Bottom-up (how you thought):**
> "We collected data → cleaned it → ran 4 models → compared them → XGBoost won → AUC 0.87 → therefore deploy"

**Top-down (how to present):**
> "We should deploy the XGBoost model (AUC 0.87). Here's the evidence."

Rule: **You think bottom-up, you present top-down.** (Minto, Pyramid Principle)

---

## The Three-Layer Audience Problem

Data science presentations almost always have a mixed room. Structure your deck to serve all three:

| Layer | Who | What they need | Where to put it |
|-------|-----|----------------|-----------------|
| **Strategic** | VP, Director, Exec | Recommendation + business impact | Slides 1-4 (SCQA block) |
| **Tactical** | Manager, PM | Implementation plan, tradeoffs, risks | Body slides |
| **Technical** | Data Scientist, Engineer | Methodology, model details, edge cases | Appendix |

Key technique: **answer the strategic question on slide 2, always.** Technical colleagues will follow you through the body; executives may drop off after slide 5 — don't make them wait.

---

## Data Storytelling: The Narrative Arc

### The three-act structure for DS presentations

**Act 1 — The World Before (Problem)**
- What is the current state? (Situation)
- Why does it matter? (Stakes)
- What is changing or breaking? (Complication)
- Rule: Use data to show the problem, not just describe it.

**Act 2 — The Investigation (Insight)**
- What did the data reveal?
- What was surprising or non-obvious?
- What are the key segments or patterns?
- Rule: One insight = one slide. Lead with the insight in the title.

**Act 3 — The World After (Recommendation)**
- What should change?
- What will the outcome be?
- Who does what by when?
- Rule: Make it specific enough that someone can start tomorrow.

---

## Insight vs. Observation

This is the most common DS presentation failure: reporting observations instead of insights.

| Observation (weak) | Insight (strong) |
|---------------------|------------------|
| "Churn is highest among mobile users" | "Mobile users who skip onboarding step 3 churn at 3× the rate — fixing that one step recovers $800K" |
| "The model has AUC 0.87" | "The model correctly flags 7 in 10 churners before they leave, vs. 4 in 10 with the current rules" |
| "Feature importance shows recency is top" | "Days since last purchase alone predicts churn better than all demographic features combined" |
| "We tested 4 algorithms" | "XGBoost matched neural network accuracy at 40× lower inference cost" |

**The test:** replace "observation" with "so what?" — the answer IS the insight.

---

## Slide Title Formula

Every slide title should be a **declarative sentence** stating the insight, not a label naming the topic.

```
Pattern: [Subject] [verb] [specific result/implication]

Weak labels:        Strong action titles:
"Revenue Analysis"  →  "Revenue declined 12% — driven entirely by enterprise segment"
"Model Comparison"  →  "XGBoost matches transformer accuracy at 40× lower cost"
"User Segments"     →  "Three segments account for 85% of projected impact"
"Next Steps"        →  "Two decisions needed today to hit Q3 launch"
```

---

## Chart Selection for DS Presentations

| What you want to show | Chart type | Anti-pattern to avoid |
|-----------------------|------------|-----------------------|
| Change over time | Line chart | Bar chart with 50 bars |
| Part-to-whole | Treemap or stacked bar (2-3 segments) | Pie chart with >3 slices |
| Comparison across categories | Grouped bar (max 3 groups) | Table of numbers |
| Distribution | Histogram or box plot | Reporting only the mean |
| Correlation | Scatter with regression line | Correlation matrix heat map |
| Ranking | Horizontal bar, sorted | Unsorted table |
| Model performance | Lift curve or precision-recall | ROC curve alone (unintuitive) |
| Feature importance | Horizontal bar (top 10) | Full feature list |
| A/B test result | Annotated bar with CI whiskers | Raw p-value without effect size |

**Golden rule:** Label the data directly on the chart. Legends require eye movement. Eye movement breaks attention.

---

## Handling Numbers for Non-Technical Audiences

### Translate statistics into business language

| Technical | Business translation |
|-----------|---------------------|
| AUC = 0.87 | "Our model ranks at-risk users correctly 87% of the time" |
| p < 0.01 | "Less than 1% chance this result is due to random noise" |
| Precision = 0.72 | "When we flag a user as at-risk, we're right 72% of the time" |
| RMSE = 45 | "Our revenue forecast is off by $45 on average" |
| Lift = 2.4× | "Targeting with the model gives 2.4× more conversions than random outreach" |

### The "so if 1000 users..." test
Convert relative numbers to absolute examples at scale:
- "18% churn reduction" → "18% churn reduction means 1,800 fewer churned users per quarter"
- "AUC improvement from 0.81 to 0.87" → "We catch 60 more churners per 1,000 at-risk users"

---

## Executive Presentation Patterns

### The Elevator Test
Slide 2 alone must answer: **What are you recommending and why should I care?**
If it can't, restructure.

### The Proxy Metric Problem
Executives don't care about AUC. They care about revenue, cost, risk. Always chain model metrics to business metrics:

```
Better model → Better precision → Fewer wasted interventions → $X saved
Better recall → More churners caught → $Y retained
Faster inference → Lower infra cost → $Z/month saved
```

### Handling "just tell me the number"
If an exec interrupts and asks "just give me the bottom line" — they're asking you to jump to the SCQA Answer slide immediately. Be ready to say in 30 seconds: "We should [action]. It will [impact]. We need [resource]. Decision needed by [date]."

---

## Technical Peer Presentation Patterns

### Rigor signals
Technical peers look for these — include them or lose credibility:
- Train/validation/test split strategy
- Why this model family over others (brief — details in appendix)
- How you handled class imbalance / data leakage / distribution shift
- Confidence intervals on all key metrics (not just point estimates)
- What would make you revise the recommendation (falsifiability)

### The failure modes slide
Most impressive DS presentations include one slide: "What could go wrong."
- Distribution shift
- Feature unavailability at inference
- Feedback loops
- Edge cases the model handles poorly

This signals production-readiness thinking, not just research.

---

## Timing and Pacing

| Presentation length | Target slide count | Notes |
|--------------------|-------------------|-------|
| 5-minute update | 4-6 slides | SCQA + 2 findings + CTA |
| 15-minute presentation | 10-15 slides | Full arc, 1 methodology slide |
| 30-minute deep dive | 20-25 slides | Full arc + 3-4 body findings + Q&A time built in |
| 60-minute workshop | 30-40 slides | Include hands-on / discussion slides |

Rule: **1 content slide per minute.** Title + appendix slides don't count.

Never use filler slides ("Agenda", "Background", "Today we will cover...") — they delay the value and signal the presenter doesn't trust the audience.

Exception: A 30-second roadmap slide ("Three findings, then a recommendation") is acceptable for presentations over 20 minutes.

---

## Pre-mortem: The 5 Questions to Ask Before Presenting

1. **If they only remember one thing, what is it?** — This should be your slide 2 title.
2. **What is the hardest objection?** — Pre-empt it in the complication slide or the methodology slide.
3. **Who in the room is most likely to push back?** — Prepare a BRIDGE for them specifically.
4. **What decision are you asking for?** — If you can't name it, your close is too vague.
5. **What would you say if you had 60 seconds?** — Rehearse the SCQA out loud before any presentation.
