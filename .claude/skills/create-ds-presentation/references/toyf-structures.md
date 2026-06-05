# Think on Your Feet® Communication Structures

Reference for the `create-ds-presentation` skill. Select the structure that matches the presentation goal, then apply it to the body of the deck.

Source: McLuhan & Davies Communications — Think on Your Feet® workshop (thinkonyourfeet.com / mdctraining.com)

---

## The Three Communication Modes

Every Think on Your Feet® structure falls into one of three modes:

| Mode | Use when | Effect on audience |
|------|----------|--------------------|
| **INFORMATIVE** | Sharing facts, findings, updates | Understands and remembers |
| **ASSERTIVE** | Persuading, recommending, justifying | Agrees and acts |
| **VISUAL** | Making abstract ideas concrete | Sees and connects |

---

## 10 Core Structures

### 1. PREP — Point → Reason → Example → Point
**Mode:** Assertive  
**Use for:** Answering a direct question, justifying a recommendation, defending a number.

```
Point:   State your position clearly in one sentence.
Reason:  Give the single strongest reason.
Example: Concrete data, anecdote, or result that proves it.
Point:   Restate the position with more confidence.
```

**DS application:** "Should we deploy this model?" — Answer: Yes, here's why, here's the evidence, here's the confirmation.  
**Slide shape:** 1 title slide + 1 evidence slide + 1 close slide.

---

### 2. SCQA — Situation → Complication → Question → Answer
**Mode:** Informative + Assertive  
**Use for:** Opening any presentation, framing a business problem, setting up a recommendation.

```
Situation:    What is true today? (facts everyone agrees on)
Complication: What changed, threatens, or creates urgency?
Question:     What does this force us to ask?
Answer:       Your recommendation / finding — stated NOW.
```

**DS application:** "Churn is 12% (S). We just lost our top cohort — mobile users (C). How do we retain them? (Q). Build a churn propensity model targeting mobile users in the first 30 days (A)."  
**Slide shape:** This is always the opening structure regardless of body structure chosen.

---

### 3. NOSE — Now → Obstacle → Solution → Evaluation
**Mode:** Assertive  
**Use for:** Problem-solving walkthroughs, project status updates, incident post-mortems.

```
Now:        Describe the current state with a concrete number or metric.
Obstacle:   Name the specific blocker or risk.
Solution:   Explain what you did or propose to do.
Evaluation: Show the outcome or expected impact.
```

**DS application:** "Model latency is 800ms (N). API timeout threshold is 500ms (O). We quantized to INT8 and batched requests (S). P95 latency dropped to 210ms in staging (E)."

---

### 4. BRIDGE — Acknowledge → Bridge → Land
**Mode:** Assertive (defensive)  
**Use for:** Handling hostile questions, redirecting off-topic challenges, pre-empting objections mid-presentation.

```
Acknowledge: Validate the concern without conceding the point.
Bridge:      Use a transition phrase to redirect.
             ("That's exactly why...", "Which brings me to...", "The data addresses that...")
Land:        Deliver your actual answer or evidence.
```

**DS application:** Q: "Isn't this just correlation?" → A: "Good point — correlation is always a risk (Ack). That's exactly why we ran a holdout A/B test (Bridge). The treatment group showed a statistically significant 18% lift, p<0.01, n=45K (Land)."

---

### 5. TRIANGLE — Three parallel reasons supporting one central claim
**Mode:** Assertive  
**Use for:** Building a persuasive recommendation with three independent supporting legs.

```
Claim:   [Central recommendation]
  ├── Reason A: [financial / business case]
  ├── Reason B: [technical / feasibility case]
  └── Reason C: [risk / cost-of-doing-nothing case]
```

**DS application:** "We should retrain quarterly (Claim): A) drift degrades AUC 8% per quarter (Business), B) retraining pipeline is fully automated, 4-hour job (Feasibility), C) competitor models refresh monthly — we're falling behind (Risk)."

---

### 6. HOURGLASS — General → Specific → General
**Mode:** Informative  
**Use for:** Educational presentations, explaining a new concept to a mixed room, teaching a methodology.

```
Wide:    Start at the 30,000-foot view (the category / problem space).
Narrow:  Zoom into the specific data, result, or technique.
Wide:    Zoom back out to the strategic implication.
```

**DS application:** Explaining transformers — start with "why attention?", zoom into the QKV mechanism with numbers, zoom back out to "this is why GPT-4 exists."

---

### 7. CLAP — Concern → Listen → Acknowledge → Propose
**Mode:** Assertive (empathetic)  
**Use for:** Stakeholder buy-in sessions, change management, presenting unpopular findings.

```
Concern:    Explicitly name the concern the audience is feeling.
Listen:     Show you've heard it — reference what they've told you.
Acknowledge: Validate it without dismissing it.
Propose:    Offer your solution framed as a response to that concern.
```

**DS application:** "I know the team is worried about model explainability for regulators (C). You told me compliance flagged three models last quarter (L). That concern is completely valid — black-box models are a real risk (A). This is why we built SHAP explanations into every prediction (P)."

---

### 8. CONTRAST — Old Way vs. New Way
**Mode:** Assertive + Visual  
**Use for:** Before/after comparisons, justifying a new approach, showing improvement.

```
Before: [Current state + its cost / pain]
After:  [New state + its benefit / metric]
```

**DS application:** Before: manual rule-based churn detection, 40% precision. After: gradient boosting model, 72% precision, $2.1M retained revenue.  
**Slide shape:** Two-column slide or two consecutive slides with identical structure.

---

### 9. QUESTION & ANSWER FUNNEL — Broad → Narrow
**Mode:** Informative  
**Use for:** Exploratory analysis presentations, showing how you narrowed from hypothesis to insight.

```
Question 1 (broad):  [What is the overall pattern?]
Question 2 (deeper): [Why is that pattern happening?]
Question 3 (action): [What should we do about it?]
Answer:              [Specific, data-backed recommendation]
```

**DS application:** Structure an EDA walkthrough — overall distribution → segment breakdown → root cause → intervention.

---

### 10. ANALOGY — Abstract Concept → Familiar Comparison
**Mode:** Visual  
**Use for:** Explaining complex ML concepts to non-technical audiences, making trade-offs tangible.

```
Concept: [Technical idea or model behavior]
"Think of it like...": [Everyday analogy]
"The difference is...": [Where the analogy breaks down — builds credibility]
```

**DS application:** "Regularization is like a fitness coach who penalizes you for memorizing yesterday's workout instead of building real strength. The penalty (lambda) controls how strict they are."

---

## Quick Selection Guide

| Situation | Recommended Structure |
|-----------|----------------------|
| Opening any presentation | SCQA |
| Answering "should we do X?" | PREP or TRIANGLE |
| Explaining a problem + fix | NOSE |
| Handling a hostile question | BRIDGE |
| Convincing a skeptical stakeholder | CLAP |
| Before/after comparison | CONTRAST |
| Explaining ML to non-technical audience | ANALOGY or HOURGLASS |
| Walking through an analysis | Q&A FUNNEL |
| One claim, three legs of evidence | TRIANGLE |

---

## Combining Structures

Most presentations combine structures across phases:

```
OPENING:  SCQA          (always)
BODY:     TRIANGLE      (3 supporting findings)
  ├── Finding 1: CONTRAST (before vs. after)
  ├── Finding 2: NOSE     (problem → solution → result)
  └── Finding 3: Q&A Funnel (how we got to the insight)
CLOSE:    PREP          (restate recommendation with confidence)
Q&A:      BRIDGE        (for every hard question)
```
