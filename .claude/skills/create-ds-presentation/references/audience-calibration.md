# Audience Calibration Rules

Reference for `create-ds-presentation`. Apply the gating table below before writing any slide body.

---

## Content Gating by Audience Type

| Content type | EXEC | TECH | MIXED | CLIENT |
|---|---|---|---|---|
| Business impact (revenue, cost, risk) | ✅ Always | ✅ Yes | ✅ Always | ✅ Always |
| Recommendation + owners + dates | ✅ Always | ✅ Yes | ✅ Always | ✅ Always |
| Model metrics (AUC, F1, RMSE) | ❌ Appendix only | ✅ Body slide | ✅ Body + plain-language gloss | ❌ Appendix only |
| Statistical significance (p-value, CI) | ❌ Appendix only | ✅ Body slide | ✅ Body — translated to plain language | ❌ Appendix only |
| Methodology / algorithm choice | ❌ Appendix only | ✅ Body slide | ⚠️ 1 brief slide + appendix | ❌ Appendix only |
| Feature importance / model internals | ❌ Never | ✅ Body slide | ⚠️ Only if decision-relevant | ❌ Never |
| Code snippets | ❌ Never | ✅ Appendix | ❌ Never | ❌ Never |
| Contextual background / domain basics | ❌ Skip | ❌ Skip | ⚠️ 1 slide if needed | ✅ Always |

---

## Audience-Specific Rules

### EXEC (Executive / Senior Stakeholder)
- Cut everything that doesn't connect to a decision or a dollar
- Methodology goes entirely to appendix — never body
- Recommendation on slide 2, always — no exceptions
- No stat language: no "p-value", "AUC", "RMSE" in body slides
- Translate all metrics to business outcomes before placing on slide

### TECH (Technical Peer — Data Scientist, ML Engineer)
- Include all rigor signals: confidence intervals, holdout strategy, alternatives considered
- Skip business narrative fluff — they know why it matters
- Include failure modes and edge cases
- Methodology slide in body is expected and welcomed

### MIXED (Mixed room — Manager + Technical + sometimes Exec)
- Every technical term on a body slide gets a plain-language gloss in parentheses
  - e.g., "AUC (ability to rank at-risk users correctly, 0–1 scale)"
  - e.g., "p < 0.01 (less than 1% chance this is random noise)"
- Use the `[EXEC]` / `[TECH]` annotation system so both audiences get what they need
- 1 brief methodology slide in body; full detail in appendix

### CLIENT (External Client — outside the organization)
- Assume no internal context — set the scene before showing any data
- Replace all internal system names with descriptive labels:
  - "v3-xgb-prod" → "our purchase prediction model"
  - "ds-pipeline-v2" → "our automated data processing system"
- No methodology unless the client explicitly needs it (move to appendix)
- Plain language only; define every acronym even if it seems obvious

---

## Audience Contract Line

At the top of every deck output, print this one line:

```
AUDIENCE CONTRACT: [audience type] — showing [what's included] — hiding [what's in appendix]
```

Example:
```
AUDIENCE CONTRACT: EXEC — showing business impact + recommendation — hiding methodology, model metrics, CI (all in appendix)
```

---

## Translating Metrics for EXEC and CLIENT

| Technical metric | Plain-language translation |
|---|---|
| AUC = 0.87 | "Our model ranks at-risk users correctly 87% of the time" |
| p < 0.01 | "Less than 1% chance this result is due to random noise" |
| Precision = 0.72 | "When we flag a user as at-risk, we're right 72% of the time" |
| RMSE = 45 | "Our revenue forecast is off by $45 on average" |
| Lift = 2.4× | "Targeting with the model gives 2.4× more conversions than random outreach" |
| F1 = 0.81 | "The model balances finding churners and avoiding false alarms at 81% effectiveness" |

**The "1000 users" test:** Always convert relative numbers to absolute examples:
- "18% churn reduction" → "18% fewer churned users = 1,800 users retained per quarter"
- "AUC improvement 0.81 → 0.87" → "We catch 60 more churners per 1,000 at-risk users"
