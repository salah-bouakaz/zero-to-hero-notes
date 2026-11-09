# How to write a lab log entry

## General rules

- One entry per project step or work session, written the same day.
- Add new entries at the bottom. Don't rewrite old ones.
- Keep it to 6–12 lines. If it is longer, you are copying your notes.
  Formulas and drawings stay on paper or in the code.
- Use your own words, so that you in 3 months can understand it without
  the video.

## The four parts

| Part | What goes in | Good example |
|---|---|---|
| **What I did** | Facts only: what I built, ran or watched, with numbers, and where the full notes or code live. | "Trained the model for 5 epochs, test accuracy 86.6%. Code in `train.py`." |
| **What broke / what confused me** | Every error, wrong guess and thing I looked up, with the cause or fix in one line each. Include the silly mistakes. | "Accuracy came out as 8659% because `len(loader)` counts batches, not images." |
| **What I learned** | 2–5 sentences, no formulas, ideas I could now explain to a friend. If I could have written it before the session, it is not a lesson. | "The gradient points uphill, so the update subtracts it." |
| **Open questions** | Specific things I still don't get. Write "none yet" if truly none. Later add a "Closed:" line when answered. | "How does the chain rule work with several neurons per layer?" |

## Template

```markdown
## YYYY-MM-DD: topic
**What I did:** ...

**What broke / what confused me:**
- ...

**What I learned:** ...

**Open questions:** ...
```
