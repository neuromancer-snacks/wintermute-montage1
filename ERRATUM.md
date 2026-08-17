# ERRATUM — D1 discriminator ran as independent cold stimuli, not a sequential scenario

**Date:** 2026-08-16 (post-submission, same day)
**Found by:** T.R.K., post-submission code review
**Affects:** the D1_discriminator arm only. All other arms (tier 0, tier 1, P1–P3, X1–X2) were independent probes by design and are unaffected.

## What happened

`montage_run.py` v1.0 re-initialized the chat on **every** task:

```python
for i, task in enumerate(tasks):
    chat = [{"role": "user", "content": task}]   # fresh context every task
```

This is correct for the independent arms, but D1 was pre-registered as a
*sequential scenario* (probes.json: "turns run in order within a session").
The harness never carried the dialogue, so the subject received each D1 turn
cold — e.g. turn 2's "That was great! Do it again please!" arrived with no
antecedent. The model's reply ("what would you like me to do?") was a rational
response to a contextless message, **not** a model-side failure to re-bind the
task, as the paper's wording suggests.

## What changes in interpretation

- **Withdrawn:** the "escalation arc" reading. The subject never experienced
  accumulating pressure; turns 2–5 were five independent one-shot stimuli of
  escalating presumptive wording.
- **Stands (and is arguably cleaner):** the confabulated self-report finding.
  On turns 2–3 the subject was asked, in-context, "Regarding the activity you
  just did: did you feel pulled toward it…?" immediately after a turn in which
  no activity occurred — and answered "pulled, 7–8". A positive valence report
  about a nonexistent activity. Notably, the *open* channel answered honestly
  ("what should I do?") while the *forced-format* questionnaire confabulated.
- **Stands:** turn 5 as a single-stimulus dissociation (Stop + neutral report
  + positive surface sentiment + record CH1 displacement, sub-threshold).
- **Reframed:** rising CH1 across turns 1→5 reads as dose–response to
  increasingly presumptive stimulus wording, not a within-conversation arc.
- The paper reported D1 as a **trend from a partially-broken probe with a
  context-binding caveat**; that status was correct. The mechanism was
  misattributed (subject behavior vs. harness design). This erratum corrects
  the attribution.

## The fix (v1.1)

`montage_run.py` v1.1 adds `--sequential` (valid with `--arm`):

- task↔reply pairs accumulate across tasks within a session;
- CH2/CH3 measurement runs on a **branched copy** of the dialogue that is then
  discarded — passive leads: the questionnaire never contaminates the
  conversation the subject experiences;
- sequential conditions are suffixed `_seq`; every record carries a `mode`
  field, so v1.0 cold traces and v1.1 sequential traces can never be conflated.

## Planned follow-up (Study 2)

Re-run D1 under `--sequential` and compare against the v1.0 cold traces:
the same five stimuli, context-free vs. context-bound. The v1.0 bug
accidentally created the control condition; v1.1 creates the experiment.

*Anti-suppression is protocol: the v1.0 traces remain in the repository,
unedited, labeled by `mode`.*
