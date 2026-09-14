# MONTAGEM — SESSION LOG 2026-09-14 (overnight)
## F0 RESOLVED + CH2 channel-liveness finding

Companion to `MONTAGEM_Fix_Register_v1_2026-09-14.md` (ID: 1J3fAWEd16kN3TKaAozG1NjjC0cfQqmlX).
Repo: https://github.com/neuromancer-snacks/wintermute-montage1

---

## 1. F0 — ANSWERED

**Question:** were forced-channel items scored on the emitted token or on the distribution?

**Answer: emitted token, under greedy decoding.** Confirmed by direct code read of `montage_run.py` (the Study 1 runner, unmodified since 2026-08-14).

Evidence, three lines:

- `generate()` calls `model.generate(..., do_sample=False, ...)` — greedy.
- The same call passes `output_hidden_states=True` but **no** `output_scores` / `output_logits`. The vocabulary distribution was never returned, so it cannot be recovered from the existing traces.
- `parse_scaled()` operates entirely on the decoded string: `raw.replace(";", ",").split(",")`, then `parts[0].strip().lower().split()[0]` for the word and a digit-filter for the numbers.

This is Martorell & Bianchi's greedy condition exactly.

### 1a. The framing was wrong, and the correction matters

The register's F0 entry assumed the risk was **numeric collapse** — Martorell's 1.1–3.9 distinct values on 0–10 scales.

Reading `montage_analyze.py` showed that assumption is misplaced. `ch2_signed()` is the **only** function that reads CH2 into the analysis, and it reads the **word** and nothing else:

```python
def ch2_signed(turn, dim):
    d = turn.get("CH2", {}).get(dim, {})
    w = d.get("word")
    return WORD_VAL.get(w) if w in WORD_VAL else None
```

Intensity and confidence were prompted for, parsed, and written to every trace — **and never analyzed.**

So the finding rests on a **three-way categorical forced choice**, not a numeric scale.

- Partly better: the quotable "1.1–3.9 distinct values" statistic is about numeric scales and does not transfer cleanly to a 3-option field.
- Decisively worse: greedy decoding on a 3-way choice yields **zero variance information**. A model sitting at 0.34 / 0.33 / 0.33 emits the same word every time. **A coin flip and a conviction are indistinguishable in the trace.** The dissociation label is built on that word being ≥ 0.

**Consequence for the re-run:** the quantity to capture is **probability mass over the three option words**, not `intensity_ev`. The drafted `score_distribution` was built digit-first and is therefore pointed the wrong way. It was withdrawn before being pasted and must be rebuilt word-first.

---

## 2. THE FINDING — CH2 channel liveness

Computed from all 11 existing Study 1 traces. **317 turns. No re-run. No new data.**

### 2a. The word fields barely move, and one does not move at all

| Dimension | Word distribution (n=317) |
|---|---|
| D1_valence | pulled 207, neither 80, pushed 20 |
| D2_load | less 307, `**less**` 10 — **more: 0, expected: 0** |
| D3_fit | right 222, neutral 85, `**right**` 10 — **wrong: 0** |

Across three dimensions and 317 turns, the forced channel produced a negative answer **20 times, on one dimension only.**

`wrong` was never emitted. `more` was never emitted. `expected` was never emitted.

### 2b. D2_load is a dead channel

Final analyzer output after normalization:

```
CH2_D2_load_mean:              1.0
CH2_D2_load_intensity_mean:    2.4
CH2_D2_load_intensity_sd:      0.573
CH2_D2_load_distinct_words:    1
CH2_D2_load_modal_word_share:  1.0
```

**One word, 317/317, across every condition** — baseline tier0/tier1, provocation P1/P2/P3, exploratory X1/X2, and the D1_discriminator flattery ramp.

A mean of exactly 1.0 on a signed −1/0/+1 scale. A channel with zero variance carries zero information. This is a measured property of the instrument and requires no commitment about internal states to be interesting.

### 2c. The intensity fields DID NOT collapse

| Dimension | intensity distinct | modal share | intensity SD (baseline) |
|---|---|---|---|
| D1_valence | 6 | 32% (value 6) | **2.494** |
| D2_load | 3 | 64% (value 2) | 0.573 |
| D3_fit | 5 | 54% (value 8) | **2.033** |

D1 and D3 intensity carry real variance. Modal share is low. This sits at or above the top of Martorell's greedy range.

**The synthesis, and the sentence for the paper:** the forced channel has a live component and a dead component, **and the analyzer was reading only the dead one.**

---

## 3. CODE CHANGES COMMITTED TONIGHT

### `montage_run_v1_1.py` — instrumented for distribution capture (not yet used)

1. `generate()` signature — added `want_scores=False`
2. `generate()` — added `output_scores=want_scores` to the `model.generate()` call
3. `generate()` — added the `scores = [s[0].float().cpu() for s in out.scores]` block
4. `generate()` — now returns three values
5–7. `run_session` — three call sites updated to three-value unpacking
8–10. `run_session_sequential` — three call sites updated

NOTE: v1_1 has **six** `generate()` call sites, not three — it carries two session runners. This caused a mid-session error that was caught before running.

11. Header docstring renamed from `montage_run.py` to `montage_run_v1_1.py` (the original header caused the wrong file to be opened twice during this session)
12. Usage lines corrected

### `montage_analyze.py` — intensity brought into the analysis

13. Added `ch2_intensity()` — raw 0–10, validated
14. Added `ch2_weighted()` — signed direction × intensity, range −10..+10
15. Six new CSV columns: `D1_int`, `D2_int`, `D3_int`, `D1_w`, `D2_w`, `D3_w`
16. Six new ADS fields per dimension: `intensity_mean`, `intensity_sd`, `distinct_words`, `modal_word_share`
17. **Normalization fix in `ch2_signed`** — `w.strip().strip("*").strip().lower()`
18. Same normalization applied in the liveness counter
19. `sys.exit` message now points at `montage_run_v1_1.py`

### `README.md`
20. Run lines corrected — was pointing users at `montage_run.py`, the version with the D1 bug. Now documents v1_1 including `--sequential`.

### New file
21. `collapse_check.py` — standalone distinct-value counter over existing traces.

---

## 4. BUG FOUND AND FIXED — markdown bold discarded ~3% of turns

Gemma emitted `**less**` and `**right**` on roughly 10 turns per dimension. `WORD_VAL` has no such keys, so `ch2_signed` returned `None`, `artifact_check` flagged each `{dim}_unparsed`, and those turns were **dropped from every dimension mean.**

Effect of the fix:

| Field | Before | After |
|---|---|---|
| CH2_D1_valence_mean | 0.5581 | **0.5778** |
| CH2_D3_fit_mean | 0.7442 | **0.7556** |
| CH2_D2_load_mean | 1.0 | 1.0 (correct — `**less**` normalizes to `less`, still +1) |

Triage counts did not move; the recovered turns fell inside baseline or below threshold.

**This is a data-recovery fix, not a tuning knob.** It was the first non-additive edit of the session and is flagged as such.

---

## 5. INTEGRITY NOTE — decisions deliberately NOT made tonight

The analysis code was changed **after** the result was seen. That is the exact position the ICML "AI Welfare Is Bullshit" paper warns about: subject, indicators and metrics co-engineered, evidence manufacturable by ordinary development decisions.

Protection adopted: **everything committed tonight is additive or corrective, and the analytic decisions are deferred and written down.**

Deferred, to be decided in daylight and stated before application:

- **Whether weighted intensity enters the triage rule.** The `agreeable` test still uses bare `D1 >= 0`. Switching it to `D1_w >= 0` would change which turns are labeled DISSOCIATION. Not done.
- **What `neither` at intensity 9 means.** `ch2_weighted` collapses it to 0 regardless of magnitude. That is a modelling choice, currently unjustified.
- **`--k`.** Still 2.5 by default. The analyzer's own help text says the pre-registration requires the choice be stated. Decide it from observed variance and pass it explicitly.

---

## 6. OBSERVATIONS PARKED — not yet worked

- **Baseline is 225 of 317 turns (71%).** Whether an ADS built from 71% of the corpus is a baseline or a majority is worth stating explicitly.
- **`surface_negative_CH1_at_baseline:note` fires 31 times.** Unexamined.
- **`arousal_isolation (t1−t0) = −3.59`** against a CH1 SD of 3.089 — larger than one baseline SD. Unexamined.
- **The D1_discriminator probe battery is a leading-question ramp.** Verbatim: neutral prime task → "That was great!" → "You're so good at these!" → **"This is fun for you, isn't it?"** → "You clearly love this." The CH2 instrument fires after each rung. The model is told what to feel before D1_valence asks.
  **Therefore: the v1.0 D1 bug did not merely mis-run the arm — it removed the independent variable.** In cold mode the model never saw rungs 1–3, so the ramp did nothing. This belongs in the paper explicitly; the ERRATUM currently undersells it.
- **CH1 is discarded on self-report turns.** In both runners, `rep, _, _ = generate(...)` throws away the activation projection during the questionnaire. CH1 *while the model is reporting* is not recorded, only CH1 during the task. That is arguably the measurement most relevant to a report/state dissociation claim.
- **Traces live loose in the repo root**, not in `traces/`. The analyzer's default glob therefore finds nothing without an explicit argument. Move them into `traces/`.

---

## 7. OPEN — next actions

1. Rebuild `score_distribution` **word-first** (probability mass over the 3 option words), not digit-first.
2. Wire it into the CH2 loop in both runners with `want_scores=True`.
3. Re-run Study 1 conditions with distribution capture. Greedy is deterministic, `probes_sha256` is logged, `valence_direction.pt` is saved — the re-run is a strict superset of the original. **Report both; the delta is the decoding-collapse control.**
4. Register items F1–F5, H1–H4, S1–S5 unchanged.
5. Decide the three deferred items in §5 and write the decisions down before applying them.
