# Fault World — code, data and per-run records

Companion repository for the preprint

> **When Do Audited Action Gates Help? The Cost and Failure Conditions of Epistemic
> Strictness in LLM Agents** — Rong Xiang

Repository: <https://github.com/aiegisafety/fault-world>

Everything reported in the paper can be recomputed from the records in `data/`.
No API access is needed to reproduce the *analysis*; API access is needed only to
generate new runs.

---

## What this is

A synthetic fault-diagnosis environment in which

* $K$ **true causal rules** reproduce under intervention ($P \approx 0.90$), and
* $M$ **spurious associations** are produced by hidden latent confounders: they look like
  moderately strong rules in observational data ($P \approx 0.74$–$0.78$) but collapse to
  the $0.50$ base rate under intervention.

Because the two are indistinguishable without experiments, "epistemic strictness" becomes a
decision with a real price, and the environment can be used to measure what caution costs
rather than only whether an agent is cautious.

**The interventional invariant** (20,000 samples, base configuration):

| Relation | observational `P(s | c=1)` | interventional `P(s | do(c=1))` |
|---|---|---|
| true rules (K=3) | 0.902, 0.904, 0.897 | 0.902, 0.903, 0.899 |
| spurious pairs (M=4) | 0.773, 0.740, 0.777, 0.740 | **0.503, 0.496, 0.497, 0.502** |

---

## Repository layout

```
code/     environment, experiment driver, audit, analysis
data/
  e1_2x2/
    state/<model>[__<variant>][__c30]/<COND>_<GATE>_<rep>.json
              one file per run: full prompts, model replies,
              executed interventions, diagnostic cases and answers
    *.jsonl   flattened per-run summaries
  *.png/json  figures and the numbers behind them
paper/    LaTeX source and figures
notes/    per-experiment reports (E0–E7), the revision log, prior-art status
```

### Per-run checkpoint schema (`data/e1_2x2/state/**/*.json`)

| field | meaning |
|---|---|
| `strict` | strictness condition (`LOW`/`HIGH`, ladder `L0`–`L5`, or budget `B0`–`B7`) |
| `gate` | action gate: `OFF`, `SELF` (model's own labels), `AUDIT` (harness audit) |
| `rep` | repetition index; `seed = 7000 + rep` fixes the world |
| `world` | environment variant (absent = the original `fault_world_v2`) |
| `msgs` | the complete conversation sent to the model |
| `log[i].raw` | the model's reply for episode *i* |
| `log[i].executed` | interventions actually run, with symptom counts |
| `cases` | the diagnostic cases and their ground-truth answers |
| `n_iv` | total interventions used |
| `empty_rounds` | episodes where the model returned unparseable/empty content |

Conditions share world seeds, so any two conditions can be compared **paired by `rep`**.

---

## Reproducing the analysis (no API needed)

```bash
python3 code/analyze_2x2.py qwen3.8-27b      # condition table for one model
python3 code/stats.py                        # bootstrap CIs + paired contrasts
python3 code/holm.py                         # all contrasts with Holm correction
python3 code/t3_curve.py                     # strictness ladder + figure
python3 code/s2_curve.py                     # intervention-budget ladder + figure
python3 code/bg_interaction.py               # budget x gate interaction
python3 code/e4_robustness.py                # environment-variant robustness
python3 code/t2_replication.py               # 30-case split-half main result
```

`matplotlib` is required only for the figure-producing scripts.

## Generating new runs (API needed)

```bash
cp code/.env.example code/.env      # then fill in your own endpoint and key
python3 code/probe_models.py        # check which models still have quota
EXP=E4 WORLD=base N_CASES=30 WORKERS=8 python3 code/run_2x2.py <model> 20 140
```

`run_2x2.py` is **step-wise and resumable**: each invocation advances every unfinished run
by as many single API calls as fit in its time budget, then checkpoints to disk. Re-run the
same command until it reports `0 still unfinished`.

Cost: roughly **12k tokens per run** under the 8-case protocol and **10–14k** under the
30-case protocol.

---

## Headline numbers

| Result | Value |
|---|---|
| Strictness `LOW→HIGH`: spurious adoption | −1.509 (Holm *p* = .0023, *n* = 57) |
| Strictness `LOW→HIGH`: true-rule recall | −0.111 (Holm *p* = .0117) |
| Gate `OFF→AUDIT`, 30-case held-out half | **+0.114 [+0.019, +0.208]** (*n* = 55, *p* = .024) |
| Moderation by ungated baseline | *r* = −0.537 (*n* = 55, permutation *p* = .0007) |
| Gate effect at zero verification budget | **−0.315 [−0.452, −0.178]** (Holm *p* = .0017) |
| Budget × gate interaction (DiD) | **+0.456 [+0.297, +0.625]** |
| Interventions used across 6 strictness levels | flat (all adjacent contrasts Holm *p* > .29) |

1,330 completed runs across 9 models and 18 experiment cells.

---

## Honest notes

* **Chain-of-thought was disabled** (`enable_thinking=false`) throughout, for feasibility.
  This is an untested simplification.
* The model set differs between experiments because of free-quota availability; results are
  **not pooled across experiments**.
* `notes/PLAN-REVISIONS.md` is the log of **our own hypotheses that the data refuted**,
  including two claims withdrawn after Holm correction. Read it before reusing any earlier
  statement from the notes.
* `notes/M1-related-work-status.md` records which prior-art claims were verified against
  actual arXiv pages and which were not. Unverified references are not cited in the paper.

---

## License

Code: MIT. Data and notes: CC BY 4.0. See `LICENSE`.
