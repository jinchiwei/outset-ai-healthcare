# Capstone Deliverables

**Pairs. Both build, both present.** Pick a problem (`project_options.md`), build it, improve it
honestly, present it. Graded on understanding + honesty, not accuracy (`rubric.md`).

## Timeline
- **Today:** pick your dataset, get the baseline running (`starter.ipynb` → Run all), **save a copy to Drive.**
- **Tomorrow:** improve, write the report, present (3 min).

## Hand in three things

**A · Working notebook** — runs top to bottom; reports **test-set** accuracy; includes your
experiment log (a markdown cell) and one failure case (an image it got wrong).

**B · One-page report** — copy `REPORT_TEMPLATE.md`, rename with your names, fill it in.

**C · 3-minute talk** — both speak: **(1) show it** live/one result, **(2) one finding**,
**(3) one limitation.** Show, don't narrate.

## The method (how you actually improve it)
1. **Baseline first** — get a number, however bad. It's what you beat.
2. **Change ONE thing** — augmentation, epochs, unfreeze backbone, different model/lr. Re-run.
3. **Keep what helps, log it** — `"augmentation: 0.82 → 0.86"`. That log *is* your talk.
4. **Evaluate honestly** — test split only; pick the metric that matters (a missed case usually
   beats a false alarm → sensitivity, not just accuracy); look at what it gets **wrong**.

*One change, one measurement. Change five things and you've learned nothing about why.*

## Claude
Expected and encouraged — for ideas, debugging, and explanations. Costs zero points. The **only**
rule: you must be able to explain every line. If Claude wrote it, read it, understand it, own it.

## Don't
Skip the baseline · tune five things at once · trust *training* accuracy · paste code you can't
explain · go broad ("we tried everything") when narrow wins ("tuned to miss <5% of cases").

---

## 3-minute talk structure
```
0:00–1:00  SHOW IT       "We predict ___ from ___." Run it live / show one result.
1:00–2:00  ONE FINDING   The single most interesting thing you learned (with the numbers).
2:00–3:00  LIMITATION    A failure case + why; one next step. Each can field a Q on the other's part.
```

See `rubric.md` for the 5 points and `project_options.md` (bottom) for concrete project ideas if stuck.
