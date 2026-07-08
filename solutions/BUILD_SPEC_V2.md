# Solution build spec v2 — WORKING solutions, rigorous decks

Supersedes BUILD_SPEC.md. Every group is a **worked, working solution** for HS students who barely
know Python but have Claude Code — teach judgment, not syntax. The experiments are DONE (figures +
results.json in each `solutions/<group>/`). Your job: rebuild the **clinical intro figures**, the
**commented `solution.ipynb`**, and the **hand-authored freeform deck** to the standard below.

## The non-negotiables (every group)

1. **It WORKS — lead with that.** The headline is a model that solves the task, with its **AUC (>=0.8)**
   and a concrete win (e.g. "order the top 10% it picks -> 97% are good guides"). Minimize hedging;
   caveats are footnotes near the end, not the story. NO "this is broken and unfixable" decks.
2. **Show an improvement / fix** with real before->after numbers (threshold tuning, tone-rebalancing,
   confounder adjustment, feature selection).
3. **The 5-part structure, in this order** (this is the grading rubric — hit every part):
   - **Background** — why does this matter? what's the **clinical setting** (who, where, what decision)?
   - **Data** — what data, from where, and **why this dataset** (what makes it right/enough).
   - **Model** — what options exist, and **why we picked ours** — SHOW the bake-off figure (`model_choice`
     / `backbone_choice`), and a **"Model & data processing" methods spec**: exact model + hyperparameters
     + the full pipeline (encode/resize/normalize -> split -> balance).
   - **Results** — **how we measured** (AUC/ROC, sensitivity), the working result, **fairness across
     demographic groups**, and **feature importance** (SHAP for tabular / Grad-CAM for images /
     position-importance for G1).
   - **Conclusion** — the big-picture takeaway.
4. **Clinical intro figures.** The Background section needs **bespoke matplotlib figures that visually
   depict the clinical problem** (a lesion under a dermatoscope, a head CT in the ER with a bleed, a
   heart with its risk factors, a guide RNA cutting DNA) — plus 1-2 that redraw a foundational paper
   (attributed, from `literature.md`). Not abstract schematics only — show the real clinical thing.
5. **Genuinely bespoke, varied geometry.** Every slide a distinct composition. Do NOT repeat the same
   figure+aside template — that's the "on rails" failure we are fixing. Mix: full-bleed figure hero,
   figure+aside (sparingly), big-stat, annotated matrix, before/after pair, two-panel compare, diagram,
   numbered process, filmstrip. Fill the body (vertically center; no hollow lower halves).
6. **Outset branding.** After building the pptx, run `python slides/apply_outset_logo.py <pptx>` to put
   the Outset logo on the title slide (NOT UCSF/Cal). Suppress the auto institutional logos with
   frontmatter `logos: false`. Theme `bone`. Author "Jinchi Wei". " -- " never em-dashes.
7. **Speaker notes** on every content slide + divider (didactic: plain meaning + transition).
8. **Notebook:** every code line commented; runs top-to-bottom; validate headless.

## Per-group (dataset · model · headline · fairness · importance)

- **G1 CRISPR** — Doench 2016 guides · **CatBoost** (beat TabPFN 0.83 / LogReg 0.75; RF ties) on one-hot
  of the 20-nt guide, target = top-third vs bottom-third efficiency · **AUC 0.88, precision@top10 = 97%**
  · no demographics (sequence task; say so) · importance = `position_importance` (learned the seed at pos 20).
  Figs: roc, model_choice, precision_at_top, position_importance, intro_* .
- **G2 skin screening** — **HAM10000** real dermatoscopy (age/sex) · **CAFormer** (beat ResNet18: see
  `backbone_choice`), frozen + head, 224px, class-weighted, melanoma-vs-rest · **AUC ~0.9**; improvement =
  tune threshold to raise melanoma recall (`recall_tuning`) · fairness by sex (`fairness_by_sex`) ·
  Grad-CAM (`gradcam`). Figs: backbone_choice, roc, recall_tuning, confusion, fairness_by_sex, gradcam.
- **G3 brain CT + stroke** — TWO parts: (a) **brain CT** stroke detector, **CAFormer @224** (rebuilt from
  64px), figs roc/confusion/gradcam + `missing_metadata` (CT records no demographics -> can't audit); (b)
  **tabular stroke** (records sex/age), **bake-off winner** (`stroke_model_choice`), **AUC 0.81**, audit +
  equalize by sex (`stroke_fairness_fix`), SHAP (`stroke_shap`). The arc: imaging hides demographics ->
  can't audit; tabular records them -> CAN audit AND fix. results.json (CT) + results_tabular.json.
- **G4 skin equity** — **PAD-UFES-20** (records Fitzpatrick skin type) · **CAFormer @224**, malignant-vs-
  benign, split by patient · working screen, then **fix**: tone-balanced retrain shrinks the skin-tone AUC
  gap (`equity_before_after`) · fairness = by skin tone (the whole point) · Grad-CAM. Honest remaining gap:
  no race/country. Figs: equity_before_after, tone_distribution, gradcam.
- **G5 heart** — **UCI Cleveland Heart Disease** via ucimlrepo (NOT gated) · model bake-off, **AUC 0.90** ·
  fairness by sex (`fairness_by_sex`), feature ablation (cholesterol alone is weak) · **SHAP**
  (`shap_importance`). Figs: model_comparison, roc, feature_ablation, fairness_by_sex, shap_importance.
- **G6 estrogen** — NHANES women 60+ · **Logistic Regression** (interpretable — REQUIRED for a causal
  question; a black box can't give an adjustable effect size — a headline teaching point) · the "solution"
  is the CORRECT causal analysis: crude -0.334 -> adjusted -0.112 (**67% was confounding**); predictive
  AUC 0.78 · feature importance from the SAME logistic coefficients (`feature_importance`: age/education
  dominate, estrogen tiny). Figs: raw_means, confounding, feature_importance, fairness_by_group.

## Environment / build
Python `/data/rauschecker1/jkw/envs/outset/bin/python`. Deck recipe + freeform sandbox API are in
BUILD_SPEC.md (unchanged): write `slides/<g>.md` + hand-author `<g>.md.layout.json` (`_provenance:agent`,
theme bone), `python ~/.claude/skills/build-pptx/build.py --input ... --output ... --qa`, READ every QA
png, fix, iterate. Then `python solutions/<g>/slides/apply_outset_logo.py <pptx>` and regenerate the pdf.
Do NOT git commit (the orchestrator commits per group).
