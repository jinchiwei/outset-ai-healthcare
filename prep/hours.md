# Prep hour log

Tracking time toward the 15-hour prep budget (Outset agreement, 2026 summer course). Instructor said "don't worry about cap, want quality" — tracking anyway for accountability.

| Date | Hours | What was done |
|------|-------|---------------|
| 2026-04-28 | 0.5 | Repo skeleton, requirements.txt, README, initial syllabus drafting |
| 2026-04-29 | 4.0 | Build plan v1, CEO review v1, locked decisions for v2: APTOS DR anchor, 6-model ladder structure for D1, PyRadiomics+TabPFN multimodal, Track A/B/C reframe, no pre-course quiz. Wrote v2. |
| 2026-04-30 | 1.0 | CEO review v2 caught critical issues: D2 synthetic-report leakage, PyRadiomics on fundus near-noise, D1 ladder over-budgeted. Locked v3 decisions: dropped tier system, dropped XGBoost, swapped D2 to Open-i CXR with cached LLM, cut capstones 5→3. Wrote v3. |
| 2026-05-06 | 0.5 | Reviewed MIT 6.S191 + JHU repos for lab structure. Adopted MIT TODO-blank + paired-solution pattern (v3.1). Students all have Claude Pro now; reframed Claude as ambient pair programmer (D1/D2) and full build tool (D3), not the course subject. |
| 2026-05-06 | 3.0 | Built + validated Phase 0 (theme, nbutil+code_with_todos, smoke runner, colab_setup, warmup) and all of D1: common.py (loader, 5 model factories, train/eval, Grad-CAM, viz), build_day1.py emitting lab+solution. Prepped APTOS subset, pushed dreamxjei/aptos-mini (1326 train/332 val). Ran the full ladder end-to-end: logreg .53 / mlp .48 / cnn .55 / resnet .58 / vit .64 (10.7 min MPS). Fixed ViT (freeze backbone) and CNN (BatchNorm). Repo made public. |
| 2026-05-06 | 3.5 | D1 slides (29). Built + validated D2: pyradiomics won't install on py3.12 -> swapped to scikit-image radiomics features. Open-i pipeline (download, rules-mode LLM cache, synthetic demographics, balanced 690-case feature table), build_day2.py lab+solution, 23 slides. Validated: multimodal .89 vs image-only .64, leakage corr .81. Built + validated D3: 3 MedMNIST capstone kits (pneumonia validated test .795), options/rubric docs, 12 slides. Final smoke + README refresh. |

**Total: 12.5 / 15.0 hr**

**Note on revised scope:** v3.1 keeps v3's scope; the TODO/solution split is one builder emitting both notebooks. Estimated total build effort: 25-35 hr. Instructor accepted this scope.

**Critical pre-class task (Task 15 in build-plan):** instructor runs the D1 ladder notebook end-to-end on a free-tier Colab T4 the week before July 6 to validate wall-clock fits in 70 min. Recommended: Thu Jul 2.
