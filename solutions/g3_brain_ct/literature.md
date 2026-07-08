# Literature — Deep-Learning Detection of Stroke / Intracranial Hemorrhage on Head CT

Focus: (a) what real head-CT triage AI does, and (b) why a dataset with **no demographics** (age / sex / race / scanner) cannot be fairness-audited — the gap is the teaching point.

## References

1. **Chilamkurthy et al. 2018** — Deep learning algorithms for detection of critical findings in head CT scans: a retrospective study. *The Lancet* 392(10162):2388-2396. Deep networks trained on ~313,000 head CT scans detected intracranial hemorrhage (and its subtypes), skull fractures, midline shift, and mass effect with AUCs above 0.90, showing AI can help triage which scans a radiologist should read first. (https://doi.org/10.1016/S0140-6736(18)31645-3 · PubMed 30318264)

2. **Flanders et al. 2020** — Construction of a Machine Learning Dataset through Collaboration: The RSNA 2019 Brain CT Hemorrhage Challenge. *Radiology: Artificial Intelligence* 2(3):e190211. Over 60 volunteer neuroradiologists labeled 25,000+ head CT exams (874,035 images) for hemorrhage presence and subtype to build the largest public brain-hemorrhage dataset, used in a 1,345-team competition. (https://doi.org/10.1148/ryai.2020190211)

3. **Seyam et al. 2022** — Utilization of Artificial Intelligence-based Intracranial Hemorrhage Detection on Emergent Noncontrast CT Images in Clinical Workflow. *Radiology: Artificial Intelligence* 4(2):e210168. A deployed AI hemorrhage-detection tool reached ~93% accuracy and 87% sensitivity overall, but did noticeably worse on some subtypes (subdural ~69%), showing real tools have uneven blind spots. (https://doi.org/10.1148/ryai.210168)

4. **Gebru et al. 2021** — Datasheets for Datasets. *Communications of the ACM* 64(12):86-92. Proposes that every dataset ship a standardized "datasheet" (57 questions across 7 categories — including who is in the data) so users know a dataset's makeup and limits before trusting a model built on it. (https://doi.org/10.1145/3458723)

5. **Tripathi et al. 2023** — Understanding Biases and Disparities in Radiology AI Datasets: A Review. *Journal of the American College of Radiology* 20(9):836-841. A review of public medical-imaging datasets finds demographic, geographic, and disease representation is often skewed or simply unreported, and calls for better dataset documentation. (https://doi.org/10.1016/j.jacr.2023.06.015)

6. **Seyyed-Kalantari et al. 2021** — Underdiagnosis bias of artificial intelligence algorithms applied to chest radiographs in under-served patient populations. *Nature Medicine* 27:2176-2182. Chest-X-ray AI missed disease more often (higher false-negative "no finding" calls) in female, Black, and low-income patients — a bias you can only *see* if the dataset records those groups. (https://doi.org/10.1038/s41591-021-01595-0)

7. **Yang et al. 2024** — The limits of fair medical imaging AI in real-world generalization. *Nature Medicine* 30:2838-2848. Shows imaging AI often secretly relies on demographic "shortcuts," so fairness measured on one hospital's patients does not carry over to a new population — making demographic metadata essential for testing. (https://doi.org/10.1038/s41591-024-03113-4)

## Positioning

Real head-CT triage AI (Chilamkurthy 2018; Flanders 2020; Seyam 2022) does one main job: scan a stack of brain CT images and flag the ones that probably show bleeding, so a radiologist reads the dangerous scans first instead of in arrival order. In a busy ER that can save minutes that matter for a stroke. But these tools are not perfect and are not equally good on every patient or every kind of bleed. To know *whether* a tool is fair, you have to compare its accuracy across groups — young vs. old, women vs. men, different races, different scanners. That comparison is only possible if the dataset records who each scan came from. Datasheets-for-Datasets work (Gebru 2021; Tripathi 2023) says datasets *should* document this, and real audits (Seyyed-Kalantari 2021; Yang 2024) show hidden biases appear exactly along those lines. Our toy model shows a small network *can* learn to flag hemorrhage — but because the dataset ships **no** age, sex, race, or scanner labels, we literally cannot check who it fails on. That silence is the lesson, not a footnote.

## Intro figure specs

1. **Datasheet checklist: what a dataset *should* record vs. what this one has** — bespoke matplotlib figure, *(concept from Gebru 2021; Tripathi 2023)*. Two side-by-side columns of checkbox rows: left column "What a datasheet asks for" (Age, Sex, Race/ethnicity, Scanner/site, Label source, Consent) all checked; right column "What our brain-CT dataset has" — only "Image" and "Hemorrhage label" checked, the demographic rows shown as empty/greyed boxes. Drives home the missing-metadata gap at a glance for sophomores.

2. **How triage AI reorders the reading queue** — bespoke matplotlib figure, *(adapted from Chilamkurthy 2018; Seyam 2022)*. Two vertical stacks of CT "scans": left = arrival order (mixed normal and bleed), right = AI-sorted with a flagged bleed pushed to the top and a highlight arrow labeled "read first." A small caption notes real tools cut time-to-treatment. Shows the concrete benefit before we discuss the catch.

3. **You can't audit a bias you can't see** — bespoke matplotlib figure, *(concept from Seyyed-Kalantari 2021; Yang 2024)*. A simple grouped-bar "accuracy by group" chart where the bars for one hypothetical subgroup are drawn as dashed empty outlines labeled "no data — cannot measure," while a note says "if demographics were recorded, we could fill these in." Visually equates missing metadata with an un-checkable blind spot.
