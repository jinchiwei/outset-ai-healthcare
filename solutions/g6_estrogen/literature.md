# Estrogen / Hormone Therapy and Cognition in Older Women — Literature

Grounding references for the NHANES toy analysis. Teaching focus: **confounding vs. causation (healthy-user bias)** and the gap between objective cognitive tests and a patient's subjective "brain fog." All entries below were checked against PubMed / journal pages; a URL or DOI is given for each.

## References

1. **Tang et al. 1996** — Effect of oestrogen during menopause on risk and age at onset of Alzheimer's disease. *The Lancet*. An early **observational** study reporting that women who had taken estrogen appeared to develop Alzheimer's less often and later than women who had not. (https://doi.org/10.1016/S0140-6736(96)03356-9 ; PMID 8709781)

2. **Zandi et al. 2002** — Hormone replacement therapy and incidence of Alzheimer disease in older women: the Cache County Study. *JAMA*, 288(17):2123-2129. Another large **observational** study: prior hormone users, especially long-term users, seemed to have a lower risk of Alzheimer's. (https://doi.org/10.1001/jama.288.17.2123 ; PMID 12413371)

3. **Shumaker et al. 2003** — Estrogen plus progestin and the incidence of dementia and mild cognitive impairment in postmenopausal women: the Women's Health Initiative Memory Study (WHIMS): a randomized controlled trial. *JAMA*, 289(20):2651-2662. The big **randomized trial**: hormone therapy did not protect memory and actually *doubled* the risk of dementia in women 65+. (https://doi.org/10.1001/jama.289.20.2651 ; PMID 12771112)

4. **Rapp et al. 2003** — Effect of estrogen plus progestin on global cognitive function in postmenopausal women: the Women's Health Initiative Memory Study: a randomized controlled trial. *JAMA*, 289(20):2663-2672. Companion WHIMS trial on cognitive test scores: hormone therapy gave **no cognitive benefit**, and scores trended slightly worse. (https://doi.org/10.1001/jama.289.20.2663 ; PMID 12771113)

5. **Wharton et al. 2009** — Cognitive benefits of hormone therapy: cardiovascular factors and healthy-user bias. *Maturitas*, 64(3):182-187. Directly examines **healthy-user bias**: hormone users start out healthier (fewer heart-risk factors), which inflates their apparent brain benefit in observational data. (https://doi.org/10.1016/j.maturitas.2009.09.014 ; PMID 19879073)

6. **Vandenbroucke 2009** — The HRT controversy: observational studies and RCTs fall in line. *The Lancet*, 373(9671):1233-1235. A plain-language explanation of **why observational studies and randomized trials disagreed** on hormone therapy, and how confounding by healthy users caused the mismatch. (https://doi.org/10.1016/S0140-6736(09)60708-X ; PMID 19362661)

7. **Maki 2024 (Harvard Health)** — Menopause and brain fog: What's the link? *Harvard Health Publishing*. Explains the **subjective "brain fog"** many women report during the menopause transition and how it often does not match objective test scores. (https://www.health.harvard.edu/womens-health/menopause-and-brain-fog-whats-the-link)

8. **Chen & Shafir (Harvard Health) 2025** — The dangerous dismissal of women's pain. *Harvard Health Publishing*. Background on how **women's symptoms are under-recognized** and more often blamed on stress or emotions than investigated. (https://www.health.harvard.edu/pain/the-dangerous-dismissal-of-womens-pain)

## Positioning

For years, studies that simply *watched* women found that hormone-therapy users had better memory and less dementia [1, 2]. That looked like proof the pills helped the brain. But the women who chose hormone therapy were also healthier to begin with — richer, more likely to exercise and see doctors, with fewer heart-risk problems. This is **healthy-user bias**: the *type of person* who takes the drug, not the drug itself, drives the good outcome [5, 6]. When researchers ran a true **randomized controlled trial** (WHIMS), where a coin flip — not the patient — decided who got hormones, the benefit vanished and dementia risk actually rose [3, 4]. Our NHANES toy analysis shows the same trap in miniature: estrogen users score higher on the cognitive test at first, but that gap shrinks by two-thirds once we adjust for background differences. Adjustment reveals confounding — but with survey data we can only *hint* at it, never *prove* cause. Only a randomized trial can do that.

## Intro figure specs

1. **The effect flip: observational vs. randomized (adapted from Zandi 2002 and Shumaker 2003).** A simple two-bar or dumbbell chart. Left/"just watching" bar points *down* (hormones look protective, risk lower). Right/"coin-flip trial" bar points *up* (dementia risk higher). Same drug, opposite answer — the single most important slide. Use turquoise for the observational side and deeppink for the trial side, with a big arrow showing the sign flip.

2. **The confounding triangle (concept: healthy-user bias).** A three-node diagram: "Takes hormone therapy" and "Better memory" at the bottom, "Already healthier / wealthier / more active" at the top pointing down to both. A dashed line between the two bottom boxes labeled "looks like cause — but isn't." Teaches, in one picture, why a correlation can be a mirage created by a hidden third factor.

3. **Objective test vs. subjective brain fog (concept: symptom recognition, ref [7]).** A paired-dot / slope plot for a set of women: left column = objective cognitive test score (many land in the "normal" band), right column = self-reported "brain fog" severity (many report it's real and high). Connecting lines cross, showing the mismatch — the tests say "fine" while the patient says "something's wrong," motivating why we should take reported symptoms seriously.
