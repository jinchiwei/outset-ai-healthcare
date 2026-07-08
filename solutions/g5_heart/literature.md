# Literature — Predicting Heart Disease from Routine Clinical Features (g5_heart)

Theme focus: sex differences and the under-diagnosis of heart disease in women. Audience: high schoolers (many sophomores). Every entry below was verified against PubMed / the publisher page during the search — authors, venue, and year are real.

## References

1. **Detrano R, Janosi A, Steinbrunn W, et al. (1989)** — International application of a new probability algorithm for the diagnosis of coronary artery disease. *American Journal of Cardiology* 64(5):304–310. This is the study behind the Cleveland Heart Disease dataset we use; it tested a formula that estimates the probability a patient has clogged heart arteries from routine clinical measurements. (https://doi.org/10.1016/0002-9149(89)90524-9 · PMID 2756873)

2. **Wilson PWF, D'Agostino RB, Levy D, et al. (1998)** — Prediction of coronary heart disease using risk factor categories. *Circulation* 97(18):1837–1847. Introduced the Framingham Risk Score, a sex-specific calculator that turns age, cholesterol, blood pressure, smoking, and diabetes into a 10-year risk estimate for heart disease. (https://doi.org/10.1161/01.CIR.97.18.1837 · PMID 9603539)

3. **Yusuf S, Hawken S, Ôunpuu S, et al. (2004)** — Effect of potentially modifiable risk factors associated with myocardial infarction in 52 countries (the INTERHEART study): case-control study. *The Lancet* 364(9438):937–952. Found that nine risk factors — with abnormal cholesterol and smoking the two biggest — explain over 90% of the risk of a first heart attack worldwide. (https://doi.org/10.1016/S0140-6736(04)17018-9 · PMID 15364185)

4. **Mehta LS, Beckie TM, DeVon HA, et al. (2016)** — Acute myocardial infarction in women: a scientific statement from the American Heart Association. *Circulation* 133(9):916–947. The AHA's landmark statement documenting that women having heart attacks are more often misdiagnosed, treated less aggressively, and have worse outcomes than men. (https://doi.org/10.1161/CIR.0000000000000351 · PMID 26811316)

5. **van Oosterhout REM, de Boer AR, Maas AHEM, et al. (2020)** — Sex differences in symptom presentation in acute coronary syndromes: a systematic review and meta-analysis. *Journal of the American Heart Association* 9(9):e014733. Pooling many studies, women were more likely than men to report back pain, nausea, and shortness of breath and less likely to report classic chest pain — a reason women's heart attacks get missed. (https://doi.org/10.1161/JAHA.119.014733 · PMID 32363989)

6. **Bugiardini R, Bairey Merz CN (2005)** — Angina with "normal" coronary arteries: a changing philosophy. *JAMA* 293(4):477–484. Argues that many patients — mostly women — have real heart-related chest pain despite "clean" artery scans, and are wrongly reassured that nothing is wrong. (https://doi.org/10.1001/jama.293.4.477 · PMID 15671433)

7. **Dua UCI Machine Learning Repository — Heart Disease Data Set** (curated from Detrano et al.). The public 303-patient Cleveland dataset with 13 features (age, sex, chest-pain type, cholesterol, resting blood pressure, max heart rate, etc.) used in this project. (https://archive.ics.uci.edu/dataset/45/heart+disease)

## Positioning

Doctors do not diagnose heart disease from any single number. They combine several routine measurements — age, blood pressure, cholesterol, smoking, and diabetes — into a risk estimate, much like the Framingham Risk Score [2], and worldwide those same factors, led by cholesterol and smoking, drive over 90% of first heart attacks [3]. A key fairness problem: heart disease is under-diagnosed in women. Women more often have symptoms other than classic chest pain [5], are more likely to be told their arteries look "normal" even when something is wrong [6], and are misdiagnosed and under-treated more than men [4]. Our toy model learns from just 303 patients from the 1980s Cleveland study [1, 7]. It can show which features carry signal and let students build a real classifier — but it cannot prove anything about today's patients, and because the data is roughly 2:1 male, it may learn men's patterns better than women's. That skew is itself a lesson about biased data.

## Intro figure specs

1. **Risk factors add up to risk (concept from the INTERHEART study, Yusuf 2004).** Redraw as a simple horizontal bar chart of the biggest modifiable risk factors (abnormal cholesterol, smoking, high blood pressure, diabetes), bar length showing roughly how much each raises heart-attack odds. Caption: "adapted from Yusuf 2004 (INTERHEART)." Message for students: no single cause — risk is a stack of factors, and cholesterol (a feature in our dataset) is near the top.

2. **Men vs. women feel it differently (concept from van Oosterhout 2020 / AHA statement, Mehta 2016).** Redraw as two side-by-side body silhouettes with labeled symptom callouts: men skewed toward classic chest pain, women more often back pain, nausea, and shortness of breath. Caption: "concept from an AHA scientific statement (Mehta 2016)." Message: the "textbook" heart-attack picture is the male picture, which is one reason women get missed.

3. **Our dataset is lopsided (from the Cleveland data itself, Detrano 1989).** Redraw as a small pictograph or stacked bar showing the ~2:1 male-to-female split among the 303 patients. Caption: "from the Cleveland Heart Disease data (Detrano 1989)." Message: a model only sees the people in its data — if women are under-represented, the model may serve them worse, tying the project's math directly to the fairness theme.
