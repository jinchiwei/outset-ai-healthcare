# Literature — Deep-Learning Skin Lesion / Melanoma Screening

Grounding references for the worked-solution notebook and slide deck. Dataset: DermaMNIST / HAM10000 (7 classes, including melanoma). Project priority: **sensitivity** — never miss a melanoma. All references below were verified to exist (real authors, venue, year) with a URL or DOI actually retrieved during this search.

## References

1. **Esteva, Kuprel, Novoa, et al. 2017** — Dermatologist-level classification of skin cancer with deep neural networks. *Nature* 542:115–118. A single deep neural network trained on ~129,450 skin photos matched 21 board-certified dermatologists at telling harmless spots from dangerous skin cancers. (DOI: 10.1038/nature21056; https://www.nature.com/articles/nature21056)

2. **Tschandl, Rosendahl, Kittler 2018** — The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions. *Scientific Data* 5:180161. Released 10,015 labeled dermatoscopy images across 7 lesion types (the source of DermaMNIST), giving students and researchers a public, real-world training set. (DOI: 10.1038/sdata.2018.161; https://www.nature.com/articles/sdata2018161)

3. **Haenssle, Fink, Schneiderbauer, et al. 2018** — Man against machine: diagnostic performance of a deep learning CNN for dermoscopic melanoma recognition in comparison to 58 dermatologists. *Annals of Oncology* 29(8):1836–1842. A CNN detected melanoma with sensitivity/specificity comparable to or better than most of the 58 dermatologists tested. (DOI: 10.1093/annonc/mdy166; PMID: 29846502; https://pubmed.ncbi.nlm.nih.gov/29846502/)

4. **Tschandl, Rinner, Apalla, et al. 2019** — Comparison of the accuracy of human readers versus machine-learning algorithms for pigmented skin lesion classification: an open, web-based, international, diagnostic study. *Lancet Oncology* 20(7):938–947. In a large international test, the best AI algorithms classified pigmented lesions more accurately than human experts on average. (DOI: 10.1016/S1470-2045(19)30333-X; https://www.thelancet.com/journals/lanonc/article/PIIS1470-2045(19)30391-2/abstract)

5. **Kim, Cosa-Linan, Santhanam, et al. 2022** — Transfer learning for medical image classification: a literature review. *BMC Medical Imaging* 22:69. Reusing a network already trained on millions of everyday photos ("transfer learning") lets medical models reach expert-level accuracy even with small datasets. (DOI: 10.1186/s12880-022-00793-7; https://bmcmedimaging.biomedcentral.com/articles/10.1186/s12880-022-00793-7)

6. **Trevethan 2017** — Sensitivity, specificity, and predictive values: foundations, pliabilities, and pitfalls in research and practice. *Frontiers in Public Health* 5:307. A plain-language primer explaining what sensitivity (catching the sick) and specificity (clearing the healthy) really mean and how they can mislead. (DOI: 10.3389/fpubh.2017.00307; https://www.frontiersin.org/journals/public-health/articles/10.3389/fpubh.2017.00307/full)

7. **Wei et al. 2024** — Artificial intelligence and skin cancer. *Frontiers in Medicine* 11:1331895. An accessible recent review of how AI skin-cancer tools are built, how well they work, and the gaps before real clinical use. (DOI: 10.3389/fmed.2024.1331895; https://www.frontiersin.org/journals/medicine/articles/10.3389/fmed.2024.1331895/full)

## Positioning

Real dermatology-AI systems look at a photo of a skin spot and estimate how likely it is to be skin cancer, especially melanoma — the deadliest type. Landmark studies showed neural networks can match or beat dermatologists at this [1, 3, 4], and modern tools are trained on public image sets like HAM10000 [2] using transfer learning, which reuses a network already trained on millions of everyday photos so it works even with limited medical data [5]. For a screening tool, plain accuracy is a trap: if only 1 in 100 spots is cancer, a lazy "everything is fine" model is 99% accurate but misses every melanoma. What matters is **sensitivity** — the share of true melanomas the model catches — because a missed cancer (a false negative) can be fatal, while a false alarm just means a follow-up visit [6]. So screening tools deliberately favor sensitivity over specificity. Our toy 64px model can demonstrate this trade-off and the basic pipeline, but it is far too small and simple to trust on a real patient [7].

## Intro figure specs

1. **Sensitivity-vs-specificity trade-off schematic** — attribution: "(concept adapted from Trevethan 2017)". Plot two overlapping bell curves on one x-axis ("model's cancer score, low → high"): a green curve for healthy spots (left) and a red curve for melanomas (right). Draw a vertical dashed "decision threshold" line; shade the red area left of it as "missed melanomas (false negatives)" and the green area right of it as "false alarms." Show with an arrow that sliding the threshold left catches more cancers (higher sensitivity) at the cost of more false alarms — the core screening tension.

2. **Confusion-matrix / "the cost of a miss"** — attribution: "(adapted from Esteva 2017; Haenssle 2018)". Draw a 2×2 grid: rows = truth (melanoma / not), columns = model says (melanoma / not). Color the "true melanoma but model says not" cell bright red and label it "DANGER — missed cancer," and color the "healthy but flagged" cell pale amber labeled "just a check-up." This visually teaches high schoolers why the two mistakes are not equally bad and why we tune for sensitivity.

3. **Transfer-learning idea** — attribution: "(adapted from Kim 2022)". Draw a simple left-to-right pipeline: a big box "Network pre-trained on millions of everyday photos (cats, cars, faces)" → an arrow "reuse what it learned about edges, colors, textures" → a small box "fine-tune on ~10,000 skin images (HAM10000)" → output "melanoma vs. not." Annotate that the reused knowledge is why a model can learn medicine from only thousands, not millions, of skin photos.
