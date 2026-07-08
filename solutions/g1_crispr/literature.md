# Literature — Predicting CRISPR-Cas9 Guide RNA On-Target Efficiency

Background and citations for the worked-solution notebook + slide deck.
Dataset: Doench et al. 2016 (~5,300 guides). Audience: high schoolers.

## References

[1] **Doench, Fusi, Sullender, et al. 2016** — Optimized sgRNA design to maximize activity and minimize off-target effects of CRISPR-Cas9. *Nature Biotechnology* 34:184–191. Built "Rule Set 2" (the Azimuth model): a gradient-boosted model that scores a guide's cutting efficiency from its sequence, and the source of the ~5,300-guide dataset we use. (https://www.nature.com/articles/nbt.3437 · DOI: 10.1038/nbt.3437)

[2] **Doench, Hartenian, Graham, et al. 2014** — Rational design of highly active sgRNAs for CRISPR-Cas9–mediated gene inactivation. *Nature Biotechnology* 32:1262–1267. The original "Rule Set 1": measured ~1,800 guides and found that which letter sits at which position (and PAM identity) strongly predicts whether a guide cuts well. (https://www.nature.com/articles/nbt.3026 · DOI: 10.1038/nbt.3026)

[3] **Hsu, Lander, Zhang 2014** — Development and Applications of CRISPR-Cas9 for Genome Engineering. *Cell* 157:1262–1278. Highly readable landmark review explaining how Cas9 is steered to a DNA site by a short RNA "search string," ideal plain-English background for a general audience. (https://www.cell.com/cell/fulltext/S0092-8674(14)00604-7 · DOI: 10.1016/j.cell.2014.05.010)

[4] **Zheng, Hou, Zhang, et al. 2017** — Profiling single-guide RNA specificity reveals a mismatch-sensitive core sequence. *Scientific Reports* 7:40638. Direct evidence for the "seed" idea: mismatches in the PAM-proximal core of the guide break cutting far more than mismatches at the far (PAM-distal) end. (https://www.nature.com/articles/srep40638 · DOI: 10.1038/srep40638)

[5] **Chuai, Ma, Yan, et al. 2018** — DeepCRISPR: optimized CRISPR guide RNA design by deep learning. *Genome Biology* 19:80. A convolutional neural network that predicts on-target cutting (and off-target risk) directly from the guide sequence, showing deep learning can beat hand-built rules. (https://link.springer.com/article/10.1186/s13059-018-1459-4 · DOI: 10.1186/s13059-018-1459-4)

[6] **Kim, Min, Song, et al. 2019** — SpCas9 activity prediction by DeepSpCas9, a deep learning-based model with high generalization performance. *Science Advances* 5:eaax9249. Trained a CNN on 12,832 measured target sites; the model generalizes to new datasets, illustrating how more data + neural nets push sequence-only prediction further. (https://www.science.org/doi/10.1126/sciadv.aax9249 · DOI: 10.1126/sciadv.aax9249)

[7] **Konstantakos, Nentidis, Krithara, Paliouras 2022** — CRISPR-Cas9 gRNA efficiency prediction: an overview of predictive tools and the role of deep learning. *Nucleic Acids Research* 50:3616–3637. Accessible survey comparing the many published guide-efficiency predictors and where they agree/disagree. (https://academic.oup.com/nar/article/50/7/3616/6555429 · DOI: 10.1093/nar/gkac192)

[8] **Abbaszadeh & Shahlai 2025** — Artificial Intelligence for CRISPR Guide RNA Design: Explainable Models and Off-Target Safety. *arXiv preprint* 2508.20130. Recent, plain-language overview of AI guide-design that uses explainability maps showing models "light up" the PAM-proximal seed region, tying modern ML back to the known biology. (https://arxiv.org/abs/2508.20130)

[9] **Popejoy & Fullerton 2016** — Genomics is failing on diversity. *Nature* 538:161–164. Landmark commentary quantifying the heavy Eurocentric bias of genomic studies and reference/variant databases, and warning that advances in genomics stand to benefit the few, not all. The equity backbone for the "who is in the databases" point: a guide validated on the (mostly European) reference may not transfer to under-represented ancestries. (https://www.nature.com/articles/538161a · DOI: 10.1038/538161a · PMID: 27734877)

[10] **Sirugo, Williams & Tishkoff 2019** — The Missing Diversity in Human Genetic Studies. *Cell* 177(1):26–31. Widely-cited review documenting that GWAS participants are ~78% European (10% Asian, 2% African, 1% Hispanic, <1% other) and that this skew is "both scientifically damaging and unfair" — the source for the ancestry-composition bar in the equity figure. (https://www.cell.com/cell/fulltext/S0092-8674(19)30451-9 · DOI: 10.1016/j.cell.2019.02.048 · PMID: 31051100)

## Positioning

When real scientists pick a CRISPR guide, they rarely eyeball it. They run the 20-letter guide through a trained scoring model, the best known being Doench's "Rule Set 2" / Azimuth [1], descended from the earlier position-and-PAM rules of Rule Set 1 [2]. These models learned from thousands of guides that cutting efficiency is partly predictable from sequence alone: which nucleotide sits at which position matters, and the handful of letters nearest the PAM (the "seed" or core region) matter most, because a Cas9 guide that mismatches there usually fails to cut [4], [3]. Newer neural-network tools (DeepCRISPR [5], DeepSpCas9 [6]) push accuracy higher with more data, and reviews [7], [8] show these tools help but still disagree and never predict perfectly. Our toy model can honestly show the real signal — that position and seed-region letters carry information — on the same Doench dataset [1]. It cannot match a production tool, capture cell-type or chromatin effects, or say anything about off-target safety. It is a demonstration of the idea, not a design tool.

## Intro figure specs

**1. Anatomy of a guide + its target (adapted from Hsu, Lander & Zhang 2014 [3])**
Redraw a simple schematic: a horizontal DNA double strand with a highlighted 20-nucleotide protospacer, the 3-letter PAM (NGG) drawn immediately to its right, and the Cas9 "scissors" cutting ~3 bp upstream of the PAM. Label the guide RNA base-pairing to the target strand. The goal is to orient students to what the 20 letters in our dataset actually are and where the cut happens; keep it flat, cartoon-style, two or three brand colors.

**2. The seed region matters most (adapted from Zheng et al. 2017 [4] / Doench 2016 [1])**
A bar chart across the 20 guide positions (x-axis = position 1 near the 5' distal end to position 20 next to the PAM), y-axis = "relative importance / sensitivity to mismatch," with bars rising toward the PAM-proximal seed region (roughly positions 13–20). Draw a bracket labeling the "seed / core region." This visually plants the single most important biological intuition: not all 20 letters count equally.

**3. Predicted vs. measured cutting efficiency (adapted from Doench 2016 [1])**
A scatter plot: x-axis = model-predicted efficiency score, y-axis = measured cutting efficiency, points forming a positive but scattered cloud with a diagonal trend line. Annotate that "the trend is real but loose (points scatter)." This sets honest expectations for our own notebook result — a model can capture signal without being perfect.

**4. Where equity lives in genome editing (adapted from Popejoy & Fullerton 2016 [9] / Sirugo et al. 2019 [10])**
Two-part schematic. Top: the SAME 20-letter guide drawn over two DNA strands — a reference genome (guide matches, Cas9 cuts) and a patient of under-represented ancestry, where a variant common in some ancestries but absent from the reference sits under the seed/PAM, creating a mismatch, an altered PAM, or a new off-target site ("may cut less well, or hit a new off-target site"). Bottom: a 100%-stacked bar of who is actually in the genomic databases (European ~78%, Asian 10%, African 2%, Hispanic/Latino 1%, other/unknown 9%; GWAS participants per Sirugo et al. 2019). The point: G1's guide-sequence data has no demographics, so per-guide fairness does not apply — but the equity issue in editing lives one step up, in whom the reference and variant panels (gnomAD / 1000 Genomes) represent.
