# Fairness, Skin-Tone Bias & Access in Dermatology AI — Literature

Answer-key literature notes for the equity lens on the DermaMNIST skin-lesion model:
*who does skin-cancer AI help, and who is missing from the training data?*

## References

1. **Fitzpatrick 1988** — The validity and practicality of sun-reactive skin types I through VI. *Archives of Dermatology* 124(6):869–871. Defined the six-point skin-type scale (I = always burns/never tans → VI = never burns, deeply pigmented) that dermatology AI now uses to describe who is in a dataset. (https://doi.org/10.1001/archderm.1988.01670060015008)

2. **Groh et al. 2021** — Evaluating Deep Neural Networks Trained on Clinical Images in Dermatology with the Fitzpatrick 17k Dataset. *CVPR 2021 Workshops (ISIC Skin Image Analysis)*. Labeled 16,577 clinical images by Fitzpatrick type and showed the dataset has far more light-skin than dark-skin images, and that a model is most accurate on the skin types it saw most during training. (https://arxiv.org/abs/2104.09957)

3. **Daneshjou et al. 2022** — Disparities in dermatology AI performance on a diverse, curated clinical image set. *Science Advances* 8(32):eabq6147. Built the Diverse Dermatology Images (DDI) set of biopsy-confirmed images and found state-of-the-art AI models drop sharply in accuracy on dark skin tones and uncommon diseases. (https://doi.org/10.1126/sciadv.abq6147)

4. **Obermeyer et al. 2019** — Dissecting racial bias in an algorithm used to manage the health of populations. *Science* 366(6464):447–453. A widely used health algorithm underserved Black patients because it predicted future cost instead of illness, and unequal access meant less was spent on equally sick Black patients — a bias hidden behind good "average" numbers. (https://doi.org/10.1126/science.aax2342)

5. **Wen et al. 2022** — Characteristics of publicly available skin cancer image datasets: a systematic review. *The Lancet Digital Health* 4(1):e64–e74. Reviewed 21 open datasets (106,950 images) and found most came from Europe, North America, and Oceania, with almost no images labeled as African, Afro-Caribbean, or South Asian skin. (https://doi.org/10.1016/S2589-7500(21)00252-1)

6. **Adamson & Smith 2018** — Machine learning and health care disparities in dermatology. *JAMA Dermatology* 154(11):1247–1248. Warns that because benchmark skin-image archives are collected mostly from fair-skinned patients, AI trained on them can widen, not close, care gaps for people with darker skin. (https://doi.org/10.1001/jamadermatol.2018.2348)

7. **Buster, Stevens & Elmets 2012** — Dermatologic health disparities. *Dermatologic Clinics* 30(1):53–59. Documents that skin-cancer and skin-disease outcomes are worse for minority, low-income, rural, and uninsured patients, partly because dermatologists cluster in wealthy urban areas — the access gap AI is often pitched to fix. (https://doi.org/10.1016/j.det.2011.08.002)

## Positioning

Skin-cancer AI is often sold as a way to reach the ~3 billion people worldwide with little dermatology access [3, 7]. But the data these models learn from is skewed: reviews of public skin-image datasets find most images come from light-skinned patients in Europe, North America, and Oceania, with almost none labeled as dark skin [5, 2]. That skew matters because a model learns best what it sees most — Groh et al. showed accuracy is highest on the skin types that dominate the training set [2], and Daneshjou et al. showed top models fail more often on dark skin [3]. So a model can look great "on average" and still miss cancers on underrepresented patients — the same trap Obermeyer et al. found in a health algorithm that scored fairly on paper but quietly underserved Black patients [4]. Using the Fitzpatrick scale [1] we can name this gap. Our toy DermaMNIST model can show that overall accuracy hides per-group differences, but DermaMNIST ships without skin-tone labels — so we can demonstrate the *risk* of a blind spot, not measure the real gap [5, 6].

## Intro figure specs

1. **Dataset skin-tone skew (adapted from Wen 2022 / Groh 2021).** A simple bar chart of image counts by Fitzpatrick type I–VI, tall on the left (light skin, types I–III) and shrinking to almost nothing on the right (dark skin, types V–VI). Use a light-to-dark color ramp for the bars so the visual and the label agree. One takeaway line: "The AI mostly practiced on light skin." Save the bar values to `figures/raw/dataset_skew.csv`.

2. **Same model, different accuracy by group (adapted from Daneshjou 2022 / Groh 2021).** A grouped bar or dumbbell chart showing one model's accuracy on light-skin vs dark-skin test images, with a dashed "overall average" line sitting between them. The point for HS students: the single average number hides that one group is doing much worse. Save the plotted accuracy values to `figures/raw/accuracy_by_group.csv`.

3. **Proxy-gone-wrong schematic (concept from Obermeyer 2019).** A two-arrow flow diagram: "What we wanted to predict (who is sick)" vs "What the algorithm actually predicted (who costs money / who looks typical)," with a small gap symbol showing where bias sneaks in. Keep it label-driven, not data-driven — it teaches *why* a fair-looking metric can still be unfair. Optionally save the node/edge labels to `figures/raw/proxy_schematic.json`.
