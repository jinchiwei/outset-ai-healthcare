"""Generate the G4 skin-cancer AI + EQUITY solution notebook (the worked answer key).

The whole point is the equity lens on a REAL, working skin-cancer screen. We use PAD-UFES-20 because,
unlike almost every other skin dataset, it records the patient's Fitzpatrick SKIN TYPE -- so we can
actually audit fairness by skin tone. The honest finding: across the tones we CAN measure the model is
already roughly fair (AUC gap ~0.01), but the dark-skin group has only 11 patients out of 1,494 -- far
too few to validate. That under-representation IS the equity crisis, and the fix is collecting
representative data (DDI, Fitzpatrick 17k), not a resampling trick.

The notebook runs top to bottom reading the saved results.json + figures/raw CSVs (no GPU, no retrain),
and EVERY code line is commented. Edit this script + rebuild; never hand-edit the .ipynb.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from nbutil import code, md, new_nb, save  # noqa: E402

HERE = Path(__file__).resolve().parent


def build():
    nb = new_nb()
    nb.cells = [
        md("""
# G4 SOLUTION -- Skin-cancer AI equity: fair for whom?

**This is the worked answer key for the EQUITY project.** It runs top to bottom and walks the whole
investigation the way the five-part rubric asks for it: the **background** (dermatology AI, and who it
is trained on), the **data** (why PAD-UFES-20 -- it records skin tone), the **model** (a real working
screen, and exactly how it is built), the **results** (it works, it is fair where we can measure, and
the real crisis is who is *missing*), and an honest **takeaway**.

Most skin-cancer AI is judged by one question: *how accurate is it?* We ask a harder one: **fair for
whom?** A model can look great on average and still be untested for an entire group of patients. The
twist in this project: to even *ask* that question you need data that records who each patient is --
and this dataset does.

You do **not** need to memorize the Python. You *do* need to be able to say, in plain English, what
each cell is for. Anything unclear -- highlight the line and ask Claude "explain this."
"""),
        # ---------------------------------------------------------------- 1. BACKGROUND
        md("""
## 1. Background: dermatology AI, and who it is trained on

**The promise.** Roughly 3 billion people have little or no access to a dermatologist [7]. Skin-cancer
AI is often pitched as the fix: point a phone at a mole, get an answer, no specialist required. If it
works, that is a genuinely big deal for access to care.

**The catch.** A model only learns what it is shown -- and the images these models learn from are badly
skewed toward light skin. A systematic review of 21 public skin datasets (106,950 images) found almost
all came from Europe, North America, and Oceania, with **almost no images labeled as darker skin** [5].
On the Fitzpatrick skin-type scale (I = lightest, VI = darkest) [1] the training data piles up on the
light end and nearly vanishes on the dark end:

![The same exam, far less practice on dark skin](figures/intro_skin_tone_exam.png)

Groh et al. built the Fitzpatrick 17k dataset and showed the consequence directly: **a model is most
accurate on the skin types it saw most during training** [2]. Daneshjou et al. then tested
state-of-the-art models on a diverse, biopsy-confirmed set (DDI) and found accuracy **drops sharply on
dark skin** [3]. Same model, very different results depending on who you are:

![A model is best at the skin it saw most](figures/intro_groh_2021.png)
"""),
        md("""
The people that skew leaves out are, painfully, the same people the technology was pitched to help --
those with the least access to a specialist in the first place [6][7]. A tool that works worst exactly
where care is scarcest can **widen** the gap it was sold to close.

![Roughly 3 billion people have little access to a dermatologist](figures/intro_access_gap.png)

The equity lens we bring is exactly this suspicion: **do not trust one average number.** Break the
score apart and ask who it works for -- and be honest about who you cannot even check.
"""),
        # ---------------------------------------------------------------- 2. DATA
        md("""
## 2. The data: PAD-UFES-20, and why it is special

To audit fairness by skin tone, you need data that **records** skin tone. Most skin datasets do not --
they carry the lesion and nothing about the patient, so a skin-tone gap is invisible to you no matter
how carefully you look. **PAD-UFES-20 is the rare exception:** real clinical skin images collected in
Brazil that record each patient's **Fitzpatrick skin type** [1]. That single column is what makes this
whole audit possible.
"""),
        code("""
# Load the numbers the full experiment saved, so every figure here matches the canonical run exactly.
import json                                                 # json reads the saved results file
import pandas as pd                                         # pandas reads the small saved data tables
R = json.load(open("results.json"))                         # the headline numbers from the real 224px run
print("dataset :", R["dataset"])                            # PAD-UFES-20, with its Fitzpatrick skin type
print("patients:", R["n"])                                  # total patients in the audit
"""),
        code("""
# Look at the data before modeling: how many patients of each skin tone does the dataset actually hold?
tones = pd.read_csv("figures/raw/tone_distribution.csv", index_col=0)   # the per-tone patient counts
print(tones)                                                # I-II (light), III-IV (medium), V-VI (dark)
light = int(tones.loc["I-II (light)", "patients"])          # how many light-skin patients
medium = int(tones.loc["III-IV (medium)", "patients"])      # how many medium-skin patients
dark = int(tones.loc["V-VI (dark)", "patients"])            # how many DARK-skin patients
print(f"\\nlight: {light}   medium: {medium}   dark: {dark}")   # the skew, in three numbers
print(f"dark skin is {dark/ (light+medium+dark):.1%} of the dataset -- only {dark} people.")  # tiny
"""),
        md("""
There it is, before we train anything: **1,029** light-skin patients, **454** medium, and just **11**
dark-skin patients out of 1,494. We *can* measure fairness for light and medium skin. For dark skin we
have 11 people -- hold that number; it becomes the whole story.

![The real crisis: 11 dark-skin patients out of 1,494](figures/tone_distribution.png)
"""),
        # ---------------------------------------------------------------- 3. MODEL
        md("""
## 3. The model: a real working screen, built honestly

The audit only means something if the underlying screen actually works -- auditing a broken model tells
you nothing. So first we build a genuine one. A blank network knows nothing about edges or color, and a
dataset this size is far too small to teach it from scratch, so we **borrow a brain**: start from a
network pretrained on millions of everyday photos and fine-tune it on skin (transfer learning). We use
the strong **CAFormer** backbone, frame the task as **malignant vs. benign** (biopsy-proven cancers
versus the rest), and -- crucially -- **split by patient**, so the same person's lesions never land in
both train and test.
"""),
        code("""
# Print the headline screening number the full experiment saved.
print(f"patients graded    : {R['n']}")                            # the audit cohort
print(f"AUC (baseline CAFormer) : {R['overall_auc_baseline']:.3f}   <- a working screen (>= 0.8)")  # headline
"""),
        md("""
**AUC 0.83** on held-out patients -- a genuinely working screen. (AUC is the area under the ROC curve:
0.5 is a coin flip, 1.0 is perfect. We use it instead of accuracy because a rare-cancer dataset can be
gamed by a lazy "always benign" guess.) With a real screen in hand, we can finally ask the equity
question honestly.

### Model & data processing (the exact recipe)

So the whole thing is reproducible, here is the full pipeline in one place:

- **Model:** `caformer_s18`, pretrained, then **fine-tuned** end-to-end with a fresh 2-class head [2].
- **Images:** every clinical photo decoded to color and **resized to 224x224**, then **normalized** to
  the fixed mean/scale the backbone was trained on.
- **Task:** **malignant vs. benign** -- BCC, SCC and melanoma count as malignant, everything else benign.
- **Split:** a **patient-level** train/test split, so no patient appears in both -- the number we report
  is on patients the model never saw.
- **Audit axis:** the patient's recorded **Fitzpatrick skin type**, grouped into light (I-II), medium
  (III-IV), and dark (V-VI).
"""),
        # ---------------------------------------------------------------- 4. RESULTS
        md("""
## 4. Results: fair where we can measure -- and the crisis where we cannot

Now the equity audit. We compute the screen's AUC **separately for each skin-tone group** and compare.
Because the dataset records skin type, this is a real measurement, not a guess.
"""),
        code("""
# Compare the screen's AUC across skin tones -- the fairness audit the recorded skin type makes possible.
byt = R["auc_by_tone_baseline"]                             # AUC computed separately per skin-tone group
for g, auc in byt.items():                                  # walk each tone group we could score...
    print(f"  {g:<16s} AUC = {auc:.3f}")                    # ...and print how well the screen ranks it
print(f"\\ngap between measurable tones = {R['gap_baseline']:.3f}  <- tiny; already roughly fair")
print("note: V-VI (dark) is ABSENT here -- with 11 patients you cannot even compute a stable AUC.")
"""),
        md("""
Across the tones we can measure, the screen is **already roughly fair**: AUC 0.836 on light skin and
0.829 on medium, a gap of only about **0.01**. That is the good news, and it is real. But notice what is
missing from that list: the **dark-skin group never appears**, because you cannot compute a trustworthy
AUC from 11 patients. The fairness we can see is genuine; the fairness we most need to check is
**invisible to us for lack of data.**
"""),
        code("""
# Show that a tone-balanced RETRAIN does not rescue this -- and, honestly, cannot.
mit = R["auc_by_tone_mitigated"]                            # AUC per tone AFTER oversampling rarer tones
for g in byt:                                               # for each measurable tone...
    print(f"  {g:<16s} baseline {byt[g]:.3f}  ->  rebalanced {mit[g]:.3f}")   # before vs after the retrain
print(f"\\ngap: {R['gap_baseline']:.3f} -> {R['gap_mitigated']:.3f}   (rebalancing did NOT shrink it)")
print("and it never could for DARK skin: resampling 11 patients cannot invent the data that isn't there.")
"""),
        md("""
This is the honest heart of the project. A common "fairness fix" is to **oversample** the
under-represented group during training. We tried it. Across the measurable tones it changed essentially
nothing (the gap stayed ~0.01), and for **dark skin it is powerless by construction**: repeating the
same 11 patients over and over does not teach the model what dark-skin cancers actually look like. You
cannot resample your way out of missing data.

![Fair where measurable -- rebalancing can't reach dark skin](figures/equity_before_after.png)
"""),
        md("""
### Feature importance for images: Grad-CAM

For a table of numbers we would ask SHAP "which columns mattered?" For an image, the equivalent is
**Grad-CAM**: it highlights *where* on the photo the model looked. We want it looking at the lesion, not
at a pen mark, a hair, or the corner of the frame.

![Where the model looks](figures/gradcam.png)

The bright areas sit **on the pigmented lesion** -- reassuring evidence the screen reads the spot itself,
the same cues a clinician uses. Grad-CAM tells us the model is looking in the right *place*; it cannot
tell us the model works on skin tones it never trained on. Only data can answer that.
"""),
        # ---------------------------------------------------------------- 5. TAKEAWAY
        md("""
## 5. Takeaway: you can only be fair to the people the data remembers

- **It works, and we could audit it.** A real malignant-vs-benign screen, **AUC 0.83**, and -- because
  PAD-UFES-20 records skin type -- we could actually *measure* fairness across tones, which most skin
  datasets make impossible. Auditing needs recorded demographics; this dataset has them.
- **It is fair where we can see.** Light vs. medium skin differ by only ~0.01 AUC. Naming that clearly
  is part of doing this honestly.
- **The real crisis is who is missing.** Only **11** dark-skin patients out of 1,494 -- too few to
  validate the screen for the very group the literature warns about most [2][3]. And we still record no
  race or country, so those gaps stay invisible here.
- **The fix is data, not a trick.** Rebalancing cannot invent dark-skin patients. Fixing dark-skin
  performance means **collecting representative data**, which is exactly what efforts like DDI [3] and
  Fitzpatrick 17k [2] set out to do.

**Bottom line:** before trusting a medical AI, ask *fair for whom?* -- then check whether the data even
lets you answer. A model you can audit honestly, and whose blind spots you can name, is worth more than
a high number you cannot question.

## References
[1] Fitzpatrick 1988, *Arch Dermatol* -- the six-point skin-type scale.
[2] Groh et al. 2021, *CVPR Workshops* -- Fitzpatrick 17k; accuracy tracks representation.
[3] Daneshjou et al. 2022, *Sci Adv* -- Diverse Dermatology Images (DDI); models drop on dark skin.
[4] Obermeyer et al. 2019, *Science* -- a fair-looking health algorithm that underserved Black patients.
[5] Wen et al. 2022, *Lancet Digital Health* -- most public skin images are light skin.
[6] Adamson & Smith 2018, *JAMA Dermatology* -- AI can widen dermatology care gaps.
[7] Buster, Stevens & Elmets 2012, *Dermatologic Clinics* -- dermatologic health disparities and access.
"""),
    ]
    save(nb, HERE / "solution.ipynb")
    print("wrote solution.ipynb")


if __name__ == "__main__":
    build()
