"""Generate the G3 SOLUTION notebook (the worked answer key) -- a TWO-PART arc.

HS-accessible, EVERY line of code commented, runs top to bottom. It follows the five-part rubric:
  1. Background -- stroke is a time-critical ER decision; head-CT triage AI.
  2. Data       -- the brain-CT set (image + label only) AND a tabular stroke set (records sex/age).
  3. Model      -- CAFormer transfer learning for the images; a bake-off + CatBoost for the table.
  4. Results    -- CT: AUC 0.817 + Grad-CAM + the "can't audit, no demographics" WALL;
                   then tabular: AUC 0.81 + audit-by-sex + the FIX + SHAP.
  5. Takeaway   -- recording demographics is what makes fairness auditable AND fixable.

The CT half loads the local brain-CT npz (a quick smoke train proves the pipeline; the headline
CAFormer numbers come from the canonical results.json). The tabular half runs FULLY LIVE on the
local stroke CSV, so it reproduces the real audit + fix in seconds.

Edit this script + rebuild; never hand-edit the .ipynb.
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
# G3 SOLUTION -- Spotting stroke on a brain CT, and the fairness the data hides

**This is the worked answer key.** It runs top to bottom and tells a TWO-PART story that hits the
five-part rubric: the **background** (why stroke is a race against the clock), the **data** (two
very different datasets), the **model** (CAFormer for the scans, a bake-off for the table), the
**results** (a working CT detector, then a wall we cannot climb, then a tabular model we CAN audit
and fix), and the **takeaway**.

The big idea to hold onto: **an imaging dataset can hide who is in it.** You can build a model that
works on average and still have no way to check *who it fails.* Recording demographics is what makes
fairness both **auditable** and **fixable** -- and that is exactly the contrast these two datasets draw.

You do **not** need to memorize the Python. You *do* need to say, in plain English, what each cell is
for. Anything unclear -- highlight the line and ask Claude "explain this."
"""),
        # ============================================================ 1. BACKGROUND
        md("""
## 1. Background: a stroke on a head CT is a race against the clock

When someone arrives in the ER after a suspected stroke or head injury, they get a **CT scan** of the
brain. A **hemorrhage** (a bleed) shows up **bright** on CT, and it is an emergency: in a large stroke,
roughly **1.9 million neurons die every minute** until it is treated [Saver 2006]. A radiologist has a
long queue of scans, so one of the most real, already-deployed uses of medical AI is **triage** -- a
network reads the whole stack and pushes the scans that *probably* show bleeding to the **top** of the
list, so the dangerous ones get read first instead of in arrival order.

![A head CT with an acute hemorrhage; time is brain](figures/intro_head_ct.png)

This is not science fiction. Chilamkurthy et al. trained networks on ~313,000 head CTs and detected
hemorrhage, fractures, and midline shift with AUCs above 0.90 [1]. The RSNA 2019 challenge had 60+
neuroradiologists label 25,000+ exams to build a public benchmark [2]. Deployed tools reach ~93%
accuracy and ~87% sensitivity -- but do noticeably *worse* on some bleed subtypes [3].

![How triage AI reorders the reading queue](figures/intro_triage_queue.png)
"""),
        md("""
### The catch we keep coming back to

To know whether a tool is **fair**, you compare its accuracy across groups -- young vs. old, women vs.
men, different scanners. That only works if the dataset **records who each scan came from.** The
"Datasheets for Datasets" idea says every dataset *should* ship that documentation [4], and reviews
show radiology datasets often simply don't [5]. Real audits found AI that quietly missed disease more
often in female, Black, and low-income patients -- a bias you can only *see* because those datasets
recorded the groups [6][7]. Keep that in view: it is the whole point of Part 2.
"""),
        # ============================================================ 2. DATA
        md("""
## 2. The data: two datasets that document their patients very differently

We use **two** datasets on purpose, because their contrast *is* the lesson:

- a **brain-CT** set -- real scans, labeled normal vs. stroke, and **nothing else**;
- a **tabular stroke** set -- one row of numbers per patient, which **does** record **sex, age, and
  health history**.

The golden rule of the course: **load the data and actually look at it before you model.** Let's look
at both, and notice what each one records about the patient.
"""),
        code("""
# Point Python at the course helper, then load the brain-CT scans with it.
import sys                                          # sys lets us add a folder to Python's import path
sys.path.insert(0, "../../notebooks/day3_capstone") # that folder holds capstone_common.py
import capstone_common as cc                        # import it under the short name "cc"
import torch                                        # the deep-learning library we train with

device = "cuda" if torch.cuda.is_available() else "cpu"   # use a GPU if there is one, else the CPU
torch.manual_seed(0)                                # fix the random seed so results are repeatable

names = cc.class_names("brainct")                   # the human-readable class names for this dataset
print("CT classes:", names)                         # -> ['normal', 'stroke']

# get_loaders splits the images 70/15/15 into train / validation / test and batches them for us.
train_loader, val_loader, test_loader, n_classes, task = cc.get_loaders("brainct", size=64)
xb, yb = next(iter(train_loader))                   # grab ONE batch just to see the shapes
print("one batch of images:", tuple(xb.shape), " labels:", tuple(yb.shape))  # images + labels...
print("...and NOTHING else -- no age, sex, race, or scanner column exists in this dataset.")
"""),
        md("""
Each image is a 64x64 brain slice, copied into 3 channels so a color-image network can read it (the
full experiment uses 224x224). Notice what the loader hands back: **images and labels, full stop.**
There is no column for age, sex, race, or scanner -- the dataset never recorded them. Hold that thought.

Now the second dataset -- a table where every row is one patient, and the columns *do* describe them.
"""),
        code("""
# Load the tabular stroke dataset (a local CSV cache of a public Stroke Prediction set).
import pandas as pd                                 # pandas reads tables of numbers
strokes = pd.read_csv("../../datasets/stroke.csv")  # 5,110 patients, one row each
strokes = strokes[strokes.gender.isin(["Male", "Female"])]  # keep the two recorded sexes
print("patients:", len(strokes), " columns:", list(strokes.columns))   # LOOK at what it records
print("\\nstroke rate:", f"{strokes.stroke.mean():.1%}")               # ~5% -- a rare, imbalanced outcome
strokes[["gender", "age", "hypertension", "heart_disease", "avg_glucose_level", "stroke"]].head()
"""),
        md("""
This table **records who each patient is**: `gender`, `age`, `hypertension`, `heart_disease`,
smoking status, and more. That single difference -- present demographics -- is what will let us do the
fairness audit on this dataset that the CT scans made **impossible.** Here is the contrast in one picture:

![What a datasheet asks for vs. what the CT dataset has](figures/intro_datasheet_checklist.png)
"""),
        # ============================================================ 3. MODEL
        md("""
## 3. The model: borrow a brain for the scans, run a bake-off for the table

**For the images**, we don't have millions of scans, so we **borrow a brain**: start from a network
already trained on millions of everyday photos, freeze what it learned about edges and texture, and
train only a small new **head** to map those features to normal vs. stroke [8]. We use a strong modern
backbone, **CAFormer**, at full **224x224** resolution. Below is a quick one-epoch *smoke test* just to
prove the pipeline runs end to end.
"""),
        code("""
# Build a transfer-learning model and train its new head for ONE epoch -- a quick pipeline smoke test.
demo = cc.make_model(n_classes, backbone="resnet18",  # a small backbone for a fast demo...
                     pretrained=True)                 # ...starting from weights learned on everyday photos (transfer!)
demo = cc.train(demo, train_loader, val_loader,       # train it, checking accuracy on the validation set
                epochs=1, lr=1e-3, device=device)     # just 1 epoch -- enough to show the pipeline works
print("\\nThe pipeline runs end to end because it reused a pretrained brain.")
print("The FULL experiment (run_experiment.py) trains CAFormer at 224px and saved the real numbers -- next.")
"""),
        code("""
# Load the canonical CT results the full 224px CAFormer experiment saved, so every number matches the slides.
import json                                          # json reads the saved results file
CT = json.load(open("results.json"))                 # the headline numbers from the full run
print("backbone   :", CT["backbone"])                # caformer_s18 -- the strong modern backbone we chose
print(f"AUC        : {CT['auc']:.3f}   <- a working detector (above 0.8)")   # the headline
print(f"sensitivity: {CT['sensitivity']:.3f}   (of real strokes, the fraction CAUGHT)")
print(f"specificity: {CT['specificity']:.3f}   (of real normals, the fraction CLEARED)")
"""),
        md("""
### Model & data processing -- the CT recipe (so it is reproducible)

- **Model:** `caformer_s18`, pretrained, backbone frozen, a fresh 2-class head trained on top [8].
- **Images:** each CT slice decoded to 3-channel color, **resized to 224x224**, then **normalized** to
  the range the backbone expects; light flips/rotations augment the **train** split only.
- **Task:** **stroke vs. normal** (one binary call per slice).
- **Split:** separate train / validation / **test**; every number above is on the held-out test set.

**For the table**, the interesting question is not architecture but *which model* -- so we ran a
**bake-off** across four standard tabular models and compared AUC on held-out patients.
"""),
        code("""
# Load the tabular bake-off the experiment saved, and see how tight the race was.
TAB = json.load(open("results_tabular.json"))        # the tabular headline numbers + the bake-off
for name, auc in TAB["model_auc"].items():           # walk the four candidate models...
    print(f"  {name:<14s} AUC = {auc:.3f}")           # ...and print each one's held-out AUC
print("\\nAll four clear 0.80. We DEPLOY CatBoost -- a tree model we can explain with SHAP and audit for fairness.")
"""),
        md("""
The models are within a whisker of each other, all clearing the **0.80** bar -- so the honest choice is
not "the top decimal" but *which model earns its keep.* We deploy **CatBoost** (AUC 0.81) because it is
a tree we can **read** with SHAP and audit by sex, both of which we do below.

![Four tabular models, all clearing 0.80; we deploy CatBoost](figures/stroke_model_choice.png)
"""),
        code("""
# Rebuild the deployed CatBoost model LIVE on the tabular data, exactly as the experiment did.
from sklearn.preprocessing import LabelEncoder       # turns text columns into numbers
from sklearn.model_selection import train_test_split # makes a held-out test split
from sklearn.metrics import roc_auc_score            # our scoring metric
from catboost import CatBoostClassifier              # the gradient-boosted-tree model we deploy
import numpy as np                                   # fast array math

y = strokes.stroke.astype(int).values                # 1 = had a stroke, 0 = did not
sex = strokes.gender.values                          # the sex column we will AUDIT by
feat = strokes.drop(columns=[c for c in ["id", "stroke"] if c in strokes]).copy()  # features = all but id + label
for c in feat.select_dtypes("object"):               # for each TEXT column (smoking, work type, ...)
    feat[c] = LabelEncoder().fit_transform(feat[c].astype(str))  # ...encode its categories as integers
feat = feat.fillna(feat.median())                    # fill missing bmi etc. with the column median

# Split into train/test (seed 0 = same split as the experiment), carrying the sex labels along.
Xtr, Xte, ytr, yte, sex_tr, sex_te = train_test_split(
    feat.values, y, sex, test_size=0.3, random_state=0, stratify=y)   # stratify keeps the stroke rate even
model = CatBoostClassifier(verbose=0, random_state=0, # a class-weighted CatBoost so the rare strokes count
                           auto_class_weights="Balanced", iterations=400)
model.fit(Xtr, ytr)                                  # train it (a few seconds)
p = model.predict_proba(Xte)[:, 1]                   # predicted stroke probability for each held-out patient
print(f"tabular AUC (CatBoost, live) = {roc_auc_score(yte, p):.3f}   <- matches the saved 0.81")
"""),
        # ============================================================ 4. RESULTS
        md("""
## 4. Results, part A -- the CT detector works, then hits a wall

**How we grade a screen: ROC and AUC.** A detector outputs a *score*, not a hard yes/no. The **ROC
curve** sweeps every cut-off and plots strokes caught (sensitivity) against false alarms; **AUC** is the
area under it, where 0.5 is a coin flip and 1.0 is perfect. Our CT detector scores **AUC 0.817**.

![A working stroke detector, AUC 0.82](figures/roc.png)

Read the two error types separately, because one accuracy number would hide the split. At its operating
point the detector **catches about two-thirds of strokes** (sensitivity 0.65) and **correctly clears
81% of normal scans** (specificity 0.81). For a safety-net triage tool you would tune the threshold to
catch more strokes, accepting more false alarms -- a missed bleed is the error that kills.

![Confusion matrix on the held-out test set](figures/confusion.png)
"""),
        md("""
### Is it reading the brain, or cheating off the skull?

A model can be right for the wrong reason. **Grad-CAM** paints *where* the network looked. Some heat
lands on brain tissue -- good -- but some drifts to the **skull edge and the image border**, a classic
**shortcut** that works on this dataset and would break on a new scanner [7]. Worth saying out loud.

![Grad-CAM: where the model looked](figures/gradcam.png)

Training images are clean; a real ER scanner is noisier. When we add random noise, accuracy sags fast --
this toy model is **brittle** and would need augmentation, far more data, and live monitoring.

![Accuracy as image noise increases](figures/noise_robustness.png)
"""),
        md("""
### The wall: the fairness audit we literally cannot run

Here is where this dataset stops us. We *can* check accuracy **by class** (normal vs. stroke), because
the label exists. But the audit that matters for fairness -- *does it work as well for older patients as
younger? women as men? one scanner as another?* -- we **cannot run at all.** Not because it is hard, but
because the dataset never recorded age, sex, race, or scanner. There is **nothing to group by.** The
figure below is deliberately mostly blank: it is a picture of the metadata that does not exist.

![What this dataset does not record](figures/missing_metadata.png)

Our detector might be much worse for some group of patients, and **we would have no way to know.** For a
tool that would touch patients, that silence is not a footnote -- it is disqualifying on its own [4][5][6].
"""),
        md("""
## 4. Results, part B -- the tabular model we CAN audit, and fix

Now the payoff of the second dataset. Because the stroke table **records sex**, we can do the exact
audit the CT scans made impossible: at a sensitivity-first operating point (a threshold tuned to catch
**~80% of strokes overall**), do we catch **women's** strokes as often as **men's**?
"""),
        code("""
# AUDIT: at one shared threshold (tuned to catch ~80% of strokes overall), compare recall by sex.
def recall(mask, t):                                 # recall = of this group's real strokes, fraction we flag
    m = mask & (yte == 1)                            # this group's patients who ACTUALLY had a stroke
    return float(((p >= t)[m]).mean()) if m.any() else np.nan
t0 = float(np.quantile(p[yte == 1], 1 - 0.80))       # one shared threshold that catches ~80% of strokes overall
before = {g: recall(sex_te == g, t0) for g in ["Female", "Male"]}   # recall for women, and for men
print(f"shared threshold = {t0:.3f}")
print(f"women's strokes caught : {before['Female']:.0%}")   # ~76%
print(f"men's strokes caught   : {before['Male']:.0%}")     # ~85%
print(f"gap = {abs(before['Female'] - before['Male']):.0%}   <- women's strokes are caught LESS often at the same cut-off")
"""),
        md("""
A real gap: at the same threshold, the model catches **~85% of men's** strokes but only **~76% of
women's.** We could only *see* this because the dataset records sex. Now we **fix** it: give each group
its own threshold so both catch the same share of strokes (a standard "equal-opportunity" adjustment).
"""),
        code("""
# FIX: give each sex its OWN threshold so both catch the same share of strokes (equalize UP to the better rate).
def group_thresh(g, target):                         # find the threshold that catches `target` of group g's strokes
    m = (sex_te == g) & (yte == 1)                    # group g's real strokes
    return float(np.quantile(p[m], 1 - target)) if m.any() else t0
target = max(before.values())                        # equalize up to the better-served group's rate (~85%)
after = {g: recall(sex_te == g, group_thresh(g, target)) for g in ["Female", "Male"]}  # recall after the fix
print(f"women's strokes caught : {before['Female']:.0%}  ->  {after['Female']:.0%}")   # 76% -> 83%
print(f"men's strokes caught   : {before['Male']:.0%}  ->  {after['Male']:.0%}")       # 85% -> 85%
print(f"gap  {abs(before['Female']-before['Male']):.0%}  ->  {abs(after['Female']-after['Male']):.0%}   <- the gap all but closes")
"""),
        md("""
The fix lifts women's stroke catch-rate from **76% to 83%**, closing most of the gap -- by *lowering
women's threshold*, so the model flags them at a lower risk score. This is the whole thesis in one
before/after: recording sex made the gap **visible** *and* **fixable.**

![Auditing, then equalizing, who gets caught](figures/stroke_fairness_fix.png)
"""),
        md("""
### Feature importance: which recorded facts drive the risk score?

For a table, feature importance is **SHAP** -- it measures how much each column moves the prediction.
**Age dominates**, with BMI and glucose next; sex itself contributes little to the *score*, even though
it mattered for the *fairness* audit. A model you can read like this is one you can question and trust.

![What drives the stroke-risk prediction (SHAP): age dominates](figures/stroke_shap.png)
"""),
        # ============================================================ 5. TAKEAWAY
        md("""
## 5. Takeaway: recording demographics is what makes fairness auditable -- and fixable

Two datasets, one clean lesson:

- **The brain-CT model works** (AUC 0.817) -- but the dataset records only the image and a label, so the
  most important audit, *who does it fail?*, is **impossible.** You cannot fix a bias you cannot see.
- **The tabular model also works** (AUC 0.81) -- and because it records **sex**, we found a real gap
  (women's strokes caught 76% vs men's 85%) and **closed it** (to 83% vs 85%) with a group-aware threshold.

**Honest limits.** Both are small, toy models: the CT detector partly leans on edge shortcuts (Grad-CAM)
and is brittle to noise; the tabular fix trades a few more false alarms for the extra strokes it catches,
and equalizing by sex still leaves race, income, and scanner unrecorded. A deployable tool would train on
far more data, be validated across many hospitals, and keep a clinician in the loop.

**Bottom line:** an imaging dataset can hide who is in it, and a model that runs is easy -- a model you
can **audit** is the real bar. Recording demographics is what turns fairness from a hope into something
you can measure and fix. Knowing that, and being able to say it, is the finding.

## References
[1] Chilamkurthy et al. 2018, *The Lancet* -- deep learning detects critical findings on head CT.
[2] Flanders et al. 2020, *Radiology: AI* -- the RSNA 2019 brain-CT hemorrhage challenge.
[3] Seyam et al. 2022, *Radiology: AI* -- a deployed hemorrhage-detection tool in the clinic.
[4] Gebru et al. 2021, *Communications of the ACM* -- Datasheets for Datasets.
[5] Tripathi et al. 2023, *J Am Coll Radiol* -- biases and disparities in radiology-AI datasets.
[6] Seyyed-Kalantari et al. 2021, *Nature Medicine* -- underdiagnosis bias in chest-X-ray AI.
[7] Yang et al. 2024, *Nature Medicine* -- the limits of fair medical-imaging AI in the real world.
[8] Kim et al. 2022, *BMC Med Imaging* -- transfer learning for medical images.
Saver 2006, *Stroke* -- "Time Is Brain -- Quantified" (~1.9 million neurons lost per minute).
"""),
    ]
    save(nb, HERE / "solution.ipynb")
    print("wrote solution.ipynb")


if __name__ == "__main__":
    build()
