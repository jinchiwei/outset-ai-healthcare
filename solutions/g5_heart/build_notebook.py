"""Generate the G5 HEART-DISEASE solution notebook (the worked answer key).

HS-accessible, and EVERY line of code is commented. It runs top to bottom the way the five-part
rubric asks for it: the BACKGROUND (why heart risk matters, and why women get under-diagnosed), the
DATA (why the UCI Cleveland set, and looking at it first), the MODEL (a four-way bake-off, and the
exact recipe), the RESULTS (graded by AUC = 0.90, which measurements drive the call via SHAP, the
cholesterol-alone ablation, and a fairness audit by sex), and an honest CONCLUSION.

It reproduces the real numbers: all four models tie near 0.80 accuracy (the data sets the ceiling),
graded properly by AUC the tool reaches 0.90, cholesterol alone is a coin flip (0.52), and the model
is more accurate for women than men -- a real fairness gap on 2:1-male data.

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
# G5 SOLUTION -- Predicting heart disease from a checkup

**This is the worked answer key.** It runs top to bottom and shows the whole project the way the
five-part rubric asks for it: the **background** (why this matters), the **data** (why the Cleveland
set), the **model** (four options, and which we pick), the **results** (how we grade, the working
number, which clues matter, and who it works for), and an honest **conclusion**.

Our job: look at **13 routine checkup numbers** for a patient -- age, blood pressure, cholesterol,
chest-pain type, exercise-test results -- and estimate the risk that they have **heart disease**. Our
priority the whole way through is **fairness across sex**: heart disease is famously *under-diagnosed
in women*, so we will not just chase one accuracy number -- we will ask **who the model works for.**

You do **not** need to memorize the Python. You *do* need to be able to say, in plain English, what
each cell is for. Anything unclear -- highlight the line and ask Claude "explain this."
"""),
        # ---------------------------------------------------------------- 1. BACKGROUND
        md("""
## 1. Background: how do doctors estimate heart risk?

A doctor never diagnoses heart disease from one number. They combine several routine checkup
measurements -- age, blood pressure, cholesterol, blood sugar, and what the heart does under stress --
into a single **risk estimate**. The classic tool is the **Framingham Risk Score** [2], and it is
deliberately **sex-specific**: it uses *different math for women and men*, because the same cholesterol
number does not mean the same thing in both.

Why combine many clues? The worldwide **INTERHEART** study [3] found that about **nine** risk factors --
led by abnormal cholesterol and smoking -- explain over **90%** of first heart attacks. No single cause;
risk is a *stack* of factors.

**The fairness problem we care about most: heart disease is under-diagnosed in women.**

- Women more often have **"atypical" symptoms** -- back pain, nausea, shortness of breath -- instead of
  the classic crushing chest pain, so their heart attacks get missed [4][5].
- Women are more likely to be told their arteries look **"normal"** even when something is truly wrong [6].
- The American Heart Association's own statement [4] documents that women are **misdiagnosed and
  under-treated** more than men.

Keep that fairness lens on the whole time. It is why, at the very end, we do not stop at one accuracy
number -- we break the score down by sex.
"""),
        # ---------------------------------------------------------------- 2. DATA
        md("""
## 2. The data: UCI Cleveland, and look at it first

We use the **Cleveland Heart Disease** dataset [1][7]: **303 patients** from a 1980s clinical study,
each with **13 checkup features** and a verified yes/no answer for heart disease. It is the right
dataset to *learn on* for three reasons: it is **real clinical data** (not made up), it is **free and
open** -- it downloads straight from the UCI repository with no gate or login -- and, crucially for our
theme, it records each patient's **sex**, so we can later audit fairness.

The golden rule of the whole course: **load the data and actually look at it before you model.** The
first thing to check here is *who is in it*, because our whole project is about fairness across sex.
"""),
        code("""
# Load the course helper that knows how to download + clean the Cleveland Heart Disease data.
import sys                                             # sys lets us tell Python where to find helper files
sys.path.insert(0, "../../notebooks/day3_capstone")   # add the capstone helpers folder to the search path
import capstone_tabular as ct                          # import it under the short name "ct"

df, meta = ct.load_heart()                             # df = one row per patient; meta = notes about the data
print("patients:", len(df))                            # how many patients we actually have (after dropping blanks)
print("features:", meta["features"])                   # the 13 checkup measurements we get to use
print("we are predicting:", meta["target"], "(1 =", meta["positive"] + ")")   # the yes/no answer we want
df.head()                                              # show the first 5 patients so we can see what a row looks like
"""),
        md("""
Each row is one patient. The 13 features are ordinary checkup numbers. A few worth knowing:

- **`age`**, **`sex`** (1 = male, 0 = female), **`trestbps`** = resting blood pressure, **`chol`** = cholesterol.
- **`cp`** = chest-pain type, **`thalach`** = highest heart rate reached on a treadmill test,
  **`exang`** = did exercise bring on chest pain, **`oldpeak`**/**`slope`**/**`ca`**/**`thal`** = results
  from an exercise ECG and a heart-imaging test.
- **`disease`** is the answer: 1 if the patient truly has heart disease, 0 if not.

These are exactly the kinds of clues the Framingham score [2] uses -- so we are asking a model to learn
the *same job a cardiologist's risk calculator does.*
"""),
        code("""
# Quick look at WHO is in the data -- because our whole project is about fairness across sex.
print(df["sex"].map({0: "female", 1: "male"}).value_counts())   # count women vs men in the dataset
print()                                                          # blank line for readability
rate = df.groupby(df["sex"].map({0: "female", 1: "male"}))["disease"].mean()   # disease rate within each group
print("share who actually have heart disease, by sex:")         # label the next print
print(rate.round(3))                                            # men have a higher disease rate in THIS sample
"""),
        md("""
Two things to notice, and both matter later:

1. **There are about twice as many men as women (201 vs 96).** A model learns from examples -- with
   fewer women, it gets less practice on them. This is the biased-data lesson at the heart of the project.
2. **The disease rate is different by sex.** In *this* 1980s sample, men were referred for testing when
   sicker, so the male rows skew sicker. That is a quirk of how the data was collected, not a fact about
   women today -- exactly the kind of thing that makes a model unfair if you are not watching.
"""),
        # ---------------------------------------------------------------- 3. MODEL
        md("""
## 3. The model: a four-way bake-off, and which to pick

Tabular data (rows of numbers) is the **easy** part -- any standard model handles it, and none needs
heavy tuning to get a reasonable score. So instead of guessing, we run a **bake-off**: train four
different models the exact same way and compare their **test accuracy** (graded only on patients the
model never trained on). We always compare against the **"just guess the majority"** baseline: if you
predicted "no disease" for everyone, how often would you be right? A real model must beat that.

The four models are the same family as Day 2: **Logistic Regression** (simple + explainable),
**Random Forest**, **CatBoost** (gradient-boosted trees), and **TabPFN**, a foundation model for tables.
"""),
        code("""
# Train all four models the same way and compare their accuracy on held-out patients.
import numpy as np                                      # numpy does the array math
from sklearn.model_selection import train_test_split    # splits patients into a study set + a test set

feats = meta["features"]                                 # the 13 checkup features (our clues)
X = df[feats].astype(float).values                      # the clues, as a grid of numbers (one row per patient)
y = df[meta["target"]].astype(int).values               # the answers: 1 = has heart disease, 0 = does not

# Hold back 25% of patients as a TEST set the model never sees during training.
# random_state=0 makes the split identical every run; stratify=y keeps the same disease rate in both halves.
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)

baseline = max(yte.mean(), 1 - yte.mean())              # accuracy of always guessing the more common answer
print(f"guess-the-majority baseline: {baseline:.3f}\\n")# the score any real model must beat

accs = {}                                               # a place to collect each model's accuracy
for name, make in ct.make_models().items():             # loop over the 4 models (name, and a function that builds it)
    model = make()                                      # build a fresh model
    model.fit(Xtr, ytr)                                 # train it on the study set only
    pred = np.asarray(model.predict(Xte)).ravel()       # ask it to predict the held-out test patients
    accs[name] = float((pred == yte).mean())            # accuracy = fraction of test patients it got right
    print(f"{name:24s} test accuracy = {accs[name]:.3f}")   # print it, lined up in a column
"""),
        md("""
All four land around **0.79-0.81**, comfortably above the ~0.54 coin-flip baseline, and **basically tie
each other.** That is the whole lesson: on clean tabular data, *the model barely matters.* Swapping a
plain logistic regression for a fancy foundation model buys almost nothing. **The data sets the ceiling**
-- 300 patients from the 1980s only contain so much signal, and no model can invent more.

![Four models, similar scores -- the data sets the ceiling](figures/model_comparison.png)
"""),
        md("""
### Model & data processing (the exact recipe)

So the whole thing is reproducible, here is the full pipeline in one place:

- **Model:** for the headline risk score we use **CatBoost** (gradient-boosted decision trees), because
  it gives a well-calibrated probability we can grade with a ROC curve. Logistic Regression is used for
  the clean, explainable feature comparisons.
- **Features:** all **13** checkup measurements, each read as a plain number.
- **Split:** a single **75% / 25%** train/test split, `random_state=0` so it is identical every run, and
  `stratify` so both halves carry the same disease rate. Every number we report is on the **held-out**
  test set the model never trained on.
- **No leakage:** the model never sees a test patient during training, so the score is an honest estimate
  of how it would do on a new patient from the same 1980s population.
"""),
        # ---------------------------------------------------------------- 4. RESULTS
        md("""
## 4. Results: how we measured, and what we found

**How we grade a risk tool: ROC and AUC.** A risk tool outputs a *score* (a probability), not a hard
yes/no. The **ROC curve** sweeps every possible cut-off and plots disease caught (sensitivity) against
false alarms. **AUC** is the area under that curve: 0.5 is a coin flip, 1.0 is perfect. AUC rewards a
model for *ranking* sick patients above healthy ones, which is exactly what a risk score should do -- and
it is a fairer grade than accuracy on data where the two groups are not the same size.
"""),
        code("""
# Load the canonical results the full experiment saved, so every number here matches the slides exactly.
import json                                             # json reads the saved results file
R = json.load(open("results.json"))                     # load the numbers the full experiment saved
print(f"patients graded (test set)     : {R['n']} total in the study")   # the cohort size
print(f"best plain accuracy            : {R['best_acc']:.2f}  (all four models tie near here)")  # accuracy ceiling
print(f"AUC of the risk tool (CatBoost): {R['auc']:.2f}   <- a genuinely working risk tool")     # the headline
"""),
        md("""
**AUC 0.90** -- a genuinely working risk tool. Notice the gap between the two numbers: the best plain
*accuracy* is only ~0.81, but graded properly by *AUC* the tool scores **0.90**. That is the point of
picking the right metric: accuracy throws away the score and just checks the final yes/no, while AUC
credits the model for confidently ranking the sickest patients at the top. The ROC curve below hugs the
top-left corner.

![A working risk tool, AUC 0.90](figures/roc.png)
"""),
        # ---------------------------------------------------------------- SHAP
        md("""
### Which measurements drive the call? (SHAP feature importance)

A risk score is more trustworthy if we can see *why* it decided what it did. **SHAP** answers "which
columns mattered?" by measuring how much each feature moves each prediction, then averaging. Bigger bar
= that measurement pushes the risk call around more.
"""),
        code("""
# Read the saved SHAP importances (mean |SHAP| per feature) and print the ranking.
import pandas as pd                                     # pandas reads the saved table
S = pd.read_csv("figures/raw/shap_importance.csv")      # one row per feature, with its importance
S = S.sort_values("importance", ascending=False)        # put the most important feature on top
print("most influential measurements (top of the SHAP chart):")   # label the list
for _, row in S.head(4).iterrows():                     # walk the top four features...
    print(f"  {row['feature']:9s} {row['importance']:.2f}")       # ...and print each one's importance
print(f"\\ncholesterol (chol) ranks low: {float(S[S.feature == 'chol']['importance'].iloc[0]):.2f}"
      "  <- remember this for the next result")         # foreshadow the ablation surprise
"""),
        md("""
The heavy hitters are **`ca`** (number of major vessels seen on imaging), **`thal`** (a stress-test
result), and **`cp`** (chest-pain type) -- the same cues a cardiologist leans on. And notice: **`chol`
(cholesterol) sits near the bottom.** That is a real surprise worth its own test.

![Which measurements drive the risk call (SHAP)](figures/shap_importance.png)
"""),
        # ---------------------------------------------------------------- ablation
        md("""
### Does cholesterol alone predict heart disease?

Everyone "knows" cholesterol causes heart disease, and INTERHEART [3] ranks it the **top** modifiable
risk factor -- yet SHAP just ranked it low. So here is a fair test: give a model **only** the cholesterol
column. If cholesterol were the whole story, that should be plenty. We compare three setups: all 13
features, cholesterol only, and everything except cholesterol.
"""),
        code("""
# Feature ablation: retrain the SAME simple model on different subsets of clues and compare.
def accuracy_with(features):                            # a small helper: train on a chosen set of columns...
    Xa = df[features].astype(float).values              # pull just those columns as numbers
    Xa_tr, Xa_te, ya_tr, ya_te = train_test_split(      # split into study + test...
        Xa, y, test_size=0.25, random_state=0, stratify=y)   # ...the exact same split as before (random_state=0)
    m = ct.make_models()["Logistic Regression"]()       # use the simple, explainable model for a clean comparison
    m.fit(Xa_tr, ya_tr)                                 # train on the study set
    pred = np.asarray(m.predict(Xa_te)).ravel()         # predict the test set
    return float((pred == ya_te).mean())                # return the test accuracy

setups = {                                              # the three feature sets we want to compare
    "all 13 features":          feats,                  # the whole checkup
    "cholesterol only":         ["chol"],               # just the cholesterol number
    "everything except chol":   [f for f in feats if f != "chol"],   # the checkup MINUS cholesterol
}
for label, cols in setups.items():                      # loop over the three setups
    print(f"{label:26s} -> {accuracy_with(cols):.3f}")  # print each one's test accuracy
"""),
        md("""
The surprise: **cholesterol alone scores ~0.52 -- basically a coin flip.** And the checkup *without*
cholesterol scores ~0.80, almost as high as the full 13 features. So in this dataset, cholesterol is a
**weak** predictor on its own, and other clues (chest-pain type, the exercise-test results) carry most of
the signal -- exactly what SHAP told us.

Does that mean cholesterol doesn't matter for the heart? **No** -- INTERHEART [3] is right that it is a
major *causal* risk factor over decades. It just isn't a good one-shot *diagnostic* clue for who already
has disease *right now*, especially in a small sample. That is the difference between **causing** a disease
slowly and **detecting** it today, and it is why doctors combine many measurements [2] instead of trusting
any single number. *How you framed the question mattered more than which model you picked.*

![Cholesterol alone is a weak predictor](figures/feature_ablation.png)
"""),
        # ---------------------------------------------------------------- fairness
        md("""
### The fairness check: does it work equally for women and men?

This is the point of the whole project. A single accuracy number can **hide** unfairness: a model can
look great overall and still be worse for one group. So we train once, then grade the model **separately**
for women and for men, and look at the gap. Given the under-diagnosis literature [4][5][6] and our
2:1-male data, we should be suspicious *before* we even look.
"""),
        code("""
# FAIRNESS AUDIT: train one model, then report its accuracy for each sex separately.
# ct.audit_by_group trains on the study set, predicts the test set, and breaks the score down by group.
fairness = ct.audit_by_group(                           # returns {group name: accuracy} and draws a bar chart
    df,                                                 # the patient table
    features=feats,                                     # use all 13 checkup features
    target=meta["target"],                              # predict heart disease
    group="sex",                                        # split the score by the sex column...
    group_names={0: "female", 1: "male"},               # ...and label 0/1 as female/male
    model="Logistic Regression",                        # the model we are auditing
)
gap = abs(fairness["female"] - fairness["male"])        # how far apart the two groups are
print(f"\\nfairness gap (difference between the two): {gap:.3f}")   # our headline fairness number
"""),
        md("""
The model is **~0.88 accurate for women but only ~0.78 for men** -- a gap of about **0.09** (9 percentage
points). Notice the twist: here it is *more* accurate for women, mostly because so few women in this
sample had disease that "predict healthy" is usually right for them. That is not good news -- an accuracy
gap in *either* direction means the model behaves differently depending on your sex, and "accurate because
it rarely says you're sick" is exactly the failure mode behind real-world under-diagnosis [4][6].

**The lesson holds regardless of direction:** one overall number ("81% accurate!") hid a real difference
between groups. You only saw it because you *looked* -- and the honest reason it exists is the 2:1 male
skew we spotted at the very start.

![Accurate overall -- but not equally for women and men](figures/fairness_by_sex.png)
"""),
        # ---------------------------------------------------------------- 5. CONCLUSION
        md("""
## 5. Conclusion: honest limits

- **It works** -- a real risk tool on real checkup data, **AUC 0.90**, that leans on the same cues a
  cardiologist uses (`ca`, `thal`, chest-pain type), and it taught us that no single number -- not even
  cholesterol -- is the whole story [3].
- **But** the accuracy is **unequal across sex** (0.88 vs 0.78), and the data is **~2:1 male** [1], with
  the male rows skewed sicker by how patients were selected. A model only learns from who is in its data,
  so it may serve **women** worse in the real world -- the exact bias [4][5][6] we set out to study.
- **A real tool** would be trained on today's diverse patients, validated across many hospitals, and use a
  **sex-specific** design like Framingham [2], with a clinician in the loop before it touches a patient.

**Bottom line:** the winning habit is not chasing a shiny accuracy number -- it is grading with the right
metric (AUC), asking *which clues actually matter*, and always checking *who the model works for.* Use
this to build that judgment; use a validated tool and a real clinician to actually care for a patient.

## References
[1] Detrano et al. 1989, *Am J Cardiol* -- the Cleveland dataset. [2] Wilson et al. 1998, *Circulation* --
the Framingham Risk Score. [3] Yusuf et al. 2004, *Lancet* -- INTERHEART. [4] Mehta et al. 2016,
*Circulation* -- AHA statement on heart attacks in women. [5] van Oosterhout et al. 2020, *JAHA* -- sex
differences in symptoms. [6] Bugiardini & Bairey Merz 2005, *JAMA* -- angina with "normal" arteries.
[7] UCI Machine Learning Repository -- Heart Disease Data Set.
"""),
    ]
    save(nb, HERE / "solution.ipynb")
    print("wrote solution.ipynb")


if __name__ == "__main__":
    build()
