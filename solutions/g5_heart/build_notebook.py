"""Generate the G5 HEART-DISEASE solution notebook (the worked answer key).

Comprehensive, HS-accessible, and EVERY line of code is commented. Runs top to bottom and
reproduces the real results (logreg ~0.81; cholesterol alone is weak; the model is more accurate
for women than men -- a fairness gap). Edit this script + rebuild; never hand-edit the .ipynb.
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

**This is the worked answer key.** It runs top to bottom and shows the whole project: the medicine,
the idea, the code (every line commented), the results, and an honest look at the limits.

Our priority is **fairness across sex** -- heart disease is famously *under-diagnosed in women*, so we
will not just chase one accuracy number; we will ask **who the model works for.**

You do **not** need to memorize the Python. You *do* need to be able to say, in plain English, what
each cell is for. Anything unclear -- highlight the line and ask Claude "explain this."
"""),
        md("""
## 1. Background: how do doctors estimate heart risk?

A doctor never diagnoses heart disease from one number. They combine several routine checkup
measurements -- age, blood pressure, cholesterol, blood sugar, and what your heart does under stress --
into a single **risk estimate**. The classic tool is the **Framingham Risk Score** [2], and it is
deliberately **sex-specific**: it uses *different math for women and men*, because the same cholesterol
number does not mean the same thing in both.

Why combine many clues? The worldwide **INTERHEART** study [3] found that about **nine** risk factors --
led by abnormal cholesterol and smoking -- explain over **90%** of first heart attacks. No single cause;
risk is a *stack* of factors. (See the intro figure `intro_risk_factor_stack.png`.)

**The fairness problem we care about most:** heart disease is under-diagnosed in women.

- Women more often have **"atypical" symptoms** -- back pain, nausea, shortness of breath -- instead of
  the classic crushing chest pain, so their heart attacks get missed [4][5].
- Women are more likely to be told their arteries look **"normal"** even when something is truly wrong [6].
- The American Heart Association's own statement [4] documents that women are **misdiagnosed and
  under-treated** more than men.

Our data is the **Cleveland Heart Disease** dataset [1][7]: 303 patients from a 1980s study, each with
**13 checkup features**. Keep the fairness lens on the whole time -- and notice up front that this data is
about **2:1 male** (see `intro_dataset_sex_skew.png`), which will come back to bite us.
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
print(rate.round(3))                                            # e.g. men have a higher disease rate here
"""),
        md("""
Two things to notice, and both matter later:

1. **There are about twice as many men as women.** A model learns from examples -- with fewer women, it
   gets less practice on them. This is the biased-data lesson at the heart of the project.
2. **The disease rate is different by sex.** In *this* 1980s sample, men were referred for testing when
   sicker, so the male rows skew sicker. That is a quirk of how the data was collected, not a fact about
   women today -- exactly the kind of thing that makes a model unfair if you are not watching.
"""),
        md("""
## 2. Build a baseline: four models, one honest number

Tabular data (rows of numbers) is the **easy** part -- any standard model handles it, and none needs
tuning to get a reasonable score. So the interesting questions are **not** "which fancy model wins."
They are *which clues matter* and *who the model works for.*

We train **four** models -- Logistic Regression, Random Forest, CatBoost, and TabPFN (a foundation model
for tables, like the one from Day 2) -- and compare their **test accuracy** (graded on patients the model
never trained on). We always compare against the **"just guess the majority"** baseline: if you predicted
"no disease" for everyone, how often would you be right? A real model has to beat that.
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

results = {}                                            # a place to collect each model's accuracy
for name, make in ct.make_models().items():             # loop over the 4 models (name, and a function that builds it)
    model = make()                                      # build a fresh model
    model.fit(Xtr, ytr)                                 # train it on the study set only
    pred = np.asarray(model.predict(Xte)).ravel()       # ask it to predict the held-out test patients
    acc = float((pred == yte).mean())                   # accuracy = fraction of test patients it got right
    results[name] = acc                                 # remember this model's score
    print(f"{name:24s} test accuracy = {acc:.3f}")      # print it, lined up in a column
"""),
        md("""
All four land around **0.79-0.81**, comfortably above the ~0.54 coin-flip baseline, and **basically tie
each other.** That is the whole lesson of this slide: on clean tabular data, *the model barely matters.*
Swapping a plain logistic regression for a fancy foundation model buys you almost nothing. **The data
sets the ceiling** -- 300 patients from the 1980s only contain so much signal, and no model can invent more.

So we stop model-shopping and spend our effort on the two questions that actually teach us something.
"""),
        md("""
## 3. Result #1 -- does cholesterol alone predict heart disease?

Everyone "knows" cholesterol causes heart disease, and INTERHEART [3] ranks it the **top** modifiable risk
factor. So here is a fair test: give a model **only** the cholesterol column and nothing else. If
cholesterol were the whole story, that should be plenty. We compare three setups:

- **all 13 features** -- the full checkup.
- **cholesterol only** -- one number per patient.
- **everything except cholesterol** -- the checkup with cholesterol removed.
"""),
        code("""
# Feature ablation: retrain the SAME model on different subsets of clues and compare.
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
**weak** predictor on its own, and other clues (chest-pain type, the exercise-ECG results) carry most of
the signal.

Does that mean cholesterol doesn't matter for the heart? **No** -- INTERHEART [3] is right that it is a
major *causal* risk factor over decades. It just isn't a good one-shot *diagnostic* clue for who already
has disease *right now*, especially in a small sample. This is the difference between **causing** a disease
slowly and **detecting** it today -- and it is exactly why doctors combine many measurements [2] instead of
trusting any single number. *How you framed the question mattered more than which model you picked.*
"""),
        md("""
## 4. Result #2 -- the fairness check: does it work equally for women and men?

This is the point of the whole project. A single accuracy number can **hide** unfairness: a model can look
great overall and still be worse for one group. So we train once, then grade the model **separately** for
women and for men, and look at the gap. Given the under-diagnosis literature [4][5][6] and our 2:1-male
data, we should be suspicious before we even look.
"""),
        code("""
# FAIRNESS AUDIT: train one model, then report its accuracy for each sex separately.
# ct.audit_by_group trains on the study set, predicts the test set, and breaks the score down by group.
fairness = ct.audit_by_group(                           # returns a dict {group name: accuracy} and draws a bar chart
    df,                                                 # the patient table
    features=feats,                                     # use all 13 checkup features
    target=meta["target"],                              # predict heart disease
    group="sex",                                        # split the score by the sex column...
    group_names={0: "female", 1: "male"},               # ...and label 0/1 as female/male
    model="Logistic Regression",                        # the model we are auditing
)
print(fairness)                                         # the accuracy for women vs men
gap = abs(fairness["female"] - fairness["male"])        # how far apart the two groups are
print(f"fairness gap (difference between the two): {gap:.3f}")   # our headline fairness number
"""),
        md("""
The model is **~0.88 accurate for women but only ~0.78 for men** -- a gap of about **0.09** (9 percentage
points). Notice the twist: here it is *more* accurate for women, mostly because so few women in this sample
had disease that "predict healthy" is usually right for them. That is not good news -- an accuracy gap in
*either* direction means the model behaves differently depending on your sex, and "accurate because it
rarely says you're sick" is exactly the failure mode behind real-world under-diagnosis [4][6].

**The lesson holds regardless of direction:** one overall number ("81% accurate!") hid a real difference
between groups. You only saw it because you *looked.* A model you would deploy in a clinic must be checked
group by group, not just in aggregate.
"""),
        md("""
## 5. Honest limits

- **It can** show that routine checkup features carry real signal, that no single number (not even
  cholesterol) is the whole story [3], and -- most importantly -- that accuracy can be **unequal across
  sex** even when the overall score looks fine.
- **It cannot** say anything trustworthy about **today's** patients: this is **303 people from one 1980s
  study** [1][7]. Medicine, populations, and tests have all moved on.
- **The data is ~2:1 male** [1], and the *way* patients were selected made the male rows sicker. A model
  can only learn from who is in its data -- so it may serve **women** worse in the real world, which is the
  exact bias [4][5][6] we set out to study. The skew is itself the lesson.
- **Accuracy is the wrong single metric** for a screening test anyway: missing a real heart attack (a false
  negative) is far worse than a false alarm, so a deployed tool would be tuned for **sensitivity**, not raw
  accuracy.

**Bottom line:** use this to build intuition about risk factors and fairness. Use a validated, sex-specific
tool [2] and a real clinician to actually care for a patient.

## References
[1] Detrano et al. 1989, *Am J Cardiol* (the Cleveland dataset) &middot; [2] Wilson et al. 1998,
*Circulation* (Framingham Risk Score) &middot; [3] Yusuf et al. 2004, *Lancet* (INTERHEART) &middot;
[4] Mehta et al. 2016, *Circulation* (AHA statement on MI in women) &middot; [5] van Oosterhout et al.
2020, *JAHA* (sex differences in symptoms) &middot; [6] Bugiardini & Bairey Merz 2005, *JAMA* (angina with
"normal" arteries) &middot; [7] UCI Machine Learning Repository -- Heart Disease Data Set.
"""),
    ]
    save(nb, HERE / "solution.ipynb")
    print("wrote solution.ipynb")


if __name__ == "__main__":
    build()
