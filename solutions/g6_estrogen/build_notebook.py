"""Generate the G6 estrogen-&-cognition SOLUTION notebook (the worked answer key).

This is a CAUSAL question, so the whole point is getting the RIGHT answer where a naive analysis
gets it wrong. It runs top to bottom and follows the five-part rubric:
  1 Background -- women's cognition, dismissed symptoms, and the 20-year observational-vs-trial flip.
  2 Data       -- NHANES women 60+, and why a national survey is the right (and only) raw material.
  3 Model      -- WHY logistic regression: a causal question REQUIRES a model you can READ. A black
                  box predicts but can't hand you an adjustable effect size, so it can't answer this.
  4 Results    -- the tempting headline ("estrogen users score better") collapses: 67% of it was
                  confounding; feature importance from the SAME model; the honest predictive AUC 0.78.
  5 Conclusion -- correlation != causation; the right method flips the answer; only a trial proves it.

EVERY line of code is commented and HS-accessible. Validate headless before shipping (see BUILD_SPEC).
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
# G6 SOLUTION -- Does estrogen keep older women's minds sharp?

**This is the worked answer key.** It runs top to bottom and shows the whole project the way the
five-part rubric asks for it: the **background** (why this matters), the **data** (why NHANES), the
**model** (why an *interpretable* model, and exactly how it is built), the **results** (the tempting
headline, why most of it was a mirage, what really matters, and an honest predictive score), and a
**conclusion** you can carry to any study you ever read.

This one is different from a normal prediction project. Our question is **causal** -- *does estrogen
help or hurt the brain?* -- and the whole lesson is that a naive analysis gets it **wrong**. The
"working solution" is the analysis that gets it **right**.

You do **not** need to memorize the Python. You *do* need to be able to say, in plain English, what
each cell is for and what the punchline is. Anything unclear -- highlight the line and ask Claude
"explain this."
"""),
        # ---------------------------------------------------------------- 1. BACKGROUND
        md("""
## 1. Background: a symptom the test can't see, and a 20-year detective story

Picture the clinical setting. A woman in her sixties sits in the exam room and says, *"I lose words
mid-sentence; I walk into a room and forget why I'm there."* Real, daily **brain fog**. Then the
objective memory test comes back **"normal."** The number on the chart cannot see what she is
describing [7] -- and women's symptoms are, too often, waved off as stress rather than investigated
[8]. Hold onto that gap; it is one of the two threads of this project.

The other thread is a detective story about the drug itself:

- **The exciting clue.** For years, studies that simply *watched* women found that estrogen users
  developed Alzheimer's **less often and later** [1], and long-term users seemed especially protected
  [2]. It looked like proof the pills helped the brain.
- **The plot twist.** Then the Women's Health Initiative Memory Study (WHIMS) let a **coin flip** --
  not the patient -- decide who got hormones (a **randomized controlled trial**, the gold standard).
  The benefit *vanished*; hormones actually **doubled** dementia risk in women 65+ [3] and gave **no**
  boost on cognitive tests [4].

Same drug, **opposite answer**. The rest of this notebook is one question: *how can just-watching and
a coin-flip trial disagree so badly -- and which one do we believe?*
"""),
        # ---------------------------------------------------------------- 2. DATA
        md("""
## 2. The data: NHANES women 60+, and why a survey is the right raw material

We use **NHANES** -- the U.S. National Health and Nutrition Examination Survey, a real government
health study of the whole country. It is the right dataset for three reasons: it is a **real,
national sample** (not something we made up), every woman has both an **objective cognition score**
*and* whether she **used estrogen**, and -- crucially -- it records the **background facts**
(age, education, income) we will need to check for the confounding trap.

The golden rule of the course: **load the data and actually look at it before you model.** Let's do
exactly that.
"""),
        code("""
# Load the course helper that knows how to download + clean the NHANES survey data for us.
import sys                                            # sys lets us tell Python where to find helper files
sys.path.insert(0, "../../notebooks/day3_capstone")  # add the capstone-kit folder to the search path
import capstone_tabular as ct                         # import it under the short name "ct"

df, meta = ct.load_estrogen()                         # df = one row per woman; meta = notes about the data
print("women in the study:", len(df))                 # how many women (all age 60+, NHANES 2013-14)
print("what we know about each:", meta["features"])   # the columns we are allowed to use
df.head()                                             # peek at the first 5 rows
"""),
        md("""
Each row is one woman age 60+. `used_estrogen` is 1 if she ever took female hormones; `age`,
`education`, and `income_ratio` describe her background; `cognition_score` is a **digit-symbol**
memory/attention test (**higher = sharper**). `low_cognition` is 1 for the bottom half of scores --
"did she land in the weaker-scoring half?" is the yes/no outcome our model will study.
"""),
        code("""
# Look at the group sizes and the average score in each group -- the first thing any analyst checks.
print("women who USED estrogen  :", int((df["used_estrogen"] == 1).sum()))   # size of the treated group
print("women who NEVER used it   :", int((df["used_estrogen"] == 0).sum()))  # size of the comparison group
print()
means = df.groupby("used_estrogen")["cognition_score"].mean()   # average memory score in each group
print("avg score, never used :", round(means[0], 1))            # the comparison group's average
print("avg score, used estrogen:", round(means[1], 1))          # the users' average -- looks higher!
"""),
        # ---------------------------------------------------------------- 3. MODEL
        md("""
## 3. The model: why we choose Logistic Regression (the causal/interpretable choice)

Here is the most important decision in the whole project, and it is **not** "which model predicts
best." Because our question is **causal**, we need a model whose **estrogen effect we can read as a
single number -- and watch change** as we add background factors.

- A **black box** (CatBoost, TabPFN, a random forest) can *predict* who scores low, but it gives you
  no clean, adjustable "estrogen effect." You cannot ask it, *"how much did the estrogen number move
  once I controlled for age?"* -- the answer is locked inside. **Prediction is not causation.**
- **Logistic Regression** is one readable equation. Its **coefficient** for `used_estrogen` *is* the
  effect size, and we can literally watch it shrink as we add confounders. For a causal question, that
  readability is worth more than a fancier predictor [5][6].

Let's prove the surprising half of that out loud: the black box does **not** even predict better here,
so there is zero reason to reach for one.
"""),
        code("""
# Compare a black box (Random Forest) vs Logistic Regression on the SAME predict-who-scores-low task.
import numpy as np                                                     # numpy for array math
from sklearn.linear_model import LogisticRegression                    # the interpretable model we will use
from sklearn.ensemble import RandomForestClassifier                    # a stand-in "black box"
from sklearn.preprocessing import StandardScaler                       # puts every feature on the same scale
from sklearn.model_selection import cross_val_predict                  # honest out-of-fold predictions
from sklearn.metrics import roc_auc_score                              # AUC = how well it ranks risk (0.5=coin flip)

feats = meta["features"]                                               # the columns the models may use
y = df["low_cognition"].astype(int).values                            # the outcome: in the weaker-scoring half?
Xs = StandardScaler().fit_transform(df[feats].astype(float).values)   # standardized features (for logistic)
X = df[feats].astype(float).values                                     # raw features (trees don't need scaling)

# cross_val_predict trains on 4/5 of the data and predicts the held-out 1/5, five times, so no cheating.
logit_auc = roc_auc_score(y, cross_val_predict(
    LogisticRegression(max_iter=2000), Xs, y, cv=5, method="predict_proba")[:, 1])
rf_auc = roc_auc_score(y, cross_val_predict(
    RandomForestClassifier(n_estimators=300, random_state=0), X, y, cv=5, method="predict_proba")[:, 1])
print(f"Logistic Regression  AUC = {logit_auc:.3f}   <- and it hands us a readable estrogen effect")
print(f"Random Forest (black box) AUC = {rf_auc:.3f}   <- no adjustable effect size, and no better here")
"""),
        md("""
The black box does **not** beat the readable model here (about **0.73** vs **0.78**). And even on a
bigger dataset where a black box *did* predict better, it still could not answer our question, because
it cannot give us an **adjustable effect size**. So the choice is easy: **Logistic Regression.**

### Model & data processing (the exact recipe)

So the whole thing is reproducible, here is the full pipeline in one place:

- **Model:** `LogisticRegression` (scikit-learn), one linear equation whose `used_estrogen`
  coefficient is the effect size we read.
- **Features:** `age`, `education`, `income_ratio`, `used_estrogen`, `years_estrogen`, each
  **standardized** (mean 0, scale 1) so the coefficients are comparable.
- **The causal move:** fit the estrogen effect **twice** -- *crude* (estrogen alone) and *adjusted*
  (estrogen + age + education + income). The change between them is the confounding.
- **Scoring:** predictive **AUC** via **5-fold cross-validation** (above); **fairness** via a
  held-out train/test split, accuracy reported per group.
"""),
        # ---------------------------------------------------------------- 4. RESULTS
        md("""
## 4. Results

### Result 1 -- the tempting headline: "estrogen users score better"

The naive look anyone would take first: just compare the average test score of the two groups.
"""),
        code("""
# Re-state the raw gap as the tempting headline (we already computed the group means above).
gap = means[1] - means[0]                              # users' average minus non-users' average
print("used estrogen   -> average score:", round(means[1], 1))   # the higher-scoring group
print("never used      -> average score:", round(means[0], 1))   # the lower-scoring group
print("raw gap (users minus non-users) :", round(gap, 1), "points  <- the exciting headline")
"""),
        md("""
Users score about **51.8** versus **43.7** -- an **8-point** head start. Taken at face value, that is
the headline the 1990s studies chased: *"estrogen users think better"* [1][2]. But we never checked
whether the two groups were the **same kind of people** to begin with.

![The tempting headline: estrogen users score higher](figures/raw_means.png)
"""),
        md("""
### Result 2 (the whole lesson) -- most of the gap was confounding

Now the honest test. We measure estrogen's effect on landing in the weak-scoring half **two ways** --
*crude* (alone) and *adjusted* (after age, education, income). If the adjusted effect **shrinks toward
zero**, the crude number was mostly the confounders -- the **healthy-user bias** -- not the pill.
(The number is a logistic coefficient: negative = "less likely to score low," i.e. looks protective.)
"""),
        code("""
# ct.association reports the effect of one feature TWICE: alone (crude), then controlling for others.
result = ct.association(                              # the helper prints both numbers and the % change
    df,                                              # our table of women
    feature="used_estrogen",                         # the thing whose effect we are testing
    target="low_cognition",                          # the outcome: did she land in the weaker-scoring half?
    controls=["age", "education", "income_ratio"],   # the confounders we suspect explain the gap
)
crude = result["crude"]                               # estrogen's effect all by itself
adjusted = result["adjusted"]                         # estrogen's effect after adjusting for background
shrink = (1 - adjusted / crude) * 100                 # how much of the crude effect disappeared, as a %
print(f"\\nThe effect shrank by {shrink:.0f}% once we adjusted for age, education, and income.")
"""),
        md("""
This is the punchline of the whole project. The crude effect (about **-0.33**) shrinks to about
**-0.11** once we adjust -- roughly **two-thirds (67%) of the apparent benefit evaporates.** That
missing two-thirds was never the estrogen; it was that users were **healthier and better-off to start
with** [5][6]. This is confounding caught in the act, and it is a miniature of exactly why the big
observational studies and the WHIMS coin-flip trial disagreed [6].

![67% of the apparent effect was confounding, not cause](figures/confounding.png)
"""),
        md("""
### What actually matters -- feature importance from the *same* model

Because we chose an interpretable model, "feature importance" is free and honest: it is just the
**standardized coefficients** of the very model we used for the causal claim. If estrogen were the
real driver, its bar would be tall. Watch what happens instead.
"""),
        code("""
# Fit the logistic model once on all the (standardized) data and read off each feature's weight.
lr = LogisticRegression(max_iter=2000).fit(Xs, y)      # one readable equation on standardized features
importance = sorted(                                   # pair each feature with the SIZE of its coefficient
    zip(feats, np.abs(lr.coef_[0])),                   # |coefficient| = how much that feature moves the risk
    key=lambda t: -t[1],                               # sort biggest-effect first
)
for name, weight in importance:                        # walk the features from most to least important
    star = "   <- the drug everyone asked about" if name == "used_estrogen" else ""
    print(f"  {name:<15s} |coef| = {weight:.3f}{star}")   # print each feature's standardized weight
"""),
        md("""
**Education and age dominate; `used_estrogen` is the *smallest* bar.** The same model that answers the
causal question also shows, visually, that estrogen barely moves cognition once you know a woman's
background. Prediction and causation agree here -- and neither points at the pill.

![What actually predicts cognition -- not estrogen](figures/feature_importance.png)
"""),
        md("""
### Can it even predict? An honest secondary number

The causal answer is the deliverable, but it is fair to ask how well the model *predicts* low
cognition at all. We already computed it above with 5-fold cross-validation.
"""),
        code("""
# Report the honest predictive score (computed earlier), so the deck and notebook match exactly.
print(f"predictive AUC (Logistic Regression, 5-fold CV) = {logit_auc:.3f}")   # ~0.78, a real signal
print("0.5 would be a coin flip; 0.78 means it ranks who scores low clearly better than chance,")
print("driven mostly by age and education -- NOT by estrogen.")
"""),
        md("""
### Fairness -- does the model work equally well for both groups?

Whenever you build a predictor for people, ask: does it work as well for *everyone*? We train one
model, then check its accuracy **separately** for women who did and did not use estrogen.
"""),
        code("""
# Train once on everyone, then report TEST accuracy separately for each group (a fairness audit).
group_acc = ct.audit_by_group(                       # the helper trains a model + splits accuracy by group
    df,                                              # our table
    features=meta["features"],                       # all the columns the model may use
    target="low_cognition",                          # what it predicts
    group="used_estrogen",                           # the two groups to compare it across
    group_names={0: "never used", 1: "used estrogen"},   # readable labels for the chart
)
print("accuracy per group:", {k: round(v, 2) for k, v in group_acc.items()})  # the two numbers
"""),
        md("""
The model is a bit **more accurate for the estrogen-users (~0.73) than for non-users (~0.65).** A gap
like that is worth flagging: a tool that is reliably right for one group and shakier for another can
quietly do harm if a clinic trusts it equally for both. Naming the gap out loud is the honest move.

![The model's accuracy is uneven across the two groups](figures/fairness_by_group.png)
"""),
        # ---------------------------------------------------------------- 5. CONCLUSION
        md("""
## 5. Conclusion: correlation is not causation, and the right method flips the answer

- **The naive answer was wrong.** "Estrogen users score better" is a real correlation -- and mostly a
  **mirage**. Adjusting for who these women were shrank the effect by **67%**. The right method
  (an interpretable model, read crude-vs-adjusted) is the whole deliverable.
- **Prediction is not causation.** A black box could rank patients but could never hand us the one
  number the question is about. That is why a **causal** question *requires* an interpretable model.
- **A survey can hint; only an experiment proves.** Even after adjusting, we only corrected for
  confounders we *measured*. Something unrecorded could still be doing the work. That is exactly why
  WHIMS [3][4] -- a coin-flip trial -- could overturn twenty years of observational headlines.
- **And take the patient's account seriously.** The tests can say "normal" while a woman lives with
  real brain fog [7], and women's symptoms are dismissed too readily [8]. The grown-up stance holds
  both: the confounding is real *and* her lived report is real evidence, not noise.

**Bottom line:** when a group that *chose* a treatment looks healthier, suspect the *chooser*, not the
treatment -- and remember that only an experiment can promote a correlation to a cause.

## References
[1] Tang et al. 1996, *Lancet* (observational) &middot;
[2] Zandi et al. 2002, *JAMA* -- Cache County (observational) &middot;
[3] Shumaker et al. 2003, *JAMA* -- WHIMS dementia (RCT) &middot;
[4] Rapp et al. 2003, *JAMA* -- WHIMS cognition (RCT) &middot;
[5] Wharton et al. 2009, *Maturitas* -- healthy-user bias &middot;
[6] Vandenbroucke 2009, *Lancet* -- why observational & RCT disagreed &middot;
[7] Maki 2024, Harvard Health -- menopause & brain fog &middot;
[8] Chen & Shafir 2025, Harvard Health -- dismissal of women's symptoms.
"""),
    ]
    save(nb, HERE / "solution.ipynb")
    print("wrote solution.ipynb")


if __name__ == "__main__":
    build()
