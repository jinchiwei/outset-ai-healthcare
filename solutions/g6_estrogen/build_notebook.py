"""Generate the G6 estrogen-&-cognition SOLUTION notebook (the worked answer key).

Comprehensive, HS-accessible, and EVERY line of code is commented. Runs top to bottom and
reproduces the real results: estrogen users LOOK better on the memory test, but two-thirds of
that gap is confounding (healthy-user bias) -- it shrinks once we adjust for age, education,
and income. Core lesson: confounding vs. causation, plus the objective-test-vs-brain-fog gap.

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

**This is the worked answer key.** It runs top to bottom and shows the whole project: the history,
the one big idea (confounding), the code (every line commented), the result, and an honest look at
what a survey can and cannot prove.

You do **not** need to memorize the Python. You *do* need to be able to say, in plain English, what
each cell is for and what the punchline is. Anything unclear -- highlight the line and ask Claude
"explain this."
"""),
        md("""
## 1. Background: a 20-year detective story

For decades doctors wondered whether **hormone therapy** (estrogen, taken by many women around
menopause) also protects the *brain* -- keeping memory sharp and holding off dementia.

- **The first clue looked exciting.** Studies that simply *watched* women found that those who had
  taken estrogen got Alzheimer's **less often and later** [1], and long-term users seemed especially
  protected [2]. It looked like proof the pills helped the brain.
- **Then someone ran a real experiment.** In the Women's Health Initiative Memory Study (WHIMS), a
  **coin flip** -- not the patient -- decided who got hormones. That is a **randomized controlled
  trial (RCT)**, the gold standard. The benefit *vanished*; worse, hormones actually **doubled** the
  risk of dementia in women 65+ [3], and gave **no** boost on cognitive tests [4].

Same drug, **opposite answer.** How can just-watching and a coin-flip trial disagree so badly?
"""),
        md("""
## 2. The one idea: confounding (a.k.a. "healthy-user bias")

Here is the trap. The women who *chose* hormone therapy were also, on average, **richer, more
active, more likely to exercise and see doctors, with fewer heart problems** to begin with [5][6].
So of course they had better memory later -- but it was the *type of person*, not the pill.

A hidden third factor ("already healthier / wealthier / more active") secretly drives **both**
taking hormones **and** better memory. That fake link is called **confounding**.

> **The ice-cream & shark-attacks analogy.** On days more ice cream is sold, more people are bitten
> by sharks. Does ice cream summon sharks? No. A hidden third factor -- **hot summer weather** --
> drives *both*: heat sells ice cream *and* sends people into the ocean. Ice cream and shark bites
> move together, but neither causes the other. "Estrogen use" and "good memory" are the ice cream
> and the sharks; "being healthier to start with" is the summer.

The fix scientists use with survey data: **adjust** for the hidden factors (age, education, income).
If an apparent effect **shrinks** once you account for them, a lot of it was confounding, not cause.
Watch our toy data do exactly that.
"""),
        code("""
# Load the course helper that knows how to download + clean the NHANES survey data for us.
import sys                                            # sys lets us tell Python where to find helper files
sys.path.insert(0, "../../notebooks/day3_capstone")  # add the capstone-kit folder to the search path
import capstone_tabular as ct                         # import it under the short name "ct"

df, meta = ct.load_estrogen()                         # df = one row per woman; meta = notes about the data
print("women in the study:", len(df))                 # how many women (all age 60+, NHANES 2013-14)
print("what we know about each:", meta["features"])   # the columns we can use
df.head()                                             # peek at the first 5 rows
"""),
        md("""
Each row is one woman age 60+. `used_estrogen` is 1 if she ever took female hormones, `age`,
`education`, and `income_ratio` describe her background, and `cognition_score` is her score on a
memory/attention test (**higher = sharper**). `low_cognition` is 1 for the bottom half of scores --
that "did she score in the weaker half?" is what our model will try to predict.
"""),
        md("""
## 3. Result #1 -- the tempting headline: "estrogen users score better"

First, the naive look anyone would take: just compare the average test score of women who used
estrogen against those who never did.
"""),
        code("""
# Group the women by whether they used estrogen, then average each group's memory-test score.
means = df.groupby("used_estrogen")["cognition_score"].mean()   # average score in each of the two groups
print("never used estrogen -> average score:", round(means[0], 1))   # the comparison group
print("used estrogen       -> average score:", round(means[1], 1))   # the hormone-users
print("raw gap (users minus non-users):", round(means[1] - means[0], 1), "points")  # the tempting headline
"""),
        md("""
Users score about **51.8** versus **43.7** for non-users -- an **8-point** head start. Taken at face
value, that is the exciting headline: *"estrogen users think better."* That is exactly the clue the
early observational studies saw [1][2]. But we have not checked whether the two groups were the
**same kind of people** to begin with. Let's find out.
"""),
        md("""
## 4. Result #2 (the whole lesson) -- most of the gap was confounding

Now the honest test. We measure the effect of estrogen on scoring in the weak half **two ways**:

1. **crude** -- estrogen *by itself*, ignoring everything else.
2. **adjusted** -- estrogen *after* accounting for age, education, and income.

If the adjusted effect **shrinks toward zero**, the crude number was mostly the confounders -- the
healthy-user bias -- not the pill. (The number is a logistic-regression coefficient: negative means
"less likely to be in the low-scoring group," i.e. looks protective.)
"""),
        code("""
# ct.association reports the effect of one feature TWICE: alone (crude), then controlling for others.
result = ct.association(                              # the helper prints both numbers and the % change
    df,                                              # our table of women
    feature="used_estrogen",                         # the thing whose effect we're testing
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
missing two-thirds was never the estrogen; it was that users were **healthier and better-off to
start with** [5][6]. This is confounding caught in the act, and it is a miniature of exactly why the
big observational studies and the WHIMS coin-flip trial disagreed [6].
"""),
        md("""
## 5. Fairness -- does the model work equally well for both groups?

Whenever you build a predictor for people, ask: does it work as well for *everyone*? We train one
model to predict `low_cognition`, then check its accuracy **separately** for women who did and did
not use estrogen. Equal accuracy is not guaranteed just because the overall number looks fine.
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
The model is a bit **more accurate for the estrogen-users (~0.73) than for non-users (~0.65).** A
gap like that is worth flagging: a tool that is reliably right for one group and shakier for another
can quietly do harm if a clinic trusts it equally for both. Naming the gap out loud is the honest
move -- the same habit you practiced all week.
"""),
        md("""
## 6. What the numbers can't capture -- the patient's account

Our data is a memory *test score*. But many women in the menopause transition report **"brain
fog"** -- a real, daily feeling of being slower or forgetful -- **even when their objective test
scores look normal** [7]. The tests can say "fine" while the person says "something's wrong."

That gap matters, because women's symptoms are more often waved off as stress or emotion than
actually investigated [8]. A good data scientist holds two ideas at once: *the confounding is real
(the pill may not be the hero the headline claimed)* **and** *the patient's reported experience is
real evidence too, not noise to be dismissed.*
"""),
        md("""
## 7. Honest limits -- a survey can hint, only an experiment can prove

- **What this analysis CAN do:** show that a tempting headline ("users score higher") can be **mostly
  a mirage** created by who chooses the drug, and that *adjusting* for background shrinks it by ~67%.
- **What it CANNOT do:** **prove cause.** Even after adjusting, we can only correct for confounders we
  *measured and thought of*. Something we didn't record could still be doing the work. Survey data
  can **hint** at confounding; it can never nail down cause.
- **What actually settles it:** a **randomized controlled trial**, where a coin flip -- not the
  patient -- assigns the drug, so the two groups start out the same. That is exactly why WHIMS [3][4]
  could overturn twenty years of observational headlines.

**Bottom line:** when a group that *chose* a treatment looks healthier, suspect the *chooser*, not
the treatment -- and remember that only an experiment can promote a correlation to a cause.

## References
[1] Tang et al. 1996, *Lancet* (observational) ·
[2] Zandi et al. 2002, *JAMA* -- Cache County (observational) ·
[3] Shumaker et al. 2003, *JAMA* -- WHIMS dementia (RCT) ·
[4] Rapp et al. 2003, *JAMA* -- WHIMS cognition (RCT) ·
[5] Wharton et al. 2009, *Maturitas* -- healthy-user bias ·
[6] Vandenbroucke 2009, *Lancet* -- why observational & RCT disagreed ·
[7] Maki 2024, Harvard Health -- menopause & brain fog ·
[8] Chen & Shafir 2025, Harvard Health -- dismissal of women's symptoms.
"""),
    ]
    save(nb, HERE / "solution.ipynb")
    print("wrote solution.ipynb")


if __name__ == "__main__":
    build()
