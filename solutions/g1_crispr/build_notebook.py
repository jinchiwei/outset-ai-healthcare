"""Generate the G1 CRISPR guide-picker SOLUTION notebook (the worked answer key).

HS-accessible, and EVERY line of code is commented. It runs top to bottom the way the five-part
rubric asks for it: the BACKGROUND (why picking a good CRISPR guide matters, clinically), the DATA
(why Doench 2016 is the benchmark), the MODEL (turn letters into numbers, then a bake-off that picks
CatBoost), the RESULTS (how we measure -- AUC 0.88 -- the concrete 97%-precision win, and the
position-importance that rediscovers the seed biology), and an honest CONCLUSION.

It does a small LIVE train to prove the pipeline runs, then loads the canonical results.json + saved
figures so every number matches the slides exactly. Edit this script + rebuild; never hand-edit the
.ipynb.
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
# G1 SOLUTION -- Picking good CRISPR guides before the lab

**This is the worked answer key.** It runs top to bottom and shows the whole project the way the
five-part rubric asks for it: the **background** (why this matters clinically), the **data** (why
Doench 2016), the **model** (how we turn a DNA sequence into numbers, and which model we picked and
why), the **results** (how we measured, the working number, the concrete win, and feature
importance), and an honest **conclusion**.

Our job is a **guide-picker**: given the 20-letter guide, predict whether CRISPR-Cas9 will *cut
well* at that site -- so a scientist can rank candidate guides and order only the best ones, instead
of testing dozens in the lab over weeks. The win is concrete: **order the top 10% the model ranks,
and 97% of them actually cut well.**

You do **not** need to memorize the Python. You *do* need to be able to say, in plain English, what
each cell is for. Anything unclear -- highlight the line and ask Claude "explain this."
"""),
        # ---------------------------------------------------------------- 1. BACKGROUND
        md("""
## 1. Background: what are we building, and why?

**CRISPR-Cas9** is a pair of molecular scissors. You aim it at one spot in DNA using a 20-letter
**guide** (letters A/C/G/T). Where the guide matches, the Cas9 protein cuts -- and that cut is how
scientists turn a gene off or fix a disease-causing mutation. This is not hypothetical: the same tool
is now an **approved therapy for sickle-cell disease** (2023) [3].

**The catch:** some guides cut great, some barely cut, even when they look similar. Testing each one
in the lab costs weeks. So scientists **predict** cutting efficiency from the 20 letters first -- the
best-known tool is Doench 2016's "Rule Set 2" / Azimuth [1], and that is exactly the dataset we use.

**The key biology:** Cas9 grabs a landmark called the **PAM**, then checks the letters right next to
it -- the **seed**. Guides that mismatch there usually fail to cut [3][4]. So letters near the PAM
should matter most. Keep that in mind -- our model will *rediscover* it, having only ever seen
sequences and cut-scores.
"""),
        # ---------------------------------------------------------------- 2. DATA
        md("""
## 2. The data: Doench 2016, and look at it first

We use the **Doench et al. 2016** guide-efficiency dataset [1]: about **5,300 guide RNAs**, each one
actually tested in the lab so we know how well it cut. It is the right dataset for three reasons: the
cut-scores are *real measurements* (not simulated), it is the *canonical benchmark* the whole field
compares against [1][7], and it is *big enough* to train and honestly test a model.

The golden rule of the whole course: **load the data and actually look at it before you model.**
"""),
        code("""
# Load the course helper that knows how to read the CRISPR data + turn sequences into numbers.
import sys                                              # sys lets us tell Python where to find helper files
sys.path.insert(0, "../../notebooks/day3_capstone")    # add the capstone helpers folder to the search path
import capstone_seq as cs                               # import them under the short name "cs"

df, meta = cs.load_guides()                             # df = a table of guides; meta = notes about the dataset
print("guides:", len(df), " genes:", df["gene"].nunique())   # how many guides, across how many genes
df[["guide20", "gene", "gc_content", "efficiency"]].head()   # peek at the first 5 rows
"""),
        md("""
Each row is one guide. `guide20` is the 20-letter sequence we predict from, and `efficiency` is how
well it actually cut (0 = never cut, 1 = cut every time). Everything hangs on that one measured number.
"""),
        md("""
### Framing the question: clearly-good vs clearly-poor

Efficiency is a smooth 0-to-1 number, and the guides in the *muddy middle* are genuinely ambiguous --
even the lab measurements are noisy there. So we ask the honest, clean question a scientist actually
cares about: can the model tell a **clearly-good** guide (top third of cutters) from a **clearly-poor**
one (bottom third)? We drop the middle third and label the rest 1 (good) or 0 (poor).
"""),
        code("""
# Build the clean two-class question: clearly-good (top third) vs clearly-poor (bottom third) cutters.
hi = df["efficiency"].quantile(2/3)                     # the cut-off for the TOP third (clearly-good guides)
lo = df["efficiency"].quantile(1/3)                     # the cut-off for the BOTTOM third (clearly-poor guides)
d = df[(df.efficiency >= hi) | (df.efficiency <= lo)].copy()   # keep only the clear cases; drop the muddy middle
d["y"] = (d.efficiency >= hi).astype(int)              # label: 1 = clearly-good, 0 = clearly-poor
print("guides used (clear cases):", len(d))            # how many guides are left after dropping the middle
print("clearly-good (1):", int(d.y.sum()), "   clearly-poor (0):", int((d.y == 0).sum()))  # a ~50/50 balance
print(f"base rate = {d.y.mean():.0%}  ->  the classes are balanced, so we cannot cheat by always guessing one.")
"""),
        md("""
The two classes are **balanced (~50/50)**, so a lazy model that always guesses one label scores only
50%. That makes the score honest: any lift above 50% is real signal the model found in the sequence.
"""),
        # ---------------------------------------------------------------- 3. MODEL
        md("""
## 3. The model: first turn letters into numbers

A computer cannot do math on the letter "A", so the first job is always to turn each guide into
numbers. There are two ways, and the whole lesson lives in the gap between them:

- **one-hot** -- 4 switches per position (A/C/G/T), flip the right one on. It keeps *where* each letter
  sits -- so it can learn that the seed near the PAM matters more than the far end.
- **k-mer** -- just count how many of each letter and pair. It throws the *order* away.

"LISTEN" and "SILENT" have the same letters -- k-mer cannot tell them apart; one-hot can. Since the
biology cares *where* a letter is (the seed!), we use **one-hot**.
"""),
        code("""
# Turn every clear-case guide into numbers two different ways, and look at how big each table is.
X_onehot, names = cs.featurize(d, mode="onehot", seq_col="guide20")   # order-AWARE: 20 positions x 4 = 80 numbers
X_kmer,   _     = cs.featurize(d, mode="kmer",   seq_col="guide20")   # order-BLIND: letter + pair counts
y = d["y"].values                                       # the yes/no answer we are trying to predict (1 = good guide)
print("one-hot shape:", X_onehot.shape, " <- 80 numbers per guide, keeps position")   # (rows, 80)
print("k-mer shape:  ", X_kmer.shape,   " <- just counts, order thrown away")          # (rows, 21)
"""),
        md("""
### Which model? We tested, we did not guess

Four candidate models could learn this: **Logistic Regression** (a straight-line rule), **Random
Forest** and **CatBoost** (two "tree" models that learn combinations of letters), and **TabPFN** (a
newer neural network for small tables). We do not guess -- we **bake them off** on held-out guides and
let AUC pick the winner. First, one quick live train to prove the pipeline runs end to end.
"""),
        code("""
# Train ONE model (CatBoost) live on the clean cases -- a quick check that the whole pipeline works.
from sklearn.model_selection import train_test_split    # splits data into a study set + a held-out test set
from sklearn.metrics import roc_auc_score               # the AUC score we grade on
from catboost import CatBoostClassifier                 # CatBoost = a strong gradient-boosted tree model

Xa, Xb, ya, yb = train_test_split(X_onehot, y,          # hold back 25% of guides as an unseen test set...
                                  test_size=0.25, random_state=0, stratify=y)   # ...same split each run; keep balance
model = CatBoostClassifier(verbose=0, random_state=0,   # a fresh CatBoost (verbose=0 keeps it quiet)...
                           iterations=600, depth=6, learning_rate=0.05)         # ...with sensible settings
model.fit(Xa, ya)                                       # train it on the study set
prob = model.predict_proba(Xb)[:, 1]                    # its predicted probability of "good guide" on the test set
print(f"live CatBoost AUC on held-out guides = {roc_auc_score(yb, prob):.3f}")  # ~0.88 -- a working picker
"""),
        md("""
The live model lands around **AUC 0.88** in seconds -- the pipeline works. Now load the canonical
bake-off the full experiment saved, so every number here matches the slides exactly.
"""),
        code("""
# Load the saved results so the notebook, the figures, and the slides all report the same numbers.
import json                                             # json reads the saved results file
R = json.load(open("results.json"))                     # load the numbers the full experiment saved
print("Model bake-off (higher AUC wins):")              # WHY we picked the model we picked
for name, auc in R["model_auc"].items():                # walk each of the four candidate models...
    print(f"  {name:<14s} AUC = {auc:.3f}")             # ...and print its AUC on the held-out guides
print("\\nThe two TREE models tie at 0.88 and win; both beat TabPFN (0.83) and crush Logistic Reg (0.75).")
print("We pick CatBoost -- a strong, fast tree model -- as our guide-picker.")
"""),
        md("""
**The two tree models (CatBoost and Random Forest) tie at AUC 0.88**, both beat TabPFN (0.83), and
both crush Logistic Regression (0.75). We pick **CatBoost**. This is the "which tool?" decision made
honestly -- by measuring on held-out data, not by guessing.

![We tested four models -- the tree models win](figures/model_choice.png)
"""),
        md("""
### Model & data processing (the exact recipe)

So the whole thing is reproducible, here is the full pipeline in one place:

- **Task:** tell a **clearly-good** guide (top-third efficiency) from a **clearly-poor** one
  (bottom-third); the muddy middle third is dropped.
- **Encode:** each 20-letter guide becomes an **80-number one-hot** vector (4 switches x 20 positions),
  which keeps *where* every letter sits.
- **Split:** a stratified **75% train / 25% test** split; every number we report is on the held-out
  test guides the model never saw.
- **Model:** `CatBoostClassifier` (600 trees, depth 6, learning rate 0.05), `random_state=0` so it is
  reproducible.
- **Grade:** **AUC** on the test set -- not accuracy -- so the score cannot be gamed.
"""),
        # ---------------------------------------------------------------- 4. RESULTS
        md("""
## 4. Results: how we measured, and what we found

**How we measure a picker: ROC and AUC.** The model outputs a *score*, not a hard yes/no. The **ROC
curve** sweeps every possible cut-off and plots good guides correctly flagged against false alarms.
**AUC** is the area under that curve: 0.5 is a coin flip, 1.0 is perfect.
"""),
        code("""
# Print the headline numbers the full experiment saved.
print(f"guides used (clear cases) : {R['n_used']}")                    # how many clear-case guides we graded on
print(f"class balance (base rate) : {R['base_rate']:.0%}")             # the ~50/50 balance -> 0.5 is the coin-flip line
print(f"AUC (CatBoost)            : {R['auc']:.3f}   <- a working guide-picker, well above 0.8")  # the headline
"""),
        md("""
**AUC 0.88** -- a genuinely working picker. The ROC curve below hugs the top-left corner: at a good
setting it flags most of the good guides while raising few false alarms.

![A working guide-picker, AUC 0.88](figures/roc.png)
"""),
        md("""
### The concrete win: order the top picks, get good guides

AUC is abstract; here is the number a scientist feels. Rank every guide by the model's score, then
**order only the top 10%** -- how many of those actually cut well?
"""),
        code("""
# Show the payoff: how many of the model's TOP-ranked picks really are good guides.
print(f"order the top 10% the model ranks -> {R['precision_top10']:.0%} of them actually cut well")  # 97%!
print(f"pick guides at random             -> {R['base_rate']:.0%} would cut well")                    # 50%
print("\\nThat is the whole value: far less wasted lab time, because the shortlist is almost all winners.")
"""),
        md("""
**97% of the top 10% actually cut well**, versus 50% if you picked at random. That is the concrete,
usable win: the shortlist the model hands you is almost all winners, so far less lab time is wasted.

![Order the model's top picks, get mostly good guides](figures/precision_at_top.png)
"""),
        md("""
### Feature importance: the model rediscovered the biology

For a table of numbers, feature importance asks *which inputs mattered?* Here each input is a
(position, letter), so we can ask *which positions along the guide did the model rely on?* If it leans
on the letters near the PAM (the seed), it found real CRISPR biology on its own -- nobody told it about
seeds.
"""),
        code("""
# Print WHERE along the guide the signal lives, from the saved importance analysis.
print(f"most important position = {R['top_position']} of 20   (position 20 sits right next to the PAM)")  # the seed!
print("The importance rises toward the PAM end -- exactly the 'seed' region decades of CRISPR work describe.")
"""),
        md("""
The most important position is **20 of 20** -- right next to the PAM -- and the importance rises
toward that end: the **seed region**. The model agrees with decades of CRISPR science, having only ever
seen sequences and cut-scores. **A model you can explain, that matches known biology, is one people
will trust.**

![It rediscovered the biology: importance peaks at the seed](figures/position_importance.png)
"""),
        md("""
### Fairness across groups: not applicable here (and that is honest)

Most medical-AI projects must check they work equally well across patient groups (sex, age, skin
tone). **This task has none:** the input is a 20-letter DNA sequence, with no patient, no demographics,
nothing to be unfair *about*. So there is no fairness gap to audit -- and saying that plainly, rather
than forcing a fake analysis, is the honest move. (The real fairness question for CRISPR lives one step
later, in *which* patients a therapy is tested on and reaches -- outside this sequence model.)
"""),
        # ---------------------------------------------------------------- 5. CONCLUSION
        md("""
## 5. Conclusion: honest limits

- **It works** -- a real guide-picker on the field's benchmark data: **AUC 0.88**, and its top-10%
  shortlist is **97%** good guides, so it genuinely saves lab time.
- **It learned real biology** -- importance peaks at the **seed** next to the PAM, matching decades of
  CRISPR science [3][4], which is why it is trustworthy.
- **But** it cannot match a production tool like Azimuth [1], or capture cell-type / chromatin effects
  in a real cell [5][7], and it says nothing about **off-target** safety -- does the guide also cut the
  wrong places? [8].

**Bottom line:** use this to build intuition and shortlist candidates; use a validated tool and the lab
to actually choose and confirm a guide.

## References
[1] Doench et al. 2016, *Nat Biotechnol* -- Rule Set 2 / Azimuth, and the dataset we use.
[2] Doench et al. 2014, *Nat Biotechnol* -- Rule Set 1: position + PAM predict cutting.
[3] Hsu, Lander & Zhang 2014, *Cell* -- how Cas9 is steered to a DNA site (the seed).
[4] Zheng et al. 2017, *Sci Rep* -- direct evidence the PAM-proximal seed matters most.
[5] Chuai et al. 2018, *Genome Biol* -- DeepCRISPR: deep learning for guide design.
[6] Kim et al. 2019, *Sci Adv* -- DeepSpCas9: more data + neural nets push accuracy further.
[7] Konstantakos et al. 2022, *NAR* -- survey of guide-efficiency predictors.
[8] Abbaszadeh & Shahlai 2025, *arXiv* -- explainable AI for guide design + off-target safety.
"""),
    ]
    save(nb, HERE / "solution.ipynb")
    print("wrote solution.ipynb")


if __name__ == "__main__":
    build()
