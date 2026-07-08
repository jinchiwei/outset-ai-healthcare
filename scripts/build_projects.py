"""Build the six TAILORED capstone starter notebooks (one per project group).

AUDIENCE: high-school students, many sophomores, who barely know Python. So every code cell
gets a plain-English "what this does" before it, and every result gets a "why does this happen"
after it, in concise layman's terms with analogies. Best performance is nice; UNDERSTANDING is
the goal. See scripts/build_capstone.py for the generic-kit version.

Three templates:
  IMAGE     (capstone_common.py)  -- G2 skin screening, G3 brain CT, G4 skin access
  TABULAR   (capstone_tabular.py) -- G5 heart/cholesterol, G6 estrogen & cognition
  SEQUENCE  (capstone_seq.py)     -- G1 CRISPR guide efficiency

Run:  python scripts/build_projects.py
Notebooks land in notebooks/day3_capstone/projects/<folder>/starter.ipynb.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from nbutil import code, md, new_nb, save  # noqa: E402

BASE = ROOT / "notebooks/day3_capstone/projects"
REPO = "https://github.com/jinchiwei/outset-ai-healthcare.git"


def _how_to_run():
    return md("""
### How to use this notebook (read this first)

- A notebook is a stack of **cells**. A gray cell is **code**; this white cell is a note.
- To run a code cell, click it and press **Shift+Enter**. Run them **top to bottom, in order** --
  later cells use things earlier cells made.
- **Red text is normal**, not a disaster -- it's usually a warning. A real error stops with a
  message; read the last line, or paste it to Claude and ask "what does this mean?"
- You will not understand every line, and that's fine. But you should be able to say, in plain
  English, *what each cell is for*. Stuck on a line? Ask Claude: **"explain this cell line by line."**
- Nothing you click here can break anything. Experiment.
""")


def _your_job():
    return md("""
### You have Claude Code -- so here's your *real* job

You can already make a computer write code: just describe what you want to Claude Code and it will
build it. So **writing code is no longer the hard part.** The skill that actually makes you good at
this -- the thing this notebook is training -- is **judgment**:

1. **What are you trying to build?** "Accurate" isn't a goal. A cancer screen that must never miss a
   tumor is a *different tool* than one that's just right-on-average. Decide what "good" means first.
2. **What tools exist?** Different models, different ways to prep the data, different ways to grade a
   model. This notebook is a guided tour of that toolbox.
3. **Why does each tool exist** -- what problem it was invented to solve.
4. **How do you pick the right one?** That's the whole game. The wrong tool gives a wrong answer even
   with flawless code.

So: read the notes, play with the dials, and when you want to change or add something, **describe it
to Claude Code and let it write the code** -- then be ready to explain *why you chose it*. That "why"
is what you're graded on, and it's what a real scientist or doctor would ask you.
""")


def _setup_cell(folder, ensure_line):
    return code(f"""
# WHAT THIS DOES: gets the course files onto this Colab machine and installs the tools we need.
# (On your own computer it does nothing.) Just run it once and wait for "setup ok".
import os, sys
if not os.path.exists("../../capstone_common.py"):
    os.system("git clone -q {REPO}")
    os.chdir("outset-ai-healthcare/notebooks/day3_capstone/projects/{folder}")
sys.path.insert(0, "../..")            # so we can import the course helper files
sys.path.insert(0, "../../../_shared")
import colab_setup
{ensure_line}
""")


def _rubric_line():
    return ("**Your goal (the rubric):** build it, check honestly how well it works, find a case where "
            "it fails, defend one choice you made, and make sure *both* partners can explain everything.")


def _stroke_tabular_cells():
    """A 'Part 2' section (for G3): the fairness audit the CT can't give -- a tabular stroke set
    that DOES record sex + age. Runnable baseline + a by-sex audit, same style as the image part."""
    return [
        md("""
---
## Part 2 -- the fairness the brain CT can't give you

Your CT model works, but it records **nothing about the patient** -- no age, no sex, no race. So you
**cannot check** whether it misses strokes more often in one group than another. That blank is the
whole problem.

The fix: bring in a **different** stroke dataset -- rows of numbers per patient (age, sex, blood
pressure, glucose, smoking) that **does** record who each person is. Now you can build a stroke-risk
model *and* audit it. This is the same idea as Day 2's tabular work.
"""),
        code("""
# Install the tabular tools + load the stroke dataset (it ships with the course repo).
colab_setup.ensure("catboost", "tabpfn==2.2.1", "scikit-learn", "pandas")
import capstone_tabular as ct                      # the tabular helpers (like Day 2)
df, meta = ct.load_stroke()                        # 5,110 patients, each a row of numbers
print(meta["name"], "->", df.shape[0], "patients")
print("features:", meta["features"])
print("we can audit by:", meta["group"], meta["group_names"])
df.head()
"""),
        md("""
**A trap to notice first:** only about **5%** of these patients had a stroke. So a lazy model that
says "no stroke" for *everyone* is right 95% of the time and completely useless. Accuracy is the wrong
score here -- watch **AUC** (0.5 = coin flip, 1.0 = perfect) and, most of all, **how many real strokes
you actually catch** in each group.
"""),
        code("""
# Build a stroke-risk model: pick features + a model, hit Run, read the score.
from ipywidgets import interact_manual, SelectMultiple, Dropdown

def build(features, model):
    if not features:
        print("Pick at least one feature."); return
    ct.fit_eval(df, list(features), meta["target"], model=model)

try:
    interact_manual(build,
        features=SelectMultiple(options=meta["features"], value=tuple(meta["features"]),
                                description="features", rows=10, style={"description_width": "initial"}),
        model=Dropdown(options=list(ct.make_models()), value="Random Forest", description="model"))
except ImportError:
    ct.fit_eval(df, meta["features"], meta["target"], model="Random Forest")
"""),
        md("""
### Audit by sex -- the check the CT could not do

Train once, then measure how well the model does **separately for women and men**. This is exactly the
fairness question the brain-CT dataset made impossible -- and here, because sex is recorded, you can
answer it. If the model catches strokes less often in one group, that's a real problem to fix
(e.g. a group-aware threshold).
"""),
        code("""
from ipywidgets import interact_manual, Dropdown

def fairness(model):
    ct.audit_by_group(df, meta["features"], meta["target"],
                      meta["group"], meta.get("group_names"), model=model)

try:
    interact_manual(fairness,
        model=Dropdown(options=list(ct.make_models()), value="Random Forest", description="model"))
except ImportError:
    ct.audit_by_group(df, meta["features"], meta["target"], meta["group"], meta.get("group_names"))
"""),
    ]


# =========================================================================== #
# IMAGE template (G2, G3, G4)
# =========================================================================== #
def image_notebook(cfg):
    medmnist = cfg["flag"] not in ("brainct",)
    ensure = ("colab_setup.ensure(*colab_setup.DAY1, \"medmnist\")" if medmnist
              else "colab_setup.ensure(*colab_setup.DAY1)   # our own dataset -> no medmnist needed")
    load_note = "" if medmnist else (
        "\n# 'brainct' is a small brain-CT dataset we prepared for you. It loads just like any other."
    )
    nb = new_nb()
    nb.cells = [
        md(f"""
# Capstone {cfg['gid']}: {cfg['title']}

{cfg['blurb']}

**Your group's priority: {cfg['priority']}.** {cfg['priority_why']}

{_rubric_line()}
"""),
        _how_to_run(),
        _your_job(),
        _setup_cell(cfg["folder"], ensure),
        md("""
### First, some plain-English vocabulary

- A **model** is a rule the computer writes *by itself* after seeing lots of examples -- like a kid
  who learns "cat vs dog" from photos, not from a dictionary definition.
- **Training** = showing it labeled examples so it can find that rule.
- To a computer, an **image is just a grid of numbers** (how bright each pixel is). The model does
  math on those numbers. It never "sees" a picture the way you do.
- We split our data into a **training set** (the model studies these) and a **test set** (held back,
  like a real exam) -- because a model that just *memorized* the study set would ace the homework and
  flunk the exam. The **test score is the only one that counts.**
"""),
        code(f"""
# WHAT THIS DOES: loads the images, splits them into train/test, and tells us the class names.
import torch, sys
sys.path.insert(0, "../..")
import capstone_common as cc

device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
print("device:", device)   # 'cuda' = a GPU is helping (fast). 'cpu' = slower but still fine.{load_note}

FLAG = "{cfg['flag']}"
train_loader, val_loader, test_loader, n_classes, task = cc.get_loaders(FLAG, size=64)
CLASS_NAMES = cc.class_names(FLAG)
print("classes:", n_classes, "->", CLASS_NAMES, "| task:", task)
"""),
        md("### Look at the data first\n**What this does:** draws a few images with their labels. "
           "Always eyeball your data before modeling -- can *you* even tell the classes apart?"),
        code("""
import matplotlib.pyplot as plt
imgs, labels = next(iter(train_loader))   # grab one batch of images
fig, axes = plt.subplots(1, 6, figsize=(12, 2.5))
for ax, im, lb in zip(axes, imgs, labels):
    ax.imshow(im[0], cmap="gray")
    ax.set_title(CLASS_NAMES[int(lb)], fontsize=8); ax.axis("off")
plt.tight_layout()
"""),
        md("""
### Build a first model (the baseline)

**What this does:** trains a model in a few seconds using a shortcut called **transfer learning**,
then prints its **test accuracy** (the fraction it got right on images it never studied).

**Why this works -- the analogy:** instead of teaching a computer to see *from scratch* (which needs
millions of images), we **borrow a network that already learned to see** from millions of everyday
photos. It already knows edges, shapes, and textures. We keep all of that ("**freeze**" it) and only
train a tiny final decision layer (a new "**head**") for our specific job. It's like hiring someone
who already speaks the language and just teaching them the medical words.
"""),
        code("""
model = cc.make_baseline(n_classes)                                   # borrowed "eyes" + a fresh decision layer
model = cc.train(model, train_loader, val_loader, epochs=3, device=device)  # 3 passes over the data
test_acc = cc.evaluate(model, test_loader, device=device)["accuracy"]
print(f"\\nbaseline TEST accuracy: {test_acc:.3f}")   # e.g. 0.800 = right 80% of the time
"""),
        md("""
### Now build your OWN model (interactive -- no coding needed)

**What this does:** gives you dials. **Pick options, press "Run Interact", read the test accuracy.**
Change **one dial at a time** so you can tell what actually helped. Write every result down --
that log is half your presentation.

**What each dial means, in plain terms:**
- **backbone** -- which "brain" to borrow. `resnet18` is small and fast; the others are bigger
  (sometimes smarter, sometimes just slower).
- **pretrained** -- ON = start from the brain that already learned to see (usually much better).
  OFF = start from a *blank* brain. With only a few thousand images, blank is like learning to read
  and diagnose at the same time -- too much at once, so it does worse. **Try toggling this; it's the
  single biggest lesson here.**
- **unfreeze** -- also re-train the borrowed brain, not just the head. More power, but easier to
  **overfit** (memorize instead of learn).
- **augment** -- show each image flipped/rotated. Teaches "the lesion," not "the lesion is top-left."
- **epochs** -- how many times it studies the whole set. More isn't always better (it can start memorizing).

**How to choose (a real strategy, not random clicking):** change **pretrained** first -- it usually
matters most. Then try **augment**, then a bigger **backbone**. **One dial at a time**, or you won't
know which change helped. And prefer the *simplest* setup that hits your goal: a smaller model you
understand beats a bigger one you can't explain.
"""),
        code("""
from ipywidgets import interact_manual, Dropdown, Checkbox, IntSlider

def build_model(backbone="resnet18", pretrained=True, unfreeze_backbone=False, augment=False, epochs=3):
    global model, train_loader, val_loader, test_loader
    train_loader, val_loader, test_loader, n_cls, _ = cc.get_loaders(FLAG, size=64, augment=augment)
    model = cc.make_model(n_cls, backbone=backbone, pretrained=pretrained, unfreeze_backbone=unfreeze_backbone)
    model = cc.train(model, train_loader, val_loader,
                     epochs=epochs, lr=(1e-4 if unfreeze_backbone else 1e-3), device=device)
    acc = cc.evaluate(model, test_loader, device=device)["accuracy"]
    print(f"\\n>>> TEST accuracy = {acc:.3f}   "
          f"[{backbone}, pretrained={pretrained}, unfreeze={unfreeze_backbone}, augment={augment}, {epochs}ep]")

try:
    interact_manual(build_model,
        backbone=Dropdown(options=cc.BACKBONES, value="resnet18", description="backbone"),
        pretrained=Checkbox(value=True, description="pretrained"),
        unfreeze_backbone=Checkbox(value=False, description="unfreeze"),
        augment=Checkbox(value=False, description="augment"),
        epochs=IntSlider(value=3, min=1, max=10, description="epochs"))
except ImportError:
    build_model()
"""),
        md(f"""
### Grade your model like a regulator

A high accuracy is **not** enough for medicine. Yesterday you set the *rules*; now hold **your**
model to them. Pick a tool below and run it. Your group's priority is **{cfg['priority']}**, so the
menu starts on **{cfg['default_tool_label']}** -- but try the others too.

**What each tool tells you (in plain terms):**
{cfg['toolkit_hint']}

**How to choose which audit:** start from *what would hurt a patient most* in your job. Screening
where a miss is deadly? Lead with **failure analysis**. Worried the model treats groups unequally?
**Fairness**. Not sure it's even looking at the right thing? **Grad-CAM**. The priority drives the tool.

*(Train a model first -- run the baseline or the builder above.)*
"""),
        code(f"""
from ipywidgets import interact_manual, Dropdown

TOOLS = list(cc.REGULATOR_TOOLS)
DEFAULT = [t for t in TOOLS if t.startswith("{cfg['default_tool_key']}")][0]

def audit(tool):
    if "model" not in globals():
        print("Train a model first (run the baseline or the builder above)."); return
    cc.REGULATOR_TOOLS[tool](model, test_loader, device=device, class_names=CLASS_NAMES)

try:
    interact_manual(audit, tool=Dropdown(options=TOOLS, value=DEFAULT,
                                          description="priority", style={{"description_width": "initial"}}))
except ImportError:
    audit(DEFAULT)
"""),
        *(_stroke_tabular_cells() if cfg.get("stroke_tabular") else []),
        md(f"""
### One strong approach (peek only if you're stuck)

*This is roughly what the worked solution does. It's a nudge, not the code -- describe these steps to
Claude and build them yourself, then be ready to explain why.*

{cfg['solution_hint']}
"""),
        md(f"""
### Where to take it (ask Claude to help)
{cfg['levelup']}
"""),
    ]
    return nb


# =========================================================================== #
# TABULAR template (G5, G6)
# =========================================================================== #
def tabular_notebook(cfg):
    nb = new_nb()
    cells = [
        md(f"""
# Capstone {cfg['gid']}: {cfg['title']}

{cfg['blurb']}

**Your group's priority: {cfg['priority']}.** {cfg['priority_why']}

{_rubric_line()}
"""),
        _how_to_run(),
        _your_job(),
        _setup_cell(cfg["folder"], "colab_setup.ensure(*colab_setup.DAY3_TABULAR)"),
        md("""
### Plain-English vocabulary

- This is **tabular** data -- a **spreadsheet**, one row per patient, one column per measurement.
- The columns you use as clues are **features** (age, cholesterol, ...). The column you're trying to
  predict is the **target** (does this person have the disease -- yes/no).
- A **model** finds a rule linking the features to the target. We test it on patients it never saw,
  because a model that *memorized* the training rows would look smart and be useless on new people.
"""),
        code(f"""
# WHAT THIS DOES: loads the spreadsheet and shows the first few rows.
import sys
sys.path.insert(0, "../..")
import capstone_tabular as ct

df, meta = ct.{cfg['loader']}()
print(meta["name"], "  ->", df.shape[0], "patients,", len(meta["features"]), "features")
print("features (the clues):", meta["features"])
print("predicting:", meta["target"], f"(1 = {{meta['positive']}})")
df.head()   # each row = one person; each column = one measurement
"""),
        md("### Look before you model\n**What this does:** counts how many people fall in each class and "
           "summarizes each feature. Ask: is it balanced? Could two features be tangled together?"),
        code(f"""
import matplotlib.pyplot as plt
print(df[meta["target"]].value_counts().rename({{0: "no", 1: "yes"}}).to_string())
df[meta["features"]].describe().T[["mean", "min", "max"]]   # typical / smallest / largest value per feature
"""),
        md("""
### Build your own model (interactive)

**What this does:** you pick which **features** the model may look at and which **model** to use, then
it reports the test accuracy. **Try dropping features** -- does accuracy hold with fewer? That tells
you which measurements actually carry the signal (and which are just along for the ride).

**How to choose a model:** start with **Logistic Regression** -- it's simple and you can explain
exactly why it decided what it did (gold in medicine). Only reach for Random Forest / CatBoost /
TabPFN if the simple one plateaus *and* you suspect the clues interact in tangled ways. On a small
spreadsheet, fancier often just overfits. Simplest tool that does the job wins.
"""),
        code("""
from ipywidgets import interact_manual, SelectMultiple, Dropdown

def build(features, model):
    if not features:
        print("Pick at least one feature."); return
    ct.fit_eval(df, list(features), meta["target"], model=model)

try:
    interact_manual(build,
        features=SelectMultiple(options=meta["features"], value=tuple(meta["features"]),
                                description="features", rows=min(10, len(meta["features"])),
                                style={"description_width": "initial"}),
        model=Dropdown(options=list(ct.make_models()), value="Logistic Regression", description="model"))
except ImportError:
    ct.fit_eval(df, meta["features"], meta["target"])
"""),
        md(f"""
### Fairness: does it work equally well for everyone?

Your priority is **{cfg['priority']}**. **What this does:** trains once, then reports the test accuracy
**separately for each group** ({cfg['group_desc']}).

**Why this matters -- the analogy:** a restaurant can average 4.5 stars while giving every vegetarian
food poisoning. An "average" hides the group it fails. A medical model that's accurate overall but
worse for one group is exactly that restaurant.
"""),
        code("""
from ipywidgets import interact_manual, Dropdown

def fairness(model):
    ct.audit_by_group(df, meta["features"], meta["target"],
                      meta["group"], meta.get("group_names"), model=model)

try:
    interact_manual(fairness,
        model=Dropdown(options=list(ct.make_models()), value="Logistic Regression", description="model"))
except ImportError:
    ct.audit_by_group(df, meta["features"], meta["target"], meta["group"], meta.get("group_names"))
"""),
    ]
    if cfg.get("confounding"):
        cells += [
            md("""
### The confounding trap (this is the heart of your project)

**The big idea -- the analogy:** ice-cream sales and shark attacks rise together every summer. Does
ice cream cause shark attacks? No -- **summer heat** drives both. Heat is a **confounder**: a hidden
third thing that makes two others *look* connected when neither causes the other.

**In your data:** women who used estrogen were also, on average, wealthier, more educated, and
healthier to begin with. So "estrogen users score better on the memory test" might be those
*advantages*, not the estrogen at all.

**What this does:** measures the effect of one feature **alone**, then **again after "controlling for"**
the confounders. "Controlling for age/education/income" means: compare women who are otherwise
*alike* in those things, so whatever difference is left is closer to the estrogen's *real* effect.
**If the effect shrinks toward zero once you control, the original number was mostly the confounders --
not cause and effect.**
"""),
            code(f"""
from ipywidgets import interact_manual, Dropdown, SelectMultiple

others = [f for f in meta["features"] if f != "{cfg['confound_feature']}"]

def confound(feature, controls):
    ct.association(df, feature, meta["target"], list(controls))

try:
    interact_manual(confound,
        feature=Dropdown(options=meta["features"], value="{cfg['confound_feature']}", description="effect of"),
        controls=SelectMultiple(options=others, value={cfg['confound_controls']!r},
                                description="control for", rows=min(6, len(others)),
                                style={{"description_width": "initial"}}))
except ImportError:
    ct.association(df, "{cfg['confound_feature']}", meta["target"], list({cfg['confound_controls']!r}))
"""),
        ]
    cells.append(md(f"""
### One strong approach (peek only if you're stuck)

*This is roughly what the worked solution does. It's a nudge, not the code -- describe these steps to
Claude and build them yourself, then be ready to explain why.*

{cfg['solution_hint']}
"""))
    cells.append(md(f"""
### Where to take it (ask Claude to help)
{cfg['levelup']}
"""))
    nb.cells = cells
    return nb


# =========================================================================== #
# SEQUENCE template (G1)
# =========================================================================== #
def sequence_notebook(cfg):
    nb = new_nb()
    nb.cells = [
        md(f"""
# Capstone {cfg['gid']}: {cfg['title']}

{cfg['blurb']}

**Your group's priority: {cfg['priority']}.** {cfg['priority_why']}

{_rubric_line()}
"""),
        _how_to_run(),
        _your_job(),
        _setup_cell(cfg["folder"], "colab_setup.ensure(*colab_setup.DAY3_SEQ)"),
        md("""
### Plain-English vocabulary

- **CRISPR** is molecular scissors. You point it at a spot in DNA with a short **guide** (20 letters of
  A/C/G/T). Some guides cut well, some barely cut. You want to predict *which*, before spending weeks
  in the lab -- that's the tool you're building.
- A **model** is a rule the computer learns from examples of guides that did and didn't cut well.
- Here the data is **text** (DNA letters), not pixels or a spreadsheet -- so your first real decision
  is *how to turn letters into numbers*, because a computer can't do math on the letter "A".
"""),
        code("""
# WHAT THIS DOES: loads 5,310 real guides, each with a score for how well it cut.
import sys
sys.path.insert(0, "../..")
import capstone_seq as cs

df, meta = cs.load_guides()
print(meta["name"], "  ->", df.shape[0], "guides")
print(meta["citation"])
df[["guide30", "guide20", "gene", "gc_content", "efficiency", "high_efficiency"]].head()
"""),
        md("""
### How to read one guide

Each row is a 30-letter stretch of DNA around one cut site:

```
[ 4 letters ][   20-letter guide   ][ NGG ][ 3 letters ]
  context        this is the guide     PAM     context
```

- The **guide** (middle 20) is the part you design.
- The **PAM** (the "NGG") is a landmark the scissors grab onto.
- **Key biology:** the scissors first grab the PAM, then check the ~10 letters *right next to it* --
  called the **seed**. If the seed doesn't match well, it won't cut. So letters near the PAM should
  matter most. **Remember this -- your model should rediscover it.**
- `efficiency` = how well it cut (0 to 1). `high_efficiency` = 1 for the best cutters, else 0.
"""),
        md("""
### A first, no-math guess: GC content

**What this does:** checks whether one simple number -- the fraction of G's and C's in the guide --
lines up with efficiency. (G-C pairs stick together more tightly, so people suspected they'd matter.)
**Why bother:** it's a baseline. If a fancy model can't beat one obvious number, it isn't earning its keep.
"""),
        code("cs.gc_vs_efficiency(df)"),
        md("""
### Turn letters into numbers -- two ways (this is the whole project)

A model needs numbers. There are two ways to convert a sequence, and the difference *is* the lesson:

- **one-hot** -- give every position 4 light-switches (A, C, G, T) and flip on the one that's there.
  This remembers **where** each letter sits. (Each position gets 4 switches, so the 20-letter guide
  becomes 20 x 4 = 80 numbers.)
- **k-mer** -- just **count** how many A/C/G/T (and letter-pairs) there are, ignoring order.

**The analogy:** "LISTEN" and "SILENT" have the *exact same letters*. k-mer counting can't tell them
apart -- it threw away the order. one-hot keeps the order. Since biology cares *where* a letter is
(the seed!), one-hot should win. Run it both ways below and see.

**How to choose a representation:** match the tool to the biology. If *order* matters -- and here it
does -- pick the one that keeps order (one-hot). If only the *amount* of each letter mattered, plain
counting (k-mer) would be simpler and just as good. Picking the representation *is* the science; the
model is almost an afterthought.

**One honest catch:** only about **1 in 5** guides is high-efficiency. So a lazy model that *always*
says "low" is right 80% of the time -- and completely useless. That's why we watch **AUC** instead of
accuracy: AUC ignores that cheat (0.5 = coin flip, 1.0 = perfect).
"""),
        code("""
from ipywidgets import interact_manual, Dropdown

def build(representation, sequence, model):
    cs.fit_eval_class(df, mode=representation, seq_col=sequence, model=model)

try:
    interact_manual(build,
        representation=Dropdown(options=["onehot", "kmer"], value="onehot", description="representation",
                                style={"description_width": "initial"}),
        sequence=Dropdown(options=["guide20", "guide30"], value="guide20", description="sequence"),
        model=Dropdown(options=list(cs.make_classifiers()), value="CatBoost", description="model"))
except ImportError:
    cs.fit_eval_class(df, mode="onehot", model="CatBoost")
"""),
        md("""
**Do this:** run it once with **onehot**, once with **kmer**, same model. Watch the **AUC**. one-hot
should win -- and that's a real result: *the way you prepared the data mattered more than the model.*
"""),
        md("""
### Predict the exact score instead of yes/no (regression)

"High vs low" throws away detail. The real target is a **number** from 0 to 1. Can the model predict it?

**How to read the score:** **R²** answers "how much of the ups-and-downs did the model actually
explain?" R² = 0 means "no better than always guessing the average"; 1.0 means perfect.

**Yes/no or a number -- how to choose?** Match it to the *decision you'll make*. If the choice is
binary (cut here or not), yes/no is enough. If you must **rank** many options -- say, pick the top 3
guides out of 100 to order -- you need the number. The task decides the tool, not the other way around.
"""),
        code("""
from ipywidgets import interact_manual, Dropdown

def build_reg(representation, model):
    cs.fit_eval_reg(df, mode=representation, model=model)

try:
    interact_manual(build_reg,
        representation=Dropdown(options=["onehot", "kmer"], value="onehot", description="representation",
                                style={"description_width": "initial"}),
        model=Dropdown(options=list(cs.make_regressors()), value="Random Forest", description="model"))
except ImportError:
    cs.fit_eval_reg(df, mode="onehot", model="Random Forest")
"""),
        md(f"""
### Did the model learn real biology? (your headline slide)

Your priority is **interpretability** -- not just *is it right*, but *is it right for the right reason*.

**What this does:** trains a simple model and plots **how much each position** in the guide matters.
**Why it's the payoff:** if the tall bars land near the PAM end (the **seed** you read about above),
then your model figured out real CRISPR biology *on its own, from data* -- nobody told it about seeds.
A black box you can explain is a black box people will trust.
"""),
        code('cs.position_importance(df)'),
        md(f"""
### One strong approach (peek only if you're stuck)

*This is roughly what the worked solution does. It's a nudge, not the code -- describe these steps to
Claude and build them yourself, then be ready to explain why.*

{cfg['solution_hint']}
"""),
        md(f"""
### Where to take it (ask Claude to help)
{cfg['levelup']}
"""),
    ]
    return nb


# =========================================================================== #
# The six groups
# =========================================================================== #
GROUPS = [
    dict(builder=sequence_notebook, folder="g1_crispr", gid="G1",
         title="Designing better CRISPR guides",
         blurb="Given a 20-letter guide, will the CRISPR scissors cut well there? You're building the "
               "tool that helps a scientist pick a good guide *before* running a slow, costly experiment.",
         priority="interpretability",
         priority_why="A lab scientist won't trust a mystery box telling them which guide to order -- your "
                      "model has to *show* it reads the DNA the way the biology actually works.",
         default_tool_label="Interpretability",
         solution_hint="Turn each guide into numbers with **one-hot encoding** (four on/off slots per "
                       "position, which keeps the letter *order*). Frame the task as **clearly-good vs "
                       "clearly-poor** cutters (the top third vs the bottom third of efficiency) -- a cleaner, "
                       "easier target than a fuzzy middle. **Bake off a few models** and let the score pick "
                       "the winner: a gradient-boosted tree (CatBoost) tends to win here, even beating TabPFN. "
                       "Then look at **which positions** the model relies on -- they should cluster in the "
                       "**seed region next to the PAM**. If they do, your model rediscovered real CRISPR biology.",
         levelup="- Train on all genes but one, then test on the held-out gene -- does guide design carry "
                 "over to a gene it never saw?\n"
                 "- Invent your own clue by hand (e.g. 'is there a G right before the PAM?') and see if it helps.\n"
                 "- Look up a real guide-design website (Benchling, CRISPOR) and compare its advice to your model's."),

    dict(builder=image_notebook, folder="g2_skin_screening", gid="G2", flag="dermamnist",
         title="Skin cancer screening",
         blurb="Seven kinds of skin spot from dermatoscope photos -- including melanoma, the dangerous one. "
               "This is a **screening** tool, so its #1 job is to *not miss* a cancer.",
         priority="sensitivity (don't miss the dangerous class)",
         priority_why="In screening, missing a real melanoma is far worse than a false alarm: a false alarm "
                      "gets a second look, a miss goes home untreated. Overall accuracy can look great while "
                      "the model quietly misses the rare, deadly class.",
         default_tool_key="Failure", default_tool_label="Failure analysis",
         toolkit_hint="- **Failure analysis** -- shows the melanoma cases it got wrong. For screening, that's the whole ballgame.\n"
                      "- **Fairness** -- is accuracy even across all 7 spot types, or does it flunk the rare ones?\n"
                      "- **Confusion matrix** -- a scorecard of truth vs guess. **Grad-CAM** -- a heat-map of where it looked. "
                      "**Monitoring** -- does it survive blurrier photos?",
         solution_hint="Screening cares about **sensitivity** (catching every melanoma) far more than raw "
                       "accuracy. So after you train, don't just accept the default 0.5 cutoff -- **lower the "
                       "decision threshold** so the model flags more suspicious spots, catching more melanomas "
                       "even at the cost of a few extra false alarms. Report melanoma **recall before vs after** "
                       "tuning. Also try a **stronger backbone** than a small ResNet, and check with **Grad-CAM** "
                       "that the model is looking at the lesion, not the background.",
         levelup="- Which class does it miss most? Try **class weighting** or **augment** to fix it, and prove "
                 "the fix on the confusion matrix.\n"
                 "- Swap in the full HAM10000 set (Kaggle), then feed it a phone photo of a mole -- watch it "
                 "stumble on data that looks nothing like training."),

    dict(builder=image_notebook, folder="g3_brain_ct", gid="G3", flag="brainct", stroke_tabular=True,
         title="Head trauma / stroke on brain CT -- and who's missing from the data",
         blurb="Brain CT scans, labeled normal vs stroke (a bleed or blockage -- what an ER checks a "
               "head-injury patient for). You'll build the classifier AND uncover a quieter problem: "
               "who is *not* in this dataset.",
         priority="fairness you can't even measure",
         priority_why="This dataset gives you an image and a label and NOTHING else -- no age, no sex, no "
                      "scanner, no hospital. So you literally *can't* check whether it works equally for "
                      "everyone. That blank is your finding: a dataset that hides who's in it.",
         default_tool_key="Failure", default_tool_label="Failure analysis",
         toolkit_hint="- **Failure analysis** -- which scans does it miss? Any visible pattern?\n"
                      "- **Fairness** -- here it can only compare *classes* (normal vs stroke), because the data "
                      "records no age/sex/race at all. Notice what you're *unable* to check -- that's the point.\n"
                      "- **Grad-CAM** -- is it looking at brain tissue, or cheating off the skull / scanner edges?",
         solution_hint="Build the stroke detector first (transfer learning on the CT slices). Then notice the "
                       "honest catch: this image dataset records **no demographics** (age/sex/race), so you "
                       "literally **cannot audit** whether it is fair. The strong move is to bring in a separate "
                       "**tabular stroke dataset that DOES record sex and age**, build a stroke-risk model there, "
                       "and check whether it catches strokes as often in women as in men (equalize with "
                       "group-aware thresholds if not). The lesson: **recording demographics is what makes "
                       "fairness checkable** -- the imaging set hides who is in it.",
         levelup="- List every fact a hospital would need before trusting this (age, sex, race, scanner, site) "
                 "and which ones this dataset gives you. The gap is your headline.\n"
                 "- Use **Monitoring** to add noise -- a real ER scanner is messier than clean training data.\n"
                 "- Try to find a brain-CT dataset that *does* report demographics. How rare is that?"),

    dict(builder=image_notebook, folder="g4_skin_access", gid="G4", flag="dermamnist",
         title="Skin cancer AI and who gets access to it",
         blurb="Same 7-class skin model as the screening group -- but your question is about **equity**: "
               "who does a dermatology AI actually help, and who gets left out?",
         priority="fairness and access",
         priority_why="Dermatology AI is often trained mostly on light skin and deployed where dermatologists "
                      "already exist. A model can be accurate on average and still widen the gap for the people "
                      "who most lack access to a specialist.",
         default_tool_key="Fairness", default_tool_label="Fairness (accuracy per class)",
         toolkit_hint="- **Fairness** -- is accuracy even across all 7 spot types? Uneven accuracy is an access "
                      "problem: the model helps some patients more than others.\n"
                      "- **Failure analysis** -- who does it fail, and could they even get a second opinion?\n"
                      "- **Confusion matrix / Grad-CAM / Monitoring** -- the other angles.",
         solution_hint="Don't report one accuracy number -- **break it down by group**. Use a dataset that "
                       "records **skin tone** (the Fitzpatrick scale) and measure accuracy separately for light, "
                       "medium, and dark skin. You'll likely find the real problem isn't the model, it's the "
                       "**data**: there are almost no dark-skin patients. The honest finding is that **you cannot "
                       "validate a model for a group that's barely in the dataset** -- and the fix is collecting "
                       "representative data, not a coding trick.",
         levelup="- Research the **skin-tone gap** in dermatology data (the Fitzpatrick scale). Does this dataset "
                 "even tell you which skin tones it trained on? (It doesn't -- like G3, that blank is a finding.)\n"
                 "- Debate it: if this model runs in a clinic with no dermatologist, does it help or harm? "
                 "Use your fairness numbers to back up a position."),

    dict(builder=tabular_notebook, folder="g5_heart", gid="G5", loader="load_heart",
         title="Predicting heart disease from a checkup",
         blurb="Thirteen numbers from a routine heart checkup -- cholesterol, blood pressure, age, chest-pain "
               "type, an exercise test -- used to flag heart disease. A classic prediction task on 303 patients.",
         priority="fairness across sex",
         priority_why="Heart disease shows up differently in women and has long been under-diagnosed in them. "
                      "A model trained on mostly-male data can be accurate overall yet worse for women.",
         group_desc="women vs men",
         solution_hint="**Bake off a few models** (logistic regression, random forest, CatBoost, TabPFN) and let "
                       "the AUC pick the winner. Then ask **which clues actually matter**: compare a "
                       "cholesterol-only model to the full one (cholesterol alone is surprisingly weak), and use "
                       "**SHAP** to see the real drivers. Finally, **audit accuracy separately for women and men** "
                       "-- heart disease is historically under-diagnosed in women, and this data skews male.",
         levelup="- Does **cholesterol alone** predict much? Compare a cholesterol-only model to the full one -- "
                 "you may be surprised which clues carry the signal.\n"
                 "- The data is about 2-to-1 male. Could that explain any accuracy gap by sex? What would you "
                 "collect to fix it?\n"
                 "- Turn the output into a risk *percentage* instead of yes/no -- how would a doctor use that?"),

    dict(builder=tabular_notebook, folder="g6_estrogen", gid="G6", loader="load_estrogen",
         title="Estrogen, cognition, and whose symptoms get believed",
         blurb="Older women (60+) from a national health survey: does estrogen/hormone use relate to memory-test "
               "scores? A small dataset with a big hidden trap -- and a question about what tests really capture.",
         priority="confounding vs causation, and the patient's own account",
         priority_why="Women who used estrogen were also, on average, healthier, wealthier, and better-educated. "
                      "So a raw 'users score better' can be entirely those *other* things. And a memory test "
                      "never captures the 'brain fog' a patient actually reports feeling.",
         group_desc="estrogen users vs non-users",
         confounding=True, confound_feature="used_estrogen",
         confound_controls=("age", "education", "income_ratio"),
         solution_hint="This is a **causation** question, not a prediction one -- so use an **interpretable model "
                       "(logistic regression)** whose effect size you can actually read. Measure estrogen's "
                       "apparent effect on cognition **alone**, then **again after controlling for age, education, "
                       "and income**. Watch the effect **shrink** -- most of the 'estrogen helps' signal is "
                       "**healthy-user bias** (users were healthier and wealthier to begin with). A black-box "
                       "model would predict just as well but couldn't answer the *why*, which is the whole point.",
         levelup="- After controlling for age/education/income, how much of the 'estrogen helps memory' effect "
                 "is left? Write the honest one-sentence conclusion.\n"
                 "- The test is *objective* (a number); brain fog is *subjective* (a feeling). Argue why the gap "
                 "between the two is itself the story a patient would tell.\n"
                 "- Reflect: what happens to care when a woman's symptoms get waved off because a test "
                 "'looks normal'? Tie your numbers to that."),
]


def build():
    for cfg in GROUPS:
        folder = BASE / cfg["folder"]
        folder.mkdir(parents=True, exist_ok=True)
        nb = cfg["builder"](cfg)
        save(nb, folder / "starter.ipynb")
        colab = (f"https://colab.research.google.com/github/jinchiwei/outset-ai-healthcare/"
                 f"blob/main/notebooks/day3_capstone/projects/{cfg['folder']}/starter.ipynb")
        print(f"wrote {cfg['gid']}  {cfg['folder']}/starter.ipynb")
        print(f"     {colab}")


if __name__ == "__main__":
    build()
