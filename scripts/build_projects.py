"""Build the six TAILORED capstone starter notebooks (one per project group).

Day 3 groups don't pick a generic MedMNIST -- each has a real question. This script
generates one starter notebook per group from three templates:

  IMAGE     (capstone_common.py)  -- G2 skin screening, G3 brain CT, G4 skin access
  TABULAR   (capstone_tabular.py) -- G5 heart/cholesterol, G6 estrogen & cognition
  SEQUENCE  (capstone_seq.py)     -- G1 CRISPR guide efficiency

Every notebook: names the group's task in the intro, states their priority, pre-opens the
tool that matches that priority, and ends on a runnable baseline they extend with Claude.

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


def _setup_cell(folder, ensure_line):
    """Colab-safe setup: clone the repo if needed, wire the import paths, install deps."""
    return code(f"""
# Setup: on Colab this grabs the course files + installs what's missing. Locally it's a no-op.
import os, sys
if not os.path.exists("../../capstone_common.py"):
    os.system("git clone -q {REPO}")
    os.chdir("outset-ai-healthcare/notebooks/day3_capstone/projects/{folder}")
sys.path.insert(0, "../..")            # day3_capstone (capstone_common / _tabular / _seq)
sys.path.insert(0, "../../../_shared") # colab_setup
import colab_setup
{ensure_line}
""")


# =========================================================================== #
# IMAGE template (G2, G3, G4)
# =========================================================================== #
def image_notebook(cfg):
    medmnist = cfg["flag"] not in ("brainct",)
    ensure = ("colab_setup.ensure(*colab_setup.DAY1, \"medmnist\")" if medmnist
              else "colab_setup.ensure(*colab_setup.DAY1)   # local dataset -> no medmnist needed")
    load_note = "" if medmnist else (
        "\n# 'brainct' is a small brain-CT set we built for this project (datasets/brain_ct.npz)."
        "\n# It loads exactly like a MedMNIST set -- same code below works unchanged.")
    nb = new_nb()
    nb.cells = [
        md(f"""
# Capstone {cfg['gid']}: {cfg['title']}

{cfg['blurb']}

**Your group's priority: {cfg['priority']}.** {cfg['priority_why']}

This notebook gives you a **working baseline**. Your job is to make it better and be able to
explain every change. Use Claude as your pair programmer. **Rubric:** build it, evaluate it
honestly, find a failure mode, defend one design decision, both partners can explain the work.
"""),
        _setup_cell(cfg["folder"], ensure),
        code(f"""
import torch, sys
sys.path.insert(0, "../..")
import capstone_common as cc

device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
print("device:", device){load_note}

FLAG = "{cfg['flag']}"
train_loader, val_loader, test_loader, n_classes, task = cc.get_loaders(FLAG, size=64)
CLASS_NAMES = cc.class_names(FLAG)
print("classes:", n_classes, "->", CLASS_NAMES, "| task:", task)
"""),
        md("## A look at the data\nBefore modeling, always look. What do these images actually show?"),
        code("""
import matplotlib.pyplot as plt
imgs, labels = next(iter(train_loader))
fig, axes = plt.subplots(1, 6, figsize=(12, 2.5))
for ax, im, lb in zip(axes, imgs, labels):
    ax.imshow(im[0], cmap="gray")
    ax.set_title(CLASS_NAMES[int(lb)], fontsize=8); ax.axis("off")
plt.tight_layout()
"""),
        md("## Baseline: transfer learning (frozen ResNet18 + new head)\nThe Day 1 trick. Run it, note the TEST accuracy, then try to beat it."),
        code("""
model = cc.make_baseline(n_classes)
model = cc.train(model, train_loader, val_loader, epochs=3, lr=1e-3, device=device)
test_acc = cc.evaluate(model, test_loader, device=device)["accuracy"]
print(f"\\nbaseline TEST accuracy: {test_acc:.3f}")
"""),
        md("""
## Build your own model (interactive)

**Pick options, hit Run, read the TEST accuracy.** Change **one** thing at a time and watch
what actually moves the number. Log every result -- that log is half your presentation.

- **backbone** -- the architecture (resnet18 is small/fast; resnet50, densenet are bigger).
- **pretrained** -- start from ImageNet weights (on) or scratch (off). *Usually matters a lot.*
- **unfreeze** -- train the whole network vs just a new head. More power, more overfitting.
- **augment** -- add flips/rotations. **epochs** -- how long to train.
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
## The regulator's toolkit: audit your own model

Yesterday you set the *rules* medical AI must follow. Now hold **your** model to them. Your
group's priority is **{cfg['priority']}**, so the dropdown starts on **{cfg['default_tool_label']}** --
but run the others too. A high accuracy is not enough if it fails one of these.

{cfg['toolkit_hint']}

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
        md(f"""
## Where to take it (with Claude)
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

This is a **tabular** problem -- a table of numbers per person, like Day 2. The model is easy;
the interesting questions are about *which features* and *who the model works for*. **Rubric:**
build it, evaluate honestly, find a failure mode, defend a decision, both partners can explain.
"""),
        _setup_cell(cfg["folder"], "colab_setup.ensure(*colab_setup.DAY3_TABULAR)"),
        code(f"""
import sys
sys.path.insert(0, "../..")
import capstone_tabular as ct

df, meta = ct.{cfg['loader']}()
print(meta["name"], "  ->", df.shape[0], "rows")
print("features:", meta["features"])
print("predicting:", meta["target"], f"(1 = {{meta['positive']}})")
df.head()
"""),
        md("## Look at the data\nHow many people are in each class? Is it balanced? What might be confounded?"),
        code(f"""
import matplotlib.pyplot as plt
print(df[meta["target"]].value_counts().rename({{0: "no", 1: "yes"}}).to_string())
df[meta["features"]].describe().T[["mean", "min", "max"]]
"""),
        md("""
## Build your own model (interactive)

Pick which **features** the model sees and which **model** to use, hit Run, read the TEST
accuracy. Try dropping features: does accuracy hold with fewer? Which features actually matter?
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
## Fairness: does the model work equally well for everyone?

Your priority is **{cfg['priority']}**. Train once, then measure TEST accuracy **separately for
each group** ({cfg['group_desc']}). A model can look great overall and quietly fail one group.
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
## The confounding trap (the heart of this project)

An association is not a cause. Below, measure the effect of one feature on the outcome **alone**,
then **again after controlling** for the obvious confounders. If the effect shrinks toward zero,
the raw association was mostly the confounders -- not the thing itself.

This is the whole question for your group: does the exposure *cause* the outcome, or do the two
just travel together because of age, education, and income?
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
## Where to take it (with Claude)
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

This one is neither pixels nor a tidy table -- it's **text**: a DNA sequence. A computer can't
multiply the letter "A", so your first real decision is *how to turn a sequence into numbers*.
That choice matters more than the model. **Rubric:** build it, evaluate honestly, find a failure
mode, defend a decision, both partners can explain the work.
"""),
        _setup_cell(cfg["folder"], "colab_setup.ensure(*colab_setup.DAY3_SEQ)"),
        code("""
import sys
sys.path.insert(0, "../..")
import capstone_seq as cs

df, meta = cs.load_guides()
print(meta["name"], "  ->", df.shape[0], "guides")
print(meta["citation"])
df[["guide30", "guide20", "gene", "gc_content", "efficiency", "high_efficiency"]].head()
"""),
        md("""
## Reading a guide

Each row is a 30-letter window of DNA around one cut site:

```
[ 4 nt context ][   20 nt guide (protospacer)   ][ NGG PAM ][ 3 nt context ]
    1 - 4              5 - 24                       25 - 27      28 - 30
```

The **guide** (positions 5-24) is what you design; the **PAM** (NGG) is where Cas9 grabs on.
`efficiency` is how well that guide cut (0..1). `high_efficiency` = 1 for the best cutters.
The bases nearest the PAM -- the **seed region** -- are known to matter most. Keep that in mind.
"""),
        md("## A first, human-readable guess: GC content\nBefore any ML, does one simple number -- the fraction of G/C bases -- track efficiency?"),
        code("cs.gc_vs_efficiency(df)"),
        md("""
## Build your own model (interactive)

Two decisions: **how to featurize** the sequence, and **which model** to use.

- **representation** -- `onehot` keeps *where* each base sits (position-aware). `kmer` only
  counts how many of each base and base-pair (position-blind: it throws the order away).
- **sequence** -- `guide20` (just the guide) or `guide30` (guide + genomic context).
- **model** -- Logistic Regression, Random Forest, a small Neural Net, or CatBoost.

Because only ~1 in 5 guides is high-efficiency, **accuracy alone is misleading** -- a model that
says "low" every time scores 80%. Watch the **AUC** instead: 0.5 is guessing, 1.0 is perfect.
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
### Compare onehot vs kmer

Run the builder twice -- once with `onehot`, once with `kmer`, same model. `onehot` should win.
Why? Because *where* a base sits matters (the seed region), and `kmer` throws that away. That gap
is a real result: **the representation, not the model, is doing the work.**
"""),
        md("""
## Predict the exact efficiency (regression)

`high_efficiency` is a yes/no simplification. The real target is a *number* from 0 to 1. Can the
model predict it? R^2 = 0 means "no better than always guessing the average"; higher is better.
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
## {cfg['default_tool_label']}: did the model learn real biology?

Your priority is **interpretability** -- not just *is it accurate*, but *is it right for the right
reason*. Train a linear model on the one-hot features and plot how much each **position** in the
guide matters. If the bases near the PAM (the right-hand end -- the seed) matter most, your model
rediscovered known CRISPR biology from data alone. That's your headline slide.
"""),
        code('cs.position_importance(df)'),
        md(f"""
## Where to take it (with Claude)
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
         blurb="Given a 20-letter guide RNA, will CRISPR-Cas9 cut efficiently there? You're building "
               "the tool that helps a scientist pick a guide *before* running a costly experiment.",
         priority="interpretability",
         priority_why="A wet-lab scientist won't trust a black box telling them which guide to order -- "
                      "your model has to show it's reading the sequence the way biology actually works.",
         default_tool_label="Interpretability",
         levelup="- Filter to one **gene** and see if a model trained on the others still works on it "
                 "(does guide design generalize across genes?).\n"
                 "- Engineer a feature by hand (e.g. 'is there a G right before the PAM?') and see if it helps.\n"
                 "- Look up a real guide-design tool (Benchling, CRISPOR) and compare its advice to your model's."),

    dict(builder=image_notebook, folder="g2_skin_screening", gid="G2", flag="dermamnist",
         title="Skin cancer screening",
         blurb="Seven kinds of skin lesion from dermatoscope images -- including melanoma, the "
               "dangerous one. This is a **screening** tool: its job is to not miss cancer.",
         priority="sensitivity (don't miss the dangerous class)",
         priority_why="In screening, a missed melanoma (false negative) is far worse than a false alarm. "
                      "Overall accuracy can look great while the model quietly misses the rare, deadly class.",
         default_tool_key="Failure", default_tool_label="Failure analysis",
         toolkit_hint="- **Failure analysis** -- look at the melanoma cases it got wrong. That's the whole ballgame for screening.\n"
                      "- **Fairness** -- is accuracy even across the 7 classes, or does it fail the rare ones?\n"
                      "- Then try the others (confusion matrix, Grad-CAM, monitoring).",
         levelup="- Which class does it miss most? Try **class weighting** or **augmentation** to fix it, "
                 "and prove the fix with the confusion matrix.\n"
                 "- Swap in full HAM10000 (Kaggle), then try a phone photo of a mole -- watch it break on "
                 "data that looks nothing like training."),

    dict(builder=image_notebook, folder="g3_brain_ct", gid="G3", flag="brainct",
         title="Head trauma / stroke on brain CT -- and who's missing from the data",
         blurb="Brain CT slices, labelled normal vs stroke (an acute bleed or infarct -- what an ER scans "
               "a head-trauma patient for). You'll build the classifier AND investigate a quieter problem: "
               "who is *not* represented in this dataset.",
         priority="fairness you can't even measure",
         priority_why="This dataset ships an image and a label and NOTHING else -- no age, no sex, no "
                      "scanner, no hospital. So you literally cannot check whether the model works equally "
                      "for everyone. That absence is your finding: a dataset that hides who's in it.",
         default_tool_key="Failure", default_tool_label="Failure analysis",
         toolkit_hint="- **Failure analysis** -- which slices does it miss? Any visible pattern?\n"
                      "- **Fairness** -- here it can only split by *class* (normal vs stroke), because the "
                      "dataset records no demographics at all. Notice what you're *unable* to audit.\n"
                      "- **Grad-CAM** -- is it looking at brain tissue, or at the skull / scanner edges?",
         levelup="- Write down every demographic a hospital would need before deploying this (age, sex, "
                 "race, scanner, site) and which ones this dataset provides. The gap is your headline.\n"
                 "- Add noise with the **monitoring** tool: a real ER scanner is messier than clean training data.\n"
                 "- Find a brain-CT dataset that *does* report demographics -- how rare is that?"),

    dict(builder=image_notebook, folder="g4_skin_access", gid="G4", flag="dermamnist",
         title="Skin cancer AI and who gets access to it",
         blurb="Same 7-class skin-lesion model as the screening group -- but your question is about "
               "**equity**: who does a dermatology AI actually help, and who gets left out?",
         priority="fairness and access",
         priority_why="Dermatology AI is often trained mostly on light skin and deployed where dermatologists "
                      "already exist. A model can be accurate on average and still widen the gap for the "
                      "people who most lack access to a specialist.",
         default_tool_key="Fairness", default_tool_label="Fairness (accuracy per class)",
         toolkit_hint="- **Fairness** -- is accuracy even across all 7 lesion types? Uneven accuracy across "
                      "conditions is an access problem: the model helps some patients more than others.\n"
                      "- **Failure analysis** -- who does it fail, and would they be able to get a second opinion?\n"
                      "- Then confusion matrix / Grad-CAM / monitoring.",
         levelup="- Research the **skin-tone gap** in dermatology datasets (Fitzpatrick scale). Does MedMNIST "
                 "even tell you the skin tones it was trained on? (It doesn't -- like G3, that gap is a finding.)\n"
                 "- Argue it out: if this model runs in a clinic with no dermatologist, does it help or harm? "
                 "Use your fairness numbers to defend a position."),

    dict(builder=tabular_notebook, folder="g5_heart", gid="G5", loader="load_heart",
         title="Predicting heart disease from a checkup",
         blurb="Thirteen numbers from a routine cardiac workup -- cholesterol, blood pressure, age, chest-pain "
               "type, an exercise test -- to flag heart disease. Classic clinical prediction on 303 patients.",
         priority="fairness across sex",
         priority_why="Heart disease presents differently in women and is historically under-diagnosed in them. "
                      "A model trained on mostly-male data can be accurate overall yet worse for women.",
         group_desc="women vs men",
         levelup="- Does **cholesterol** alone predict much? Compare a chol-only model to the full one -- you "
                 "may be surprised which features carry the signal.\n"
                 "- The dataset is ~2:1 male. Does that explain the accuracy gap by sex? What would you collect "
                 "to fix it?\n"
                 "- Turn it into a risk *score* (probability) instead of yes/no -- how would a doctor use that?"),

    dict(builder=tabular_notebook, folder="g6_estrogen", gid="G6", loader="load_estrogen",
         title="Estrogen, cognition, and whose symptoms get believed",
         blurb="Older women (60+) from a national health survey: does hormone/estrogen use relate to measured "
               "cognition? A small dataset with a big trap -- and a question about what tests actually capture.",
         priority="confounding vs causation, and the patient's own account",
         priority_why="Women who used estrogen are also, on average, healthier, wealthier, and better-educated. "
                      "So a raw 'estrogen users score better' can be entirely those other things. And a "
                      "digit-symbol test never measures the 'brain fog' a patient actually reports.",
         group_desc="estrogen users vs non-users",
         confounding=True, confound_feature="used_estrogen",
         confound_controls=("age", "education", "income_ratio"),
         levelup="- After controlling for age/education/income, how much of the 'estrogen helps cognition' "
                 "effect is left? Write the honest one-sentence conclusion.\n"
                 "- The outcome is an *objective* digit-symbol test. Brain fog is *subjective*. Argue why the "
                 "gap between the two is itself the story a patient would tell.\n"
                 "- Reflect: how does it change care when a woman's symptoms are dismissed because a test "
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
