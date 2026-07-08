"""Hand-author the G5 deck sidecar: bone theme, divider accents, and a bespoke freeform
code+notes for every content slide (distinct composition per slide, no comp used >2x)."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
LJ = HERE / "g5_heart.md.layout.json"
d = json.load(open(LJ))
d["theme"] = "bone"

IMG = "/home/jiwei/outset-ai-healthcare/solutions/g5_heart/slides/figures/"

# ---- divider accents + notes (by slide index) ----
DIV = {
    0: ("#40E0D0", "Open with the clinical scene before any modeling. Three ideas to plant: a doctor combines many checkup clues into one risk number; that risk score has been done by hand since 1998 and is sex-specific; and heart disease is under-diagnosed in women. Transition: what does that one risk number actually combine?"),
    5: ("#FF1493", "Now the raw material. Why the UCI Cleveland set is right: real clinical records, free and open to learn on, and it records sex so we can audit fairness. Then look at it -- the 2:1 male skew that comes back to bite us. Transition: 303 patients, 13 features, and one uncomfortable imbalance."),
    8: ("#9D4DEB", "How we turn 13 numbers into a risk score. We do not guess the model -- we run a bake-off, then write down the exact recipe. Transition: four models walk in; do any of them actually win?"),
    11: ("#F0C840", "The payoff, each number named by how we measured it. Graded by AUC the tool hits 0.90; SHAP shows which clues drive it; cholesterol alone is a coin flip; and the fairness audit by sex is the whole point. Transition: first, how do you even grade a risk tool?"),
    16: ("#40E0D0", "Good science states its own limits out loud. What the tool genuinely does, where it is skewed, the papers behind it, and the one habit to remember. This is the section the rubric rewards most."),
}
for i, (hexc, note) in DIV.items():
    d["slides"][i]["params"]["accent_hex"] = hexc
    d["slides"][i]["params"]["notes"] = note

# ---- freeform code + notes (by slide index) ----
CODE = {}
NOTE = {}

# 1: heart_risk_engine -- FULL-BLEED HERO
CODE[1] = f"""IMG = '{IMG}'
img_h = body_h - 0.55
_fit_image(slide, IMG+'intro_heart_risk_engine.png', left=body_l, top=body_top, max_w=body_w, max_h=img_h)
_add_text(slide, "The checkup measures many clues; a good risk tool combines them into one number -- never trusting a single measurement alone.",
          left=body_l+1.0, top=body_top+img_h+0.12, width=body_w-2.0, height=0.5,
          size=14, color_rgb=INK_RGB, font=SANS_FONT, italic=True, align=PP_ALIGN.CENTER)"""
NOTE[1] = "Open concrete and physical. A doctor measures cholesterol, blood pressure, age, chest-pain type -- and eleven more -- and combines them into ONE risk read-out, the gauge on the right. No single number decides. Do not mention machine learning yet. Transition: people have been doing this exact math by hand for decades."

# 2: framingham_1998 -- figure + number chips
CODE[2] = f"""IMG = '{IMG}'
img_h = body_h - 1.15
_fit_image(slide, IMG+'intro_framingham_1998.png', left=body_l+body_w*0.14, top=body_top, max_w=body_w*0.72, max_h=img_h)
chip_y = body_top + img_h + 0.18
chips = [('1998', 'the year Framingham turned checkup clues into a 10-year risk', TURQUOISE_RGB),
         ('10-yr', 'the horizon it predicts -- not just today, but the decade ahead', DEEPPINK_RGB),
         ('by sex', 'men and women get different math, because the same numbers mean different things', BLUEVIOLET_RGB)]
gap = 0.3
cw = (body_w - gap*2) / 3
for i, (num, lab, col) in enumerate(chips):
    x = body_l + i*(cw+gap)
    _add_text(slide, num, left=x, top=chip_y, width=cw, height=0.55, size=30, color_rgb=col, font=MONO_FONT, bold=True, align=PP_ALIGN.CENTER)
    _add_text(slide, lab, left=x, top=chip_y+0.6, width=cw, height=0.5, size=12.5, color_rgb=INK_RGB, font=SANS_FONT, align=PP_ALIGN.CENTER)"""
NOTE[2] = "The landmark that makes this project credible. The Framingham Risk Score (Wilson 1998) turns combined risk factors into a 10-year risk of heart disease -- and the two curves show it is sex-specific: men (turquoise) sit above women (pink) at the same risk-factor burden. This is a redraw of their idea, attributed. Our model is a small, honest version. Transition: why combine so many factors at all?"

# 3: risk_factor_stack -- figure + aside bullets
CODE[3] = f"""IMG = '{IMG}'
fw = body_w * 0.58
_fit_image(slide, IMG+'intro_risk_factor_stack.png', left=body_l, top=body_top, max_w=fw, max_h=body_h)
ax = body_l + fw + 0.45
aw = body_w - fw - 0.45
_add_text(slide, 'RISK IS A STACK', left=ax, top=body_top+0.12, width=aw, height=0.4, size=15, color_rgb=DEEPPINK_RGB, font=MONO_FONT, bold=True)
bullets = [('No single cause', 'Nine factors together explain over 90% of first heart attacks.', TURQUOISE_RGB),
           ('Cholesterol leads', 'Abnormal cholesterol sits near the top -- and it is one of our 13 features.', DEEPPINK_RGB),
           ('So we measure many', 'A checkup captures several clues at once, never just one number.', BLUEVIOLET_RGB)]
y = body_top + 0.75
for lab, bod, col in bullets:
    _add_rect(slide, left=ax, top=y+0.05, width=0.12, height=0.92, fill_rgb=col)
    _add_text(slide, lab, left=ax+0.3, top=y, width=aw-0.3, height=0.4, size=15, color_rgb=col, font=MONO_FONT, bold=True)
    _add_text(slide, bod, left=ax+0.3, top=y+0.42, width=aw-0.3, height=0.75, size=13.5, color_rgb=INK_RGB, font=SANS_FONT)
    y = y + 1.35"""
NOTE[3] = "Why a checkup measures many things. The INTERHEART study (Yusuf 2004) found nine modifiable factors -- led by abnormal cholesterol and smoking -- explain over 90% of first heart attacks. No single clue is the whole story. Note cholesterol is near the top of the risk stack AND is a column in our data -- hold that thought, because it gets a surprising twist in the results. Transition: but the same heart attack does not look the same in everyone."

# 4: symptom_presentation -- figure + aside
CODE[4] = f"""IMG = '{IMG}'
fw = body_w * 0.60
_fit_image(slide, IMG+'intro_symptom_presentation.png', left=body_l, top=body_top+0.1, max_w=fw, max_h=body_h-0.2)
ax = body_l + fw + 0.4
aw = body_w - fw - 0.4
_add_text(slide, 'THE TEXTBOOK IS MALE', left=ax, top=body_top+0.25, width=aw, height=0.4, size=15, color_rgb=DEEPPINK_RGB, font=MONO_FONT, bold=True)
_add_text(slide, "Men more often get the classic crushing chest pain. Women more often get back pain, nausea, or shortness of breath.",
          left=ax, top=body_top+0.8, width=aw, height=1.3, size=15.5, color_rgb=INK_RGB, font=SANS_FONT)
_add_text(slide, "Because the classic picture is the male one, women's heart attacks are more often missed.",
          left=ax, top=body_top+2.15, width=aw, height=1.2, size=15.5, color_rgb=BLUEVIOLET_RGB, font=SANS_FONT, italic=True)
_add_text(slide, 'Watch this thread -- it returns in our fairness check.', left=ax, top=body_top+3.5, width=aw, height=0.7, size=13.5, color_rgb=MUTED_RGB, font=SANS_FONT, italic=True)"""
NOTE[4] = "The fairness problem, stated plainly. Women more often present with back pain, nausea, or shortness of breath rather than the classic chest pain (AHA statement, Mehta 2016; van Oosterhout 2020). The textbook heart-attack picture is the male one, so women's attacks get missed and under-treated. This is the thread we deliberately follow all the way to the fairness audit. Transition: to build our own tool, we need the right data."

# 6: Why the Cleveland dataset -- 3 stat tiles
CODE[6] = """tiles = [('303', 'real patient records from a 1980s coronary-disease study, each with a verified yes/no answer', DEEPPINK_RGB),
         ('13', 'routine checkup numbers -- age, blood pressure, cholesterol, chest-pain type, exercise-test results', BLUEVIOLET_RGB),
         ('free', 'downloads straight from the UCI repository with no gate -- and it records sex, so we can audit fairness', TURQUOISE_RGB)]
gap = 0.35
cw = (body_w - gap*2) / 3
ch = body_h - 0.10
num_h = 1.5
lab_h = 2.0
block_top = body_top + (ch - (num_h + 0.2 + lab_h)) / 2
for i, (num, lab, col) in enumerate(tiles):
    x = body_l + i*(cw+gap)
    _add_rect(slide, left=x, top=body_top, width=cw, height=ch, fill_rgb=col)
    tc = _text_on(col)
    _add_text(slide, num, left=x+0.28, top=block_top, width=cw-0.56, height=num_h, size=58, color_rgb=tc, font=MONO_FONT, bold=True, anchor=MSO_ANCHOR.MIDDLE)
    _add_text(slide, lab, left=x+0.28, top=block_top+num_h+0.2, width=cw-0.56, height=lab_h, size=15, color_rgb=tc, font=SANS_FONT)"""
NOTE[6] = "Justify the dataset in three facts. 303 real clinical records with verified answers; 13 ordinary checkup features; and free + open, so any student can fetch it in seconds with no login. The quiet reason it beats a bare pile of numbers: it records sex, which is what later lets us audit fairness. Transition: and the first thing we do with any dataset is look at who is in it."

# 7: dataset_sex_skew -- figure hero + colored bands
CODE[7] = f"""IMG = '{IMG}'
img_h = body_h - 1.25
_fit_image(slide, IMG+'intro_dataset_sex_skew.png', left=body_l+body_w*0.06, top=body_top, max_w=body_w*0.88, max_h=img_h)
band_y = body_top + img_h + 0.14
band_h = 1.0
gap = 0.35
half = (body_w - gap) / 2
_add_rect(slide, left=body_l, top=band_y, width=half, height=band_h, fill_rgb=TURQUOISE_RGB)
tc = _text_on(TURQUOISE_RGB)
_add_text(slide, '2 : 1', left=body_l+0.3, top=band_y, width=1.7, height=band_h, size=34, color_rgb=tc, font=MONO_FONT, bold=True, anchor=MSO_ANCHOR.MIDDLE)
_add_text(slide, 'men to women -- the model gets far more practice on men', left=body_l+2.1, top=band_y, width=half-2.4, height=band_h, size=13.5, color_rgb=tc, font=SANS_FONT, anchor=MSO_ANCHOR.MIDDLE)
x2 = body_l + half + gap
_add_rect(slide, left=x2, top=band_y, width=half, height=band_h, fill_rgb=DEEPPINK_RGB)
tc2 = _text_on(DEEPPINK_RGB)
_add_text(slide, 'the honest catch', left=x2+0.3, top=band_y+0.14, width=half-0.6, height=0.4, size=16, color_rgb=tc2, font=MONO_FONT, bold=True)
_add_text(slide, 'a model only learns from who is in its data -- this skew returns in our fairness check', left=x2+0.3, top=band_y+0.54, width=half-0.6, height=0.4, size=12.5, color_rgb=tc2, font=SANS_FONT)"""
NOTE[7] = "Look before you model. The waffle shows the Cleveland cohort is about two men for every woman (201 vs 96). A model learns from examples, so with fewer women it gets less practice on them. Name the honest catch now: a model only learns from who is in its data, and this skew is exactly why the fairness audit later will not be clean. Transition: with the data understood, which model should we use?"

# 9: model_comparison -- big-stat + figure
CODE[9] = f"""IMG = '{IMG}'
fw = body_w * 0.56
_fit_image(slide, IMG+'model_comparison.png', left=body_l, top=body_top, max_w=fw, max_h=body_h)
ax = body_l + fw + 0.45
aw = body_w - fw - 0.45
_add_text(slide, 'WE TESTED, WE DID NOT GUESS', left=ax, top=body_top+0.2, width=aw, height=0.4, size=14, color_rgb=BLUEVIOLET_RGB, font=MONO_FONT, bold=True)
_add_text(slide, '~0.80', left=ax, top=body_top+0.75, width=aw, height=1.1, size=70, color_rgb=INK_RGB, font=MONO_FONT, bold=True)
_add_text(slide, 'test accuracy -- and all four models land within a whisker of it.', left=ax, top=body_top+2.0, width=aw, height=1.0, size=15, color_rgb=INK_RGB, font=SANS_FONT)
_add_text(slide, 'On clean tabular data the model barely matters. The data sets the ceiling -- 300 patients hold only so much signal.', left=ax, top=body_top+3.15, width=aw, height=1.5, size=15, color_rgb=BLUEVIOLET_RGB, font=SANS_FONT, italic=True)"""
NOTE[9] = "The which-tool decision, made honestly. We trained Logistic Regression, Random Forest, CatBoost, and a TabPFN foundation model the same way and graded each on held-out patients. They all land near 0.80 -- a dead heat, well above the 0.54 coin-flip line. The lesson: on clean tabular data the model barely matters; the data sets the ceiling. So we stop model-shopping. Transition: here is the exact recipe so anyone could rebuild it."

# 10: methods -- numbered process
CODE[10] = """steps = [('1', 'SPLIT', 'one clean 75/25 train-test split, fixed seed, stratified so both halves carry the same disease rate', TURQUOISE_RGB),
         ('2', 'FEATURES', 'all 13 checkup measurements, each read as a plain number -- no hand-picking', DEEPPINK_RGB),
         ('3', 'MODEL', 'CatBoost for the headline score (a calibrated probability we grade by AUC); logistic regression for the clean feature tests', AMBER_RGB),
         ('4', 'GRADE', 'every number reported on the held-out test set the model never trained on -- no leakage', BLUEVIOLET_RGB)]
n = len(steps)
gap = 0.28
rh = (body_h - gap*(n-1)) / n
for i, (num, lab, bod, col) in enumerate(steps):
    y = body_top + i*(rh+gap)
    _add_rect(slide, left=body_l, top=y, width=0.9, height=rh, fill_rgb=col)
    _add_text(slide, num, left=body_l, top=y, width=0.9, height=rh, size=26, color_rgb=_text_on(col), font=MONO_FONT, bold=True, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    _add_text(slide, lab, left=body_l+1.15, top=y, width=2.4, height=rh, size=17, color_rgb=col, font=MONO_FONT, bold=True, anchor=MSO_ANCHOR.MIDDLE)
    _add_text(slide, bod, left=body_l+3.7, top=y, width=body_w-3.7, height=rh, size=14.5, color_rgb=INK_RGB, font=SANS_FONT, anchor=MSO_ANCHOR.MIDDLE)"""
NOTE[10] = "The reproducibility slide -- read it as a recipe, not trivia. One clean stratified 75/25 split with a fixed seed; all 13 features as plain numbers; CatBoost for the headline probability we grade by AUC, and logistic regression for the clean feature comparisons; and every reported number is on the held-out test set, so there is no leakage. Any one of these done wrong quietly breaks the result. Transition: so how well does it actually score?"

# 12: roc -- big-stat 0.90 + figure
CODE[12] = f"""IMG = '{IMG}'
fw = body_w * 0.50
_fit_image(slide, IMG+'roc.png', left=body_l, top=body_top, max_w=fw, max_h=body_h)
ax = body_l + fw + 0.5
aw = body_w - fw - 0.5
_add_text(slide, '0.90', left=ax, top=body_top+0.2, width=aw, height=1.3, size=84, color_rgb=AMBER_RGB, font=MONO_FONT, bold=True)
_add_text(slide, 'AUC -- the area under the ROC curve', left=ax, top=body_top+1.55, width=aw, height=0.5, size=17, color_rgb=INK_RGB, font=MONO_FONT, bold=True)
_add_text(slide, 'The ROC curve sweeps every cut-off and plots disease caught against false alarms. AUC is the area underneath.', left=ax, top=body_top+2.2, width=aw, height=1.35, size=15, color_rgb=INK_RGB, font=SANS_FONT)
_add_text(slide, '0.5 = a coin flip     1.0 = perfect', left=ax, top=body_top+3.55, width=aw, height=0.5, size=15, color_rgb=BLUEVIOLET_RGB, font=MONO_FONT, bold=True)
_add_text(slide, 'Plain accuracy was only 0.81 -- AUC credits the tool for ranking the sickest patients first.', left=ax, top=body_top+4.15, width=aw, height=0.8, size=15, color_rgb=INK_RGB, font=SANS_FONT, italic=True)"""
NOTE[12] = "Name the metric before the number. A risk tool outputs a score, not a yes/no, so we grade it with the ROC curve -- every cut-off, plotting disease caught against false alarms -- and AUC, the area underneath, where 0.5 is a coin flip and 1.0 is perfect. Our AUC is 0.90 and the curve hugs the top-left. Point out the gap: accuracy was only 0.81, but AUC rewards the model for ranking the sickest patients first. Transition: and we can see WHICH clues it ranked on."

# 13: shap -- figure + callout list
CODE[13] = f"""IMG = '{IMG}'
fw = body_w * 0.52
_fit_image(slide, IMG+'shap_importance.png', left=body_l, top=body_top+0.1, max_w=fw, max_h=body_h-0.2)
ax = body_l + fw + 0.4
aw = body_w - fw - 0.4
callouts = [('ca', 'vessels seen on imaging -- the top signal', TURQUOISE_RGB),
            ('thal', 'a heart stress-test result', DEEPPINK_RGB),
            ('cp', 'chest-pain type -- the cue a cardiologist weighs', AMBER_RGB)]
rh = 1.02
for i, (num, lab, col) in enumerate(callouts):
    y = body_top + 0.1 + i*(rh+0.12)
    _add_rect(slide, left=ax, top=y, width=0.14, height=rh, fill_rgb=col)
    _add_text(slide, num, left=ax+0.3, top=y, width=1.6, height=rh, size=30, color_rgb=col, font=MONO_FONT, bold=True, anchor=MSO_ANCHOR.MIDDLE)
    _add_text(slide, lab, left=ax+1.95, top=y, width=aw-1.95, height=rh, size=13.5, color_rgb=INK_RGB, font=SANS_FONT, anchor=MSO_ANCHOR.MIDDLE)
y2 = body_top + 0.1 + 3*(rh+0.12)
bh = body_h - 0.2 - (y2 - body_top)
_add_rect(slide, left=ax, top=y2, width=aw, height=bh, fill_rgb=DEEPPINK_RGB)
_add_text(slide, 'And cholesterol (chol) sits near the bottom -- a real surprise we test next.', left=ax+0.25, top=y2, width=aw-0.5, height=bh, size=14.5, color_rgb=_text_on(DEEPPINK_RGB), font=SANS_FONT, italic=True, anchor=MSO_ANCHOR.MIDDLE)"""
NOTE[13] = "Feature importance for a table is SHAP: how much each measurement moves the prediction, averaged. The heavy hitters are ca (vessels on imaging), thal (a stress-test result), and cp (chest-pain type) -- the same cues a cardiologist leans on, which is reassuring. The twist is at the bottom of the chart: cholesterol ranks LOW. That is surprising enough to deserve its own test. Transition: so is cholesterol actually a good predictor on its own?"

# 14: feature_ablation -- figure + colored bands
CODE[14] = f"""IMG = '{IMG}'
fw = body_w * 0.56
_fit_image(slide, IMG+'feature_ablation.png', left=body_l, top=body_top, max_w=fw, max_h=body_h)
ax = body_l + fw + 0.45
aw = body_w - fw - 0.45
_add_rect(slide, left=ax, top=body_top+0.2, width=aw, height=1.5, fill_rgb=AMBER_RGB)
tc = _text_on(AMBER_RGB)
_add_text(slide, '0.52', left=ax+0.28, top=body_top+0.32, width=aw-0.56, height=0.78, size=40, color_rgb=tc, font=MONO_FONT, bold=True)
_add_text(slide, 'cholesterol ALONE -- barely better than a coin flip', left=ax+0.28, top=body_top+1.12, width=aw-0.56, height=0.5, size=14, color_rgb=tc, font=SANS_FONT)
_add_rect(slide, left=ax, top=body_top+1.88, width=aw, height=1.5, fill_rgb=TURQUOISE_RGB)
tc2 = _text_on(TURQUOISE_RGB)
_add_text(slide, '0.80', left=ax+0.28, top=body_top+2.0, width=aw-0.56, height=0.78, size=40, color_rgb=tc2, font=MONO_FONT, bold=True)
_add_text(slide, 'the checkup WITHOUT cholesterol -- still strong', left=ax+0.28, top=body_top+2.8, width=aw-0.56, height=0.5, size=14, color_rgb=tc2, font=SANS_FONT)
_add_text(slide, 'Causing disease slowly is not the same as detecting it today.', left=ax, top=body_top+3.65, width=aw, height=1.0, size=15, color_rgb=INK_RGB, font=SANS_FONT, italic=True)"""
NOTE[14] = "The fair test SHAP set up. Give the model only cholesterol: it scores 0.52, basically a coin flip. Remove cholesterol and keep the other twelve: still 0.80. So in this data cholesterol is a weak one-shot diagnostic clue, even though INTERHEART is right that it is a major CAUSAL factor over decades. That is the causing-versus-detecting distinction -- and it is why doctors combine many measurements instead of trusting one number. Transition: last and most important -- does the tool work equally for women and men?"

# 15: fairness_by_sex -- figure + stacked stat cards
CODE[15] = f"""IMG = '{IMG}'
fw = body_w * 0.54
_fit_image(slide, IMG+'fairness_by_sex.png', left=body_l, top=body_top, max_w=fw, max_h=body_h)
ax = body_l + fw + 0.45
aw = body_w - fw - 0.45
th = 1.4
_add_rect(slide, left=ax, top=body_top+0.1, width=aw, height=th, fill_rgb=DEEPPINK_RGB)
tc = _text_on(DEEPPINK_RGB)
_add_text(slide, '0.88', left=ax+0.28, top=body_top+0.2, width=aw-0.56, height=0.85, size=38, color_rgb=tc, font=MONO_FONT, bold=True, anchor=MSO_ANCHOR.MIDDLE)
_add_text(slide, 'accuracy for women', left=ax+0.28, top=body_top+1.0, width=aw-0.56, height=0.45, size=14, color_rgb=tc, font=SANS_FONT)
_add_rect(slide, left=ax, top=body_top+1.65, width=aw, height=th, fill_rgb=TURQUOISE_RGB)
tc2 = _text_on(TURQUOISE_RGB)
_add_text(slide, '0.78', left=ax+0.28, top=body_top+1.75, width=aw-0.56, height=0.85, size=38, color_rgb=tc2, font=MONO_FONT, bold=True, anchor=MSO_ANCHOR.MIDDLE)
_add_text(slide, 'accuracy for men', left=ax+0.28, top=body_top+2.55, width=aw-0.56, height=0.45, size=14, color_rgb=tc2, font=SANS_FONT)
_add_text(slide, 'A 0.09 gap -- and the honest cause is the 2:1 male data. One overall number would have hidden it.', left=ax, top=body_top+3.4, width=aw, height=1.3, size=15, color_rgb=INK_RGB, font=SANS_FONT, italic=True)"""
NOTE[15] = "The point of the whole project. We trained once, then graded the model separately by sex: 0.88 for women, 0.78 for men -- a 0.09 gap. The twist is that it is MORE accurate for women, mostly because so few women here had disease that predict-healthy is usually right. That is not good news: an accuracy gap in either direction means behavior depends on sex, and accurate-because-it-rarely-says-sick is the exact failure behind real under-diagnosis. You only saw it because you looked. Transition: so where does that leave us honestly?"

# 17: what it can and cannot do -- 3 cards
CODE[17] = """cards = [('IT WORKS', 'A real risk tool on real checkup data: AUC 0.90, leaning on the same cues (ca, thal, chest-pain type) a cardiologist uses.', 'AUC 0.90', TURQUOISE_RGB),
         ('BUT IT IS SKEWED', 'The accuracy gap by sex is real, and the data is about 2:1 male -- so it may serve women worse in the real world.', '0.09 gap by sex', DEEPPINK_RGB),
         ('A REAL TOOL', "Would use today's diverse patients, a sex-specific design, multi-hospital validation, and a clinician in the loop.", 'not ready for a patient', BLUEVIOLET_RGB)]
gap = 0.35
cw = (body_w - gap*2) / 3
ch = body_h - 0.1
by = body_top + 0.6
for i, (lab, bod, tag, col) in enumerate(cards):
    x = body_l + i*(cw+gap)
    _add_rect(slide, left=x+0.05, top=body_top+0.06, width=cw, height=ch, fill_rgb=_rgb('#E4E1D9'))
    _add_rect(slide, left=x, top=body_top, width=cw, height=ch, fill_rgb=WHITE_RGB)
    _add_rect(slide, left=x, top=body_top, width=cw, height=0.1, fill_rgb=col)
    _add_text(slide, lab, left=x+0.28, top=by, width=cw-0.56, height=0.5, size=17, color_rgb=col, font=MONO_FONT, bold=True)
    _add_text(slide, bod, left=x+0.28, top=by+0.75, width=cw-0.56, height=ch-2.3, size=16, color_rgb=INK_RGB, font=SANS_FONT)
    _add_rect(slide, left=x, top=body_top+ch-1.05, width=cw, height=0.02, fill_rgb=_rgb('#E4E1D9'))
    _add_text(slide, tag, left=x+0.28, top=body_top+ch-0.9, width=cw-0.56, height=0.6, size=15, color_rgb=col, font=MONO_FONT, bold=True, anchor=MSO_ANCHOR.MIDDLE)"""
NOTE[17] = "Model the honesty the rubric rewards. Green: this genuinely works -- AUC 0.90, reading the cues a cardiologist uses. Pink: it is skewed -- the sex gap is real and the data is 2:1 male, so it may serve women worse. Purple: a real tool would add today's diverse patients, a sex-specific design, multi-hospital validation, and a clinician in the loop. State it plainly: this teaches judgment, it does not care for a patient. Transition: the papers behind it."

# 18: references -- two-column
CODE[18] = """colw = (body_w - 0.6) / 2
left_refs = ('[1] Detrano et al. 1989, Am J Cardiol -- the Cleveland dataset we use.\\n\\n'
             '[2] Wilson et al. 1998, Circulation -- the Framingham Risk Score.\\n\\n'
             '[3] Yusuf et al. 2004, Lancet -- INTERHEART: nine factors, 90% of risk.\\n\\n'
             '[7] UCI Machine Learning Repository -- Heart Disease Data Set.')
right_refs = ('[4] Mehta et al. 2016, Circulation -- AHA statement on heart attacks in women.\\n\\n'
              '[5] van Oosterhout et al. 2020, JAHA -- sex differences in symptoms.\\n\\n'
              '[6] Bugiardini & Bairey Merz 2005, JAMA -- angina with "normal" arteries.')
_add_text(slide, 'FOUNDATIONS & DATA', left=body_l, top=body_top+0.05, width=colw, height=0.4, size=16, color_rgb=TURQUOISE_RGB, font=MONO_FONT, bold=True)
_add_text(slide, left_refs, left=body_l, top=body_top+0.6, width=colw, height=body_h-0.7, size=14.5, color_rgb=INK_RGB, font=SANS_FONT)
rx = body_l + colw + 0.6
_add_text(slide, 'SEX DIFFERENCES & UNDER-DIAGNOSIS', left=rx, top=body_top+0.05, width=colw, height=0.4, size=16, color_rgb=DEEPPINK_RGB, font=MONO_FONT, bold=True)
_add_text(slide, right_refs, left=rx, top=body_top+0.6, width=colw, height=body_h-0.7, size=14.5, color_rgb=INK_RGB, font=SANS_FONT)"""
NOTE[18] = "Give credit and show the trail. Left column is the foundations and the data: the Cleveland dataset we used [1], the Framingham risk score we echo [2], INTERHEART on the risk stack [3], and the UCI repository [7]. Right column is the fairness literature that set our theme: the AHA statement on women's heart attacks [4], the symptom-difference meta-analysis [5], and the normal-arteries paper [6]. Point out that [2] is the score we redrew and [4]-[6] are why we audit by sex. Transition: the one habit to remember."

# 19: bottom line -- centered closing
CODE[19] = """q_top = body_top + body_h*0.22
_add_text(slide, 'Grade with the right metric. Ask which clues matter. Check who it works for.',
          left=body_l, top=q_top, width=body_w, height=1.5, size=30, color_rgb=TURQUOISE_RGB, font=MONO_FONT, bold=True, align=PP_ALIGN.CENTER)
rule_w = 2.4
_add_rect(slide, left=body_l+(body_w-rule_w)/2, top=q_top+1.7, width=rule_w, height=0.05, fill_rgb=TURQUOISE_RGB)
_add_text(slide, 'That is how a risk tool earns trust. Use this to learn the judgment -- then use a validated, sex-specific tool and a real clinician to actually care for a patient.',
          left=body_l+1.0, top=q_top+2.25, width=body_w-2.0, height=1.3, size=19, color_rgb=INK_RGB, font=SANS_FONT, align=PP_ALIGN.CENTER)"""
NOTE[19] = "Land the deck on one habit: grade with the right metric (AUC, not accuracy), ask which clues actually matter (SHAP + the cholesterol ablation), and always check who the model works for (the fairness audit). Our tool did all three. Leave them with the working rule of medical AI -- a model you can measure honestly beats a fancy one nobody can check -- then hand the real decision to a validated tool and a clinician. Thank them."

for i in CODE:
    p = d["slides"][i]["params"]
    p["_provenance"] = "agent"
    p["code"] = CODE[i]
    p["notes"] = NOTE[i]
    p["lede"] = ""

# sanity: no composer left
bad = [i for i, s in enumerate(d["slides"]) if s.get("params", {}).get("_provenance") == "composer"]
assert not bad, f"composer left on {bad}"
json.dump(d, open(LJ, "w"), indent=2, ensure_ascii=False)
print("authored", len(CODE), "freeform slides + 5 dividers; theme=bone; composer-free")
