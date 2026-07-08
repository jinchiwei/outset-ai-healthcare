"""Hand-author the G3 freeform sidecar: theme bone, divider accents, and a DISTINCT body
composition per content slide (no template cloned). Loads the generated scaffold, mutates
params in place (matching freeform slides by ORDER so en-dash title conversion can't misalign),
writes back.  Run:  python author_sidecar.py
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
LJ = HERE / "g3_brain_ct.md.layout.json"
FIG = str(HERE / "figures")                       # absolute figure dir; _fit_image takes absolute paths fine

d = json.load(open(LJ))
d["theme"] = "bone"                               # MUST match the figures' canvas or figures show a grey box

# ---- section-divider accents (turquoise -> deeppink -> blueviolet -> amber -> turquoise) ----
DIV = {
    "Background": ("#40E0D0", "Open on the clinical stakes so the modeling lands later. Three beats: a bleed is bright on CT and every minute costs ~1.9M neurons; triage AI reads the dangerous scans first; and the catch that runs through the whole talk -- a dataset can hide who is in it. Transition: meet the scan."),
    "The data": ("#FF1493", "The raw material, and the contrast that IS the lesson. Two datasets: brain-CT images with no patient facts, and a tabular stroke set that records sex, age, and history. Only the second can be audited for fairness. Transition: what a good dataset should have recorded."),
    "The model": ("#8A2BE2", "How we build both halves. For the scans, transfer learning with a CAFormer backbone; a reproducible recipe. For the table, a four-model bake-off, won on held-out AUC, deploying CatBoost because we can explain and audit it. Transition: first, borrow a brain."),
    "Results": ("#F0C840", "The payoff in two acts. Act one: the CT detector works (AUC 0.817), reads mostly the right regions, but hits a wall -- no demographics, so the fairness audit cannot run. Act two: the tabular model records sex, so we audit AND fix a real gap. Transition: how do you grade a detector?"),
    "The takeaway": ("#40E0D0", "Land the one idea the rubric rewards: recording demographics is what makes fairness auditable and fixable. Recap the contrast, credit the papers, and leave them with the rule -- a model you can audit is the real bar."),
}

# ---- per-slide freeform bodies, in slide ORDER (17 content slides) ----
CODES, NOTES = [], []
def add(code, note):
    CODES.append(code); NOTES.append(note)

# S1 -- FULL-BLEED HERO: the head CT scene
add(f"""
img_h = body_h - 0.55
_fit_image(slide, '{FIG}/intro_head_ct.png', left=body_l, top=body_top, max_w=body_w, max_h=img_h)
_add_text(slide, 'A fresh bleed is bright (hyperdense) on CT -- and every minute of delay costs about 1.9 million neurons, so it must be found and read first.',
          left=body_l + 0.8, top=body_top + img_h + 0.12, width=body_w - 1.6, height=0.5,
          size=14, color_rgb=INK_RGB, font=SANS_FONT, italic=True, align=PP_ALIGN.CENTER)
""",
"Open concrete and physical. A head CT: the bright blob is an acute hemorrhage (blood is hyperdense), pointed out in pink. The clock on the right is the stakes -- in a large stroke ~1.9 million neurons die every minute (Saver 2006), so the bleed has to be found and read first. Do not mention datasets yet. Transition: that is exactly the job triage AI does.")

# S2 -- FIGURE LEFT + PROSE ASIDE: triage reorders the queue
add(f"""
fw = body_w * 0.56
_fit_image(slide, '{FIG}/intro_triage_queue.png', left=body_l, top=body_top, max_w=fw, max_h=body_h)
ax = body_l + fw + 0.45
aw = body_w - fw - 0.45
_add_text(slide, 'SORT, DON\\'T JUST QUEUE', left=ax, top=body_top + 0.2, width=aw, height=0.45,
          size=15, color_rgb=DEEPPINK_RGB, font=MONO_FONT, bold=True)
_add_text(slide, 'A radiologist faces a long list of scans. Read in arrival order, a bleed can wait behind ten normal heads.',
          left=ax, top=body_top + 0.85, width=aw, height=1.35, size=16, color_rgb=INK_RGB, font=SANS_FONT)
_add_text(slide, 'Triage AI scores every scan and lifts the likely bleeds to the TOP, so the dangerous ones are read first.',
          left=ax, top=body_top + 2.25, width=aw, height=1.35, size=16, color_rgb=INK_RGB, font=SANS_FONT)
_add_text(slide, 'Real tools do this with AUCs above 0.90 and cut time-to-treatment.',
          left=ax, top=body_top + 3.65, width=aw, height=1.1, size=16, color_rgb=BLUEVIOLET_RGB, font=SANS_FONT, italic=True)
""",
"The concrete benefit before the catch. Left: seven scans in arrival order, two bleeds buried in the middle; right: after AI triage the bleeds jump to the top, flagged 'read first'. This is the real, deployed job -- Chilamkurthy 2018 hit AUCs above 0.90 -- and it saves the minutes that matter. Transition: so these tools work; what's the problem?")

# S3 -- BIG STATEMENT + TWO MINI CARDS: the thesis
add(f"""
_add_text(slide, 'A model can work on average and still hide who it fails.',
          left=body_l, top=body_top + 0.15, width=body_w, height=1.0,
          size=27, color_rgb=TURQUOISE_RGB, font=MONO_FONT, bold=True, align=PP_ALIGN.CENTER)
gap = 0.5
cw = (body_w - gap) / 2
ctop = body_top + 1.55
ch = body_h - 1.75
cards = [('WORKS ON AVERAGE', 'A detector can hit a strong overall AUC and still be much worse for one group of patients -- a hidden failure.', TURQUOISE_RGB),
         ('BUT UN-CHECKABLE', 'If age, sex, race, and scanner were never recorded, there is nothing to group by. The audit that would catch it cannot even run.', DEEPPINK_RGB)]
for i, (lab, body, col) in enumerate(cards):
    x = body_l + i * (cw + gap)
    _add_rect(slide, left=x, top=ctop, width=cw, height=ch, fill_rgb=WHITE_RGB)
    _add_rect(slide, left=x, top=ctop, width=cw, height=0.12, fill_rgb=col)
    _add_text(slide, lab, left=x + 0.32, top=ctop + 0.4, width=cw - 0.64, height=0.5,
              size=19, color_rgb=col, font=MONO_FONT, bold=True)
    _add_text(slide, body, left=x + 0.32, top=ctop + 1.15, width=cw - 0.64, height=ch - 1.5,
              size=16.5, color_rgb=INK_RGB, font=SANS_FONT)
""",
"The spine of the whole talk, stated once. Read the top line slowly: a model can score well on average and still fail a subgroup badly. Left card: that hidden failure is real. Right card: you can only SEE it if the data records the groups -- otherwise there is nothing to group by and the audit cannot run. Everything after this is a demonstration of this one sentence. Transition: so it comes down to what the data records.")

# S4 -- TWO-PANEL COMPARE: the two datasets (the data-contrast centerpiece)
add(f"""
gap = 0.5
cw = (body_w - gap) / 2
ch = body_h - 0.1
pairs = [('BRAIN CT', 'normal vs. stroke', 'Real head-CT slices, one label each. That is ALL the dataset records -- no age, sex, race, or scanner.', 'you can train it -- but not audit who it fails', DEEPPINK_RGB),
         ('TABULAR STROKE', '5,109 patients', 'One row per patient, and it records sex, age, hypertension, glucose, and smoking -- the facts an audit needs.', 'you can audit AND fix it', TURQUOISE_RGB)]
for i, (lab, sub, body, tag, col) in enumerate(pairs):
    x = body_l + i * (cw + gap)
    _add_rect(slide, left=x, top=body_top, width=cw, height=ch, fill_rgb=WHITE_RGB)
    _add_rect(slide, left=x, top=body_top, width=0.14, height=ch, fill_rgb=col)
    _add_text(slide, lab, left=x + 0.4, top=body_top + 0.35, width=cw - 0.7, height=0.5,
              size=21, color_rgb=col, font=MONO_FONT, bold=True)
    _add_text(slide, sub, left=x + 0.4, top=body_top + 0.95, width=cw - 0.7, height=0.45,
              size=15, color_rgb=MUTED_RGB, font=MONO_FONT)
    _add_text(slide, body, left=x + 0.4, top=body_top + 1.7, width=cw - 0.7, height=ch - 3.0,
              size=17, color_rgb=INK_RGB, font=SANS_FONT)
    _add_rect(slide, left=x + 0.4, top=body_top + ch - 1.05, width=cw - 0.8, height=0.02, fill_rgb=_rgb('#E4E1D9'))
    _add_text(slide, tag, left=x + 0.4, top=body_top + ch - 0.9, width=cw - 0.7, height=0.6,
              size=15, color_rgb=col, font=MONO_FONT, bold=True, anchor=MSO_ANCHOR.MIDDLE)
""",
"The centerpiece of the data section. Two datasets, side by side. Left (pink): the brain-CT set -- images and a label, nothing else, so you can train but not audit. Right (turquoise): the tabular stroke set -- one row per patient WITH sex, age, and history, so you can audit and fix. Same disease, opposite documentation. The bottom tags say it plainly. Transition: what a proper dataset SHOULD have recorded.")

# S5 -- FIGURE RIGHT + BULLETS LEFT (mirror of S2): the datasheet
add(f"""
fw = body_w * 0.56
_fit_image(slide, '{FIG}/intro_datasheet_checklist.png', left=body_l + body_w - fw, top=body_top + 0.3, max_w=fw, max_h=body_h - 0.4)
aw = body_w - fw - 0.5
bullets = [('SIX QUESTIONS', 'A datasheet asks who is in the data -- age, sex, race, scanner, label source, consent -- so you know its limits before you trust it.', TURQUOISE_RGB),
           ('TWO ANSWERS', 'Our brain-CT set answers only two: the image, and a stroke / normal label. The four demographic rows are simply blank.', DEEPPINK_RGB)]
y = body_top + 0.35
for lab, body, col in bullets:
    _add_rect(slide, left=body_l, top=y, width=0.12, height=1.7, fill_rgb=col)
    _add_text(slide, lab, left=body_l + 0.3, top=y, width=aw - 0.3, height=0.45,
              size=16, color_rgb=col, font=MONO_FONT, bold=True)
    _add_text(slide, body, left=body_l + 0.3, top=y + 0.55, width=aw - 0.3, height=1.5,
              size=16, color_rgb=INK_RGB, font=SANS_FONT)
    y += 2.35
""",
"The Datasheets-for-Datasets idea (Gebru 2021): every dataset should ship a standard record of who is in it, so users know its makeup before trusting a model built on it. Left: the six questions a datasheet asks. Right figure: our set checks only two boxes -- image and label -- the four demographic rows are empty. That emptiness is what disarms the fairness audit later. Transition: now the model itself.")

# S6 -- THREE-STAGE HORIZONTAL PIPELINE: transfer learning
add(f"""
stages = [('PRETRAINED', 'a CAFormer already trained on millions of everyday photos', 'it knows edges, texture, shape', TURQUOISE_RGB),
          ('FINE-TUNE', 'freeze that body; train only a small new head on our 224px CT slices', 'it learns the look of a bleed', DEEPPINK_RGB),
          ('DETECT', 'ask the fine-tuned network for one call on a new scan', 'normal, or stroke', BLUEVIOLET_RGB)]
gap = 1.0
cw = (body_w - gap * 2) / 3
ch = body_h - 1.1
ctop = body_top + 0.1
for i, (lab, body, det, col) in enumerate(stages):
    x = body_l + i * (cw + gap)
    _add_rect(slide, left=x, top=ctop, width=cw, height=ch, fill_rgb=_rgb('#EDEAE1'))
    _add_rect(slide, left=x, top=ctop, width=cw, height=0.12, fill_rgb=col)
    _add_text(slide, lab, left=x + 0.28, top=ctop + 0.45, width=cw - 0.56, height=0.5,
              size=19, color_rgb=col, font=MONO_FONT, bold=True)
    _add_text(slide, body, left=x + 0.28, top=ctop + 1.25, width=cw - 0.56, height=ch - 2.6,
              size=15.5, color_rgb=INK_RGB, font=SANS_FONT)
    _add_text(slide, det, left=x + 0.28, top=ctop + ch - 0.8, width=cw - 0.56, height=0.6,
              size=14, color_rgb=col, font=MONO_FONT, bold=True, anchor=MSO_ANCHOR.MIDDLE)
    if i < 2:
        _add_text(slide, '\\u2192', left=x + cw + 0.05, top=ctop + ch / 2 - 0.45, width=gap - 0.1, height=0.9,
                  size=40, color_rgb=INK_RGB, font=MONO_FONT, bold=True, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
_add_text(slide, 'Borrowed knowledge is why a network can learn stroke from a few hundred scans, not a few million.',
          left=body_l, top=ctop + ch + 0.18, width=body_w, height=0.55, size=15,
          color_rgb=BLUEVIOLET_RGB, font=SANS_FONT, italic=True, align=PP_ALIGN.CENTER)
""",
"The one modeling idea for the images. We have hundreds of scans, not millions, so we borrow a brain: take a CAFormer pretrained on everyday photos, freeze its sense of edges and texture, and train only a small new head on our CT slices. Walk the three stages left to right. Same transfer-learning trick as Day 1. Transition: here is the exact recipe.")

# S7 -- NUMBERED PROCESS ROWS: the CT recipe
add(f"""
steps = [('1', 'MODEL', 'caformer_s18, pretrained, backbone FROZEN, a fresh 2-class head trained on top', TURQUOISE_RGB),
         ('2', 'RESIZE', 'each CT slice decoded to 3-channel color and resized to 224 x 224 pixels', DEEPPINK_RGB),
         ('3', 'NORMALIZE', 'pixels rescaled to the mean/scale the backbone was trained on; light flips augment TRAIN only', AMBER_RGB),
         ('4', 'BALANCE', 'stroke vs. normal framed as one binary call, graded on sensitivity, not just accuracy', BLUEVIOLET_RGB),
         ('5', 'SPLIT', 'separate train / val / test; every number we report is on the held-out test set', TURQUOISE_RGB)]
n = len(steps)
gap = 0.25
rh = (body_h - gap * (n - 1)) / n
for i, (num, lab, body, col) in enumerate(steps):
    y = body_top + i * (rh + gap)
    _add_rect(slide, left=body_l, top=y, width=0.9, height=rh, fill_rgb=col)
    _add_text(slide, num, left=body_l, top=y, width=0.9, height=rh,
              size=26, color_rgb=_text_on(col), font=MONO_FONT, bold=True, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    _add_text(slide, lab, left=body_l + 1.15, top=y, width=2.6, height=rh,
              size=17, color_rgb=col, font=MONO_FONT, bold=True, anchor=MSO_ANCHOR.MIDDLE)
    _add_text(slide, body, left=body_l + 3.9, top=y, width=body_w - 3.9, height=rh,
              size=15, color_rgb=INK_RGB, font=SANS_FONT, anchor=MSO_ANCHOR.MIDDLE)
""",
"The reproducibility slide -- read it as a recipe, not trivia. Five steps: (1) a frozen CAFormer with a fresh 2-class head; (2) every slice resized to 224x224; (3) normalized to the backbone's range, train-only augmentation; (4) framed as one stroke/normal call, graded on sensitivity; (5) a clean train/val/test split, all reporting on held-out data. Any one done wrong quietly breaks the result. Transition: the images were the easy half -- the table needs a model choice.")

# S8 -- FIGURE LEFT + CARD ASIDE: the tabular bake-off
add(f"""
fw = body_w * 0.56
_fit_image(slide, '{FIG}/stroke_model_choice.png', left=body_l, top=body_top, max_w=fw, max_h=body_h)
ax = body_l + fw + 0.45
aw = body_w - fw - 0.45
_add_rect(slide, left=ax, top=body_top + 0.15, width=aw, height=body_h - 0.35, fill_rgb=WHITE_RGB)
_add_rect(slide, left=ax, top=body_top + 0.15, width=aw, height=0.1, fill_rgb=TURQUOISE_RGB)
_add_text(slide, 'WE TESTED, WE DID NOT GUESS', left=ax + 0.3, top=body_top + 0.45, width=aw - 0.6, height=0.4,
          size=14, color_rgb=TURQUOISE_RGB, font=MONO_FONT, bold=True)
_add_text(slide, '0.81', left=ax + 0.3, top=body_top + 1.0, width=aw - 0.6, height=0.9,
          size=52, color_rgb=INK_RGB, font=MONO_FONT, bold=True)
_add_text(slide, 'CatBoost AUC. Four models all cleared 0.80, so the honest choice is not the top decimal.',
          left=ax + 0.3, top=body_top + 2.0, width=aw - 0.6, height=1.3, size=15.5, color_rgb=INK_RGB, font=SANS_FONT)
_add_text(slide, 'We deploy CatBoost because it is a tree we can explain with SHAP and audit by sex.',
          left=ax + 0.3, top=body_top + 3.35, width=aw - 0.6, height=1.3, size=15.5, color_rgb=BLUEVIOLET_RGB, font=SANS_FONT, italic=True)
""",
"The 'which model?' decision, made honestly. We ran a bake-off: Logistic Regression, Random Forest, CatBoost, TabPFN, judged by held-out AUC. All four cleared the 0.80 line -- a tight race -- so we did not chase the top decimal. We deploy CatBoost (0.81) because it is a gradient-boosted TREE, which means we can read it with SHAP and audit it by sex, both of which we do next. The teachable habit: pick for the job, not the leaderboard. Transition: on to results.")

# S9 -- FIGURE RIGHT + BIG-STAT LEFT: the CT ROC
add(f"""
fw = body_w * 0.50
_fit_image(slide, '{FIG}/roc.png', left=body_l + body_w - fw, top=body_top, max_w=fw, max_h=body_h)
aw = body_w - fw - 0.5
_add_text(slide, '0.817', left=body_l, top=body_top + 0.2, width=aw, height=1.3,
          size=84, color_rgb=AMBER_RGB, font=MONO_FONT, bold=True)
_add_text(slide, 'AUC -- the area under the ROC curve', left=body_l, top=body_top + 1.55, width=aw, height=0.5,
          size=17, color_rgb=INK_RGB, font=MONO_FONT, bold=True)
_add_text(slide, 'The ROC curve sweeps every cut-off and plots strokes caught against false alarms. AUC is the area underneath.',
          left=body_l, top=body_top + 2.2, width=aw, height=1.4, size=15.5, color_rgb=INK_RGB, font=SANS_FONT)
_add_text(slide, '0.5 = a coin flip     1.0 = perfect', left=body_l, top=body_top + 3.75, width=aw, height=0.5,
          size=15, color_rgb=BLUEVIOLET_RGB, font=MONO_FONT, bold=True)
_add_text(slide, 'The curve hugs the top-left corner -- a genuinely working detector.',
          left=body_l, top=body_top + 4.3, width=aw, height=0.6, size=15, color_rgb=INK_RGB, font=SANS_FONT, italic=True)
""",
"Name the metric before the number. A detector outputs a score, not a yes/no, so we grade it with the ROC curve -- every cut-off, plotting strokes caught against false alarms -- and AUC, the area underneath (0.5 coin flip, 1.0 perfect). We use AUC, not accuracy, because accuracy is fooled by the rare positive. Ours is 0.817 and the curve hugs the top-left: a real, working detector. Transition: but grade the two error types apart.")

# S10 -- FIGURE LEFT + CALLOUT ROWS: read the confusion matrix
add(f"""
fw = body_w * 0.50
_fit_image(slide, '{FIG}/confusion.png', left=body_l, top=body_top + 0.15, max_w=fw, max_h=body_h - 0.3)
ax = body_l + fw + 0.45
aw = body_w - fw - 0.45
rows = [('65%', 'of real STROKES caught (sensitivity) -- most, but not all', DEEPPINK_RGB),
        ('81%', 'of real NORMALS cleared (specificity) -- few false alarms', TURQUOISE_RGB),
        ('the dial', 'for a safety-net triage tool, lower the threshold to catch more strokes, accepting more false alarms', BLUEVIOLET_RGB)]
rh = 1.42
for i, (num, lab, col) in enumerate(rows):
    y = body_top + 0.15 + i * (rh + 0.18)
    _add_rect(slide, left=ax, top=y, width=0.14, height=rh, fill_rgb=col)
    _add_text(slide, num, left=ax + 0.3, top=y, width=1.9, height=rh,
              size=34, color_rgb=col, font=MONO_FONT, bold=True, anchor=MSO_ANCHOR.MIDDLE)
    _add_text(slide, lab, left=ax + 2.3, top=y, width=aw - 2.3, height=rh,
              size=15, color_rgb=INK_RGB, font=SANS_FONT, anchor=MSO_ANCHOR.MIDDLE)
""",
"Read the confusion matrix out loud -- rows are truth, columns are what the model said. Diagonal is correct; off-diagonal is the two error types, and they are NOT equal. Sensitivity ~0.65: it catches about two-thirds of real strokes. Specificity ~0.81: it clears most normals, so false alarms are modest. A missed bleed is the deadly error, so for triage you would lower the threshold to catch more strokes on purpose. One accuracy number would have hidden this split. Transition: is it looking at the right place?")

# S11 -- FULL-BLEED + BOTTOM BAND: Grad-CAM
add(f"""
img_h = body_h - 1.0
_fit_image(slide, '{FIG}/gradcam.png', left=body_l, top=body_top, max_w=body_w, max_h=img_h)
band_y = body_top + img_h + 0.12
_add_rect(slide, left=body_l, top=band_y, width=body_w, height=0.8, fill_rgb=_rgb('#EDEAE1'))
_add_text(slide, 'Grad-CAM = feature importance for an image: some heat sits on brain tissue (good), but some drifts to the SKULL EDGE and border -- a shortcut that works here and would break on a new scanner.',
          left=body_l + 0.3, top=band_y + 0.1, width=body_w - 0.6, height=0.6,
          size=14.5, color_rgb=INK_RGB, font=SANS_FONT, anchor=MSO_ANCHOR.MIDDLE)
""",
"The image version of feature importance. For a table we'd ask SHAP which columns mattered; for a scan we use Grad-CAM, which highlights WHERE the model looked. Top row: scans; bottom row: the heat overlay. Honest read: some heat lands on brain tissue, but some bleeds onto the skull edge and image border -- a classic shortcut. The model is partly reading anatomy and partly cheating off shape cues, which would fail on a different scanner. Worth saying out loud. Transition: and now the wall.")

# S12 -- FIGURE RIGHT + PROSE + DEEPPINK CARD: the un-runnable audit
add(f"""
fw = body_w * 0.48
_fit_image(slide, '{FIG}/missing_metadata.png', left=body_l + body_w - fw, top=body_top + 0.2, max_w=fw, max_h=body_h - 1.4)
aw = body_w - fw - 0.5
_add_text(slide, 'We can audit by CLASS (normal vs. stroke) because the label exists. But the audit that matters -- older vs. younger, women vs. men, one scanner vs. another -- we cannot run at all.',
          left=body_l, top=body_top + 0.2, width=aw, height=2.0, size=16.5, color_rgb=INK_RGB, font=SANS_FONT)
_add_text(slide, 'Not because it is hard: the dataset never recorded those fields, so there is nothing to group by.',
          left=body_l, top=body_top + 2.35, width=aw, height=1.3, size=16.5, color_rgb=INK_RGB, font=SANS_FONT)
cy = body_top + body_h - 1.55
_add_rect(slide, left=body_l, top=cy, width=body_w, height=1.5, fill_rgb=DEEPPINK_RGB)
tc = _text_on(DEEPPINK_RGB)
_add_text(slide, 'DISQUALIFYING, NOT A FOOTNOTE', left=body_l + 0.35, top=cy + 0.22, width=body_w - 0.7, height=0.5,
          size=17, color_rgb=tc, font=MONO_FONT, bold=True)
_add_text(slide, 'The detector might be far worse for some group of patients, and we would have no way to know. For a tool that touches patients, an un-runnable fairness audit sinks it on its own.',
          left=body_l + 0.35, top=cy + 0.78, width=body_w - 0.7, height=0.65, size=15, color_rgb=tc, font=SANS_FONT)
""",
"The gap that IS the finding. The figure is deliberately mostly blank -- a picture of metadata that does not exist. We CAN audit by class (the label is there), but the fairness audit -- who does it fail? -- cannot run, because age, sex, race, and scanner were never recorded. The pink bar drives it home: this is not a small caveat. A detector that might be far worse for some patients, with no way to check, is disqualified on its own (Gebru 2021; Seyyed-Kalantari 2021). Transition: so let's use a dataset that DID record demographics.")

# S13 -- FIGURE LEFT + TWO STAT BANDS: audit + fix by sex
add(f"""
fw = body_w * 0.56
_fit_image(slide, '{FIG}/stroke_fairness_fix.png', left=body_l, top=body_top, max_w=fw, max_h=body_h)
ax = body_l + fw + 0.45
aw = body_w - fw - 0.45
_add_rect(slide, left=ax, top=body_top + 0.15, width=aw, height=1.7, fill_rgb=TURQUOISE_RGB)
tc = _text_on(TURQUOISE_RGB)
_add_text(slide, '76% \\u2192 83%', left=ax + 0.28, top=body_top + 0.32, width=aw - 0.56, height=0.75,
          size=34, color_rgb=tc, font=MONO_FONT, bold=True)
_add_text(slide, "women's strokes caught, after a group-aware threshold (men held at 85%)",
          left=ax + 0.28, top=body_top + 1.12, width=aw - 0.56, height=0.6, size=13.5, color_rgb=tc, font=SANS_FONT)
_add_rect(slide, left=ax, top=body_top + 2.05, width=aw, height=1.5, fill_rgb=BLUEVIOLET_RGB)
tc2 = _text_on(BLUEVIOLET_RGB)
_add_text(slide, 'gap 9% \\u2192 2%', left=ax + 0.28, top=body_top + 2.2, width=aw - 0.56, height=0.7,
          size=30, color_rgb=tc2, font=MONO_FONT, bold=True)
_add_text(slide, 'recording sex made the gap visible AND closable', left=ax + 0.28,
          top=body_top + 2.95, width=aw - 0.56, height=0.5, size=13.5, color_rgb=tc2, font=SANS_FONT)
_add_text(slide, 'The audit the CT scans made impossible -- run, and fixed.',
          left=ax, top=body_top + 3.75, width=aw, height=1.0, size=15, color_rgb=INK_RGB, font=SANS_FONT, italic=True)
""",
"The payoff of the second dataset, and the whole thesis in one before/after. Because the table records SEX, we ran the audit the scans blocked. At a threshold tuned to catch 80% of strokes overall, women's strokes were caught only 76% of the time vs men's 85% -- a real 9-point gap (grey bars). The fix: give each sex its own threshold, equalizing UP -- women rise to 83% (turquoise bars), the gap all but closes. Recording sex made the gap both visible and fixable. That contrast with the CT wall is the point of the talk. Transition: and we can read WHY it predicts.")

# S14 -- FIGURE RIGHT + RANKED LIST LEFT: SHAP
add(f"""
fw = body_w * 0.55
_fit_image(slide, '{FIG}/stroke_shap.png', left=body_l + body_w - fw, top=body_top, max_w=fw, max_h=body_h)
aw = body_w - fw - 0.5
_add_text(slide, 'SHAP = feature importance for a table: how much each column moves the prediction.',
          left=body_l, top=body_top + 0.2, width=aw, height=1.1, size=16, color_rgb=INK_RGB, font=SANS_FONT)
ranks = [('age', 'by far the strongest driver', DEEPPINK_RGB),
         ('bmi + glucose', 'the next tier of risk', AMBER_RGB),
         ('sex', 'barely moves the score -- yet it mattered for FAIRNESS', BLUEVIOLET_RGB)]
y = body_top + 1.5
for lab, note, col in ranks:
    _add_text(slide, lab, left=body_l, top=y, width=aw, height=0.45,
              size=18, color_rgb=col, font=MONO_FONT, bold=True)
    _add_text(slide, note, left=body_l, top=y + 0.48, width=aw, height=0.7,
              size=15, color_rgb=INK_RGB, font=SANS_FONT)
    y += 1.25
""",
"Feature importance for the table. SHAP measures how much each column moves each prediction. Age dominates -- unsurprising for stroke -- with BMI and glucose next. The quiet point: SEX barely moves the risk SCORE, yet it mattered enormously for the fairness audit two slides ago. That is why you audit by group directly rather than trusting that a small feature-importance means a small disparity. A model you can read like this is one you can question and trust. Transition: pull it together.")

# S15 -- THREE-CARD RECAP: the contrast + the rule
add(f"""
cards = [('CT: WORKS, CAN\\'T AUDIT', 'AUC 0.817 -- a real detector. But the dataset records only image + label, so the who-does-it-fail audit simply cannot run.', 'no demographics = no audit', DEEPPINK_RGB),
         ('TABULAR: AUDITED + FIXED', 'AUC 0.81, and because it records sex we found a real gap (76% vs 85%) and closed it (to 83% vs 85%) with a group-aware threshold.', 'metadata = auditable + fixable', TURQUOISE_RGB),
         ('THE RULE', 'A model that runs is easy. A model you can AUDIT is the real bar -- and recording who is in the data is what makes it possible.', 'audit is the bar', BLUEVIOLET_RGB)]
gap = 0.35
cw = (body_w - gap * 2) / 3
ch = body_h - 0.1
by = body_top + 0.55
for i, (lab, body, tag, col) in enumerate(cards):
    x = body_l + i * (cw + gap)
    _add_rect(slide, left=x, top=body_top, width=cw, height=ch, fill_rgb=WHITE_RGB)
    _add_rect(slide, left=x, top=body_top, width=cw, height=0.1, fill_rgb=col)
    _add_text(slide, lab, left=x + 0.28, top=by, width=cw - 0.56, height=0.85,
              size=16, color_rgb=col, font=MONO_FONT, bold=True)
    _add_text(slide, body, left=x + 0.28, top=by + 1.05, width=cw - 0.56, height=ch - 2.6,
              size=15.5, color_rgb=INK_RGB, font=SANS_FONT)
    _add_rect(slide, left=x, top=body_top + ch - 1.0, width=cw, height=0.02, fill_rgb=_rgb('#E4E1D9'))
    _add_text(slide, tag, left=x + 0.28, top=body_top + ch - 0.85, width=cw - 0.56, height=0.6,
              size=14.5, color_rgb=col, font=MONO_FONT, bold=True, anchor=MSO_ANCHOR.MIDDLE)
""",
"The recap the rubric rewards, three cards. Pink: the CT model works (0.817) but cannot be audited -- no demographics. Turquoise: the tabular model works (0.81) AND, because it records sex, we audited and fixed a real gap (76 to 83 percent). Purple: the rule -- a model that runs is easy, a model you can audit is the real bar, and recording who is in the data is what makes it possible. Say the through-line one more time: imaging can hide who's in it; metadata is what lets you check fairness. Transition: the papers behind it.")

# S16 -- THREE-COLUMN REFERENCES
add(f"""
colw = (body_w - 1.2) / 3
c1 = ('HEAD-CT TRIAGE AI', '[1] Chilamkurthy 2018, The Lancet -- critical findings on head CT.\\n\\n[2] Flanders 2020, Radiology: AI -- the RSNA hemorrhage challenge.\\n\\n[3] Seyam 2022, Radiology: AI -- a deployed tool in the clinic.', TURQUOISE_RGB)
c2 = ('DATA & DOCUMENTATION', '[4] Gebru 2021, Comms of the ACM -- Datasheets for Datasets.\\n\\n[5] Tripathi 2023, J Am Coll Radiol -- biases in radiology-AI data.\\n\\n[8] Kim 2022, BMC Med Imaging -- transfer learning.', DEEPPINK_RGB)
c3 = ('FAIRNESS NEEDS METADATA', '[6] Seyyed-Kalantari 2021, Nature Medicine -- underdiagnosis bias.\\n\\n[7] Yang 2024, Nature Medicine -- limits of fair imaging AI.\\n\\nSaver 2006, Stroke -- "Time Is Brain".', BLUEVIOLET_RGB)
for i, (head, body, col) in enumerate([c1, c2, c3]):
    x = body_l + i * (colw + 0.6)
    _add_text(slide, head, left=x, top=body_top + 0.05, width=colw, height=0.8,
              size=14.5, color_rgb=col, font=MONO_FONT, bold=True)
    _add_text(slide, body, left=x, top=body_top + 0.95, width=colw, height=body_h - 1.05,
              size=13.5, color_rgb=INK_RGB, font=SANS_FONT)
""",
"Give credit and show the trail, in three groups. Left: the head-CT triage work that shows the clinical job is real [1][2][3]. Middle: the datasets-and-documentation papers -- Datasheets [4], the radiology-data review [5], transfer learning [8]. Right: the fairness audits that only worked because the data recorded groups [6][7], plus Saver 2006 for 'time is brain'. Point out that [4] and [6] are the backbone of our whole argument. Transition: the one sentence to remember.")

# S17 -- CENTERED CLOSING STATEMENT
add(f"""
q_top = body_top + body_h * 0.22
_add_text(slide, 'An imaging dataset can hide who is in it.',
          left=body_l, top=q_top, width=body_w, height=1.0,
          size=30, color_rgb=TURQUOISE_RGB, font=MONO_FONT, bold=True, align=PP_ALIGN.CENTER)
rule_w = 2.6
_add_rect(slide, left=body_l + (body_w - rule_w) / 2, top=q_top + 1.35, width=rule_w, height=0.05, fill_rgb=TURQUOISE_RGB)
_add_text(slide, 'Recording demographics is what turns fairness from a hope into something you can measure -- and fix. That is the finding, not the footnote.',
          left=body_l + 1.0, top=q_top + 1.9, width=body_w - 2.0, height=1.3,
          size=19, color_rgb=INK_RGB, font=SANS_FONT, align=PP_ALIGN.CENTER)
""",
"Land the deck on one idea: an imaging dataset can hide who is in it, and recording demographics is what turns fairness from a hope into something you can measure and fix. Our two datasets proved it -- the CT model worked but could not be audited; the tabular model was audited AND fixed. Leave them with the working rule of medical AI: a model you can check honestly beats a fancy one nobody can. Thank them.")

# ---- apply: dividers by label, freeform by order ----
ci = 0
for s in d["slides"]:
    p = s["params"]
    if s["kind"] == "section-divider":
        acc, note = DIV[p["label"]]
        p["accent_hex"] = acc
        p["notes"] = note
    elif s["kind"] == "freeform":
        p["_provenance"] = "agent"
        p["code"] = CODES[ci].strip("\n")
        p["notes"] = NOTES[ci]
        p["lede"] = ""
        ci += 1

assert ci == len(CODES), f"authored {ci} but have {len(CODES)} code blocks"
json.dump(d, open(LJ, "w"), indent=2)
print(f"authored {ci} freeform slides + {sum(1 for s in d['slides'] if s['kind']=='section-divider')} dividers; theme={d['theme']}")
