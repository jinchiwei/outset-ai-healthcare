"""Hand-author the G4 freeform sidecar: theme bone, divider accents, and a DISTINCT body
composition per content slide (no two alike). Loads the generated scaffold, mutates params in
place, writes back.  Run:  python author_sidecar.py
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
LJ = HERE / "g4_skin_access.md.layout.json"
FIG = str(HERE / "figures")  # absolute figure dir; _fit_image takes absolute paths fine

d = json.load(open(LJ))
d["theme"] = "bone"

# ---- section-divider accents (turquoise -> deeppink -> blueviolet -> amber -> turquoise) ----
DIV = {
    "Background": ("#40E0D0", "Open with the clinical scene, not the model. Three ideas to plant: the same skin-cancer exam runs across a whole range of skin tones; AI is pitched to reach people who cannot see a dermatologist; and a model is most accurate on the skin it saw most. Transition: let us meet the exam under the lens."),
    "The data": ("#FF1493", "The raw material, and the single fact that makes this project possible. PAD-UFES-20 is real clinical skin imaging that RECORDS the patient Fitzpatrick skin type -- most datasets do not, and without it a skin-tone gap is invisible. Transition: you can only audit what you record."),
    "The model": ("#8A2BE2", "How we build a screen worth auditing. Transfer learning borrows a pretrained brain, we fine-tune on PAD-UFES, and we split by patient so the score is honest. Then a clear recipe makes it reproducible. Transition: we do not have millions of skin images, so we borrow a brain."),
    "Results": ("#F0C840", "The payoff, in order: a working AUC, a fairness audit that comes out roughly fair where we can measure, the crisis of the 11 dark-skin patients we cannot measure, and where the model looks. Transition: first, does the screen even work?"),
    "Being honest": ("#40E0D0", "Good science states its own limits out loud. What the screen genuinely does, where the real gap is, the papers behind it, and the one sentence to remember. This is the section the rubric rewards most."),
}

CODE = {}
NOTE = {}

# S1 -- FULL-BLEED HERO: the clinical exam across six skin tones
CODE["The same lesion, six different skins"] = f"""
img_h = body_h - 0.5
_fit_image(slide, '{FIG}/intro_skin_tone_exam.png', left=body_l, top=body_top, max_w=body_w, max_h=img_h)
_add_text(slide, 'The same skin-cancer check, from the lightest skin (I) to the darkest (VI) -- and how much training data any AI ever saw for each tone.',
          left=body_l + 1.0, top=body_top + img_h + 0.1, width=body_w - 2.0, height=0.5,
          size=14, color_rgb=INK_RGB, font=SANS_FONT, italic=True, align=PP_ALIGN.CENTER)
"""
NOTE["The same lesion, six different skins"] = ("Open concrete and physical. A dermatologist puts a dermatoscope on a spot and asks: is this cancer? Now run that same exam across Fitzpatrick I to VI. The exam does not change -- what changes is the bar under each tone: how many images the AI ever practiced on. Tall on the light end, almost nothing on the dark end. Do not mention fairness metrics yet; just let them SEE the skew. Transition: who is that dark-skin group in the real world?")

# S2 -- FIGURE LEFT + PROSE ASIDE: the access irony
CODE["Who can even reach a dermatologist?"] = f"""
fw = body_w * 0.56
_fit_image(slide, '{FIG}/intro_access_gap.png', left=body_l, top=body_top, max_w=fw, max_h=body_h)
ax = body_l + fw + 0.45
aw = body_w - fw - 0.45
_add_text(slide, 'A PROMISE, AND AN IRONY', left=ax, top=body_top + 0.15, width=aw, height=0.4,
          size=15, color_rgb=DEEPPINK_RGB, font=MONO_FONT, bold=True)
_add_text(slide, 'The promise: put a screen in a phone and reach the roughly three billion people who cannot see a dermatologist.',
          left=ax, top=body_top + 0.75, width=aw, height=1.35, size=16, color_rgb=INK_RGB, font=SANS_FONT)
_add_text(slide, 'The irony: those same people -- often darker-skinned, and outside Europe and North America -- are the ones barely present in the training data.',
          left=ax, top=body_top + 2.2, width=aw, height=1.5, size=16, color_rgb=INK_RGB, font=SANS_FONT)
_add_text(slide, 'A tool that works worst where care is scarcest can widen the very gap it was sold to close.',
          left=ax, top=body_top + 3.85, width=aw, height=1.2, size=16, color_rgb=BLUEVIOLET_RGB, font=SANS_FONT, italic=True)
"""
NOTE["Who can even reach a dermatologist?"] = ("The stakes slide. Each figure in the waffle is about 200 million people; the pink block is the ~3 billion with little dermatology access -- the group the technology is pitched to serve. Spell out the irony in plain words: they are also the group least present in the data these models learn from. So a model that looks fine on average can fail exactly where it was meant to help. Transition: and this is a measured effect, not a worry.")

# S3 -- FIGURE HERO + STAT CHIPS: the Groh redraw
CODE["A model is best at the skin it saw most"] = f"""
img_h = body_h - 1.15
_fit_image(slide, '{FIG}/intro_groh_2021.png', left=body_l + body_w * 0.14, top=body_top, max_w=body_w * 0.72, max_h=img_h)
chip_y = body_top + img_h + 0.18
chips = [('16,577', 'clinical images Groh labeled by Fitzpatrick type', TURQUOISE_RGB),
         ('I -> VI', 'the six skin types it audited separately', DEEPPINK_RGB),
         ('2021', 'Groh et al., CVPR -- the result we redraw', BLUEVIOLET_RGB)]
gap = 0.3
cw = (body_w - gap * 2) / 3
for i, (num, lab, col) in enumerate(chips):
    x = body_l + i * (cw + gap)
    _add_text(slide, num, left=x, top=chip_y, width=cw, height=0.55,
              size=30, color_rgb=col, font=MONO_FONT, bold=True, align=PP_ALIGN.CENTER)
    _add_text(slide, lab, left=x, top=chip_y + 0.58, width=cw, height=0.45,
              size=12.5, color_rgb=INK_RGB, font=SANS_FONT, align=PP_ALIGN.CENTER)
"""
NOTE["A model is best at the skin it saw most"] = ("The evidence slide, redrawn from Groh 2021 (attributed). Gold bars are how many training images the model saw per Fitzpatrick type; the turquoise line is its accuracy. The line tracks the bars: high on the light types it saw a lot, dropping to the dark types it barely saw. This is the mechanism behind the whole equity worry -- a model gets best at what it practices most. Daneshjou 2022 confirmed it on biopsy-proven dark-skin images. Transition: so to check our own model, we need data that records skin tone.")

# S4 -- BIG-NUMBER TILES: why PAD-UFES-20
CODE["Why PAD-UFES-20"] = f"""
tiles = [('1,494', 'real clinical skin photos, each biopsy-labeled malignant or benign', DEEPPINK_RGB),
         ('skin type', 'PAD-UFES-20 records every patient Fitzpatrick skin tone -- rare, and the whole reason we can audit', TURQUOISE_RGB),
         ('by patient', 'train and test never share a patient, so the AUC we report is honest', BLUEVIOLET_RGB)]
gap = 0.35
cw = (body_w - gap * 2) / 3
ch = body_h - 0.10
num_h = 1.5
lab_h = 2.0
block_top = body_top + (ch - (num_h + 0.2 + lab_h)) / 2
for i, (num, lab, col) in enumerate(tiles):
    x = body_l + i * (cw + gap)
    _add_rect(slide, left=x, top=body_top, width=cw, height=ch, fill_rgb=col)
    tc = _text_on(col)
    sz = 60 if i == 0 else 40
    _add_text(slide, num, left=x + 0.28, top=block_top, width=cw - 0.56, height=num_h,
              size=sz, color_rgb=tc, font=MONO_FONT, bold=True, anchor=MSO_ANCHOR.MIDDLE)
    _add_text(slide, lab, left=x + 0.28, top=block_top + num_h + 0.2, width=cw - 0.56, height=lab_h,
              size=15, color_rgb=tc, font=SANS_FONT)
"""
NOTE["Why PAD-UFES-20"] = ("Justify the dataset in three ideas. 1,494 real biopsy-labeled clinical photos -- enough to train a genuine screen. It RECORDS the Fitzpatrick skin type -- the rare, decisive feature; without it, no skin-tone audit is even possible. And we split by patient, so the score is not inflated by the model memorizing a person. The middle tile is the quiet reason this dataset beats a bigger, tone-blind pile of images. Transition: why does recording skin tone matter so much?")

# S5 -- DRAWN TWO-PANEL COMPARE: tone-blind vs records-tone
CODE["You can only audit what you record"] = f"""
gap = 0.5
pw = (body_w - gap) / 2
ph = body_h - 0.1
skin = [_rgb('#E7C39C'), _rgb('#B67C4E'), _rgb('#573421')]
# --- LEFT panel: most datasets are tone-blind ---
_add_rect(slide, left=body_l, top=body_top, width=pw, height=ph, fill_rgb=_rgb('#EDEAE1'))
_add_rect(slide, left=body_l, top=body_top, width=pw, height=0.12, fill_rgb=MUTED_RGB)
_add_text(slide, 'MOST SKIN DATASETS', left=body_l + 0.3, top=body_top + 0.35, width=pw - 0.6, height=0.45,
          size=17, color_rgb=MUTED_RGB, font=MONO_FONT, bold=True)
_add_text(slide, 'record the lesion -- and nothing about the patient. Skin tone is simply not in the file.',
          left=body_l + 0.3, top=body_top + 0.9, width=pw - 0.6, height=0.9, size=14.5, color_rgb=INK_RGB, font=SANS_FONT)
cy = body_top + 2.1
for j in range(3):
    cx = body_l + 0.45 + j * 1.55
    _add_rect(slide, left=cx, top=cy, width=1.2, height=1.2, fill_rgb=_rgb('#C7C2B4'))
    _add_text(slide, '?', left=cx, top=cy, width=1.2, height=1.2,
              size=40, color_rgb=DEEPPINK_RGB, font=MONO_FONT, bold=True, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
_add_text(slide, 'skin tone: unknown', left=body_l + 0.3, top=cy + 1.35, width=pw - 0.6, height=0.4,
          size=14, color_rgb=DEEPPINK_RGB, font=MONO_FONT, bold=True)
_add_text(slide, 'CANNOT audit fairness by skin tone', left=body_l + 0.3, top=body_top + ph - 0.7, width=pw - 0.6, height=0.5,
          size=16, color_rgb=DEEPPINK_RGB, font=MONO_FONT, bold=True)
# --- RIGHT panel: PAD-UFES-20 records skin tone ---
rx = body_l + pw + gap
_add_rect(slide, left=rx, top=body_top, width=pw, height=ph, fill_rgb=WHITE_RGB)
_add_rect(slide, left=rx, top=body_top, width=pw, height=0.12, fill_rgb=TURQUOISE_RGB)
_add_text(slide, 'PAD-UFES-20', left=rx + 0.3, top=body_top + 0.35, width=pw - 0.6, height=0.45,
          size=17, color_rgb=TURQUOISE_RGB, font=MONO_FONT, bold=True)
_add_text(slide, 'records each patient Fitzpatrick skin type, so the score can be split apart and checked.',
          left=rx + 0.3, top=body_top + 0.9, width=pw - 0.6, height=0.9, size=14.5, color_rgb=INK_RGB, font=SANS_FONT)
for j in range(3):
    cx = rx + 0.45 + j * 1.55
    _add_rect(slide, left=cx, top=cy, width=1.2, height=1.2, fill_rgb=skin[j])
    _add_text(slide, ['I-II', 'III-IV', 'V-VI'][j], left=cx, top=cy + 1.25, width=1.2, height=0.35,
              size=12, color_rgb=INK_RGB, font=MONO_FONT, bold=True, align=PP_ALIGN.CENTER)
_add_text(slide, 'skin tone: recorded', left=rx + 0.3, top=cy + 1.7, width=pw - 0.6, height=0.4,
          size=14, color_rgb=TURQUOISE_RGB, font=MONO_FONT, bold=True)
_add_text(slide, 'CAN audit fairness by skin tone', left=rx + 0.3, top=body_top + ph - 0.7, width=pw - 0.6, height=0.5,
          size=16, color_rgb=TURQUOISE_RGB, font=MONO_FONT, bold=True)
"""
NOTE["You can only audit what you record"] = ("The pivot of the whole talk. Left: most skin datasets store the lesion and nothing about the person -- skin tone is a question mark, so a skin-tone gap is invisible no matter how hard you look. Right: PAD-UFES-20 writes down the Fitzpatrick type, so the score can be split apart by tone and held accountable. Same kind of model, but only one of these can be audited. Point out that this is why we chose this dataset over bigger, tone-blind ones. Transition: with an auditable dataset in hand, we build the screen.")

# S6 -- DRAWN 3-STAGE PIPELINE: transfer learning + patient split
CODE["Borrow a brain, then split by patient"] = f"""
stages = [('PRETRAINED', 'a network already trained on millions of everyday photos -- cats, cars, faces', 'it knows edges, colors, shapes', TURQUOISE_RGB),
          ('FINE-TUNE', 'train it further on PAD-UFES-20 clinical skin images', 'it learns malignant vs benign', DEEPPINK_RGB),
          ('SPLIT BY PATIENT', 'grade only on patients the model never saw during training', 'an honest AUC', BLUEVIOLET_RGB)]
gap = 1.0
cw = (body_w - gap * 2) / 3
ch = body_h - 1.1
ctop = body_top + 0.1
for i, (lab, bod, det, col) in enumerate(stages):
    x = body_l + i * (cw + gap)
    _add_rect(slide, left=x, top=ctop, width=cw, height=ch, fill_rgb=_rgb('#EDEAE1'))
    _add_rect(slide, left=x, top=ctop, width=cw, height=0.12, fill_rgb=col)
    _add_text(slide, lab, left=x + 0.28, top=ctop + 0.45, width=cw - 0.56, height=0.5,
              size=18, color_rgb=col, font=MONO_FONT, bold=True)
    _add_text(slide, bod, left=x + 0.28, top=ctop + 1.25, width=cw - 0.56, height=ch - 2.6,
              size=15.5, color_rgb=INK_RGB, font=SANS_FONT)
    _add_rect(slide, left=x + 0.28, top=ctop + ch - 0.95, width=cw - 0.56, height=0.02, fill_rgb=col)
    _add_text(slide, det, left=x + 0.28, top=ctop + ch - 0.8, width=cw - 0.56, height=0.6,
              size=14, color_rgb=col, font=MONO_FONT, bold=True, anchor=MSO_ANCHOR.MIDDLE)
    if i < 2:
        _add_text(slide, '->', left=x + cw + 0.05, top=ctop + ch / 2 - 0.45, width=gap - 0.1, height=0.9,
                  size=40, color_rgb=INK_RGB, font=MONO_FONT, bold=True, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
_add_text(slide, 'The patient-level split is what keeps the score honest -- no person can appear in both training and test.',
          left=body_l, top=ctop + ch + 0.18, width=body_w, height=0.55, size=15,
          color_rgb=BLUEVIOLET_RGB, font=SANS_FONT, italic=True, align=PP_ALIGN.CENTER)
"""
NOTE["Borrow a brain, then split by patient"] = ("The modeling idea, in three stages. We have far too few skin images to teach a network from scratch, so we borrow a brain -- a net pretrained on millions of ordinary photos -- and fine-tune it on PAD-UFES skin. The third stage is the honesty step: split by PATIENT, so the same person never lands in both train and test, or the AUC would be inflated by memorization. Walk them left to right. Transition: here is the full recipe so anyone could rebuild it.")

# S7 -- NUMBERED PROCESS: methods spec
CODE["Model and data processing"] = f"""
steps = [('1', 'MODEL', 'caformer_s18, pretrained, then fine-tuned with a fresh 2-class head', TURQUOISE_RGB),
         ('2', 'RESIZE', 'every clinical photo decoded to color and resized to 224x224 pixels', DEEPPINK_RGB),
         ('3', 'NORMALIZE', 'pixels rescaled to the fixed mean and scale the backbone was trained on', AMBER_RGB),
         ('4', 'TASK', 'malignant (BCC, SCC, melanoma) vs benign -- one yes/no call', BLUEVIOLET_RGB),
         ('5', 'SPLIT & AUDIT', 'patient-level train/test split; then audit AUC by recorded skin tone', TURQUOISE_RGB)]
n = len(steps)
gap = 0.25
rh = (body_h - gap * (n - 1)) / n
for i, (num, lab, bod, col) in enumerate(steps):
    y = body_top + i * (rh + gap)
    _add_rect(slide, left=body_l, top=y, width=0.9, height=rh, fill_rgb=col)
    _add_text(slide, num, left=body_l, top=y, width=0.9, height=rh,
              size=26, color_rgb=_text_on(col), font=MONO_FONT, bold=True, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    _add_text(slide, lab, left=body_l + 1.15, top=y, width=2.9, height=rh,
              size=16, color_rgb=col, font=MONO_FONT, bold=True, anchor=MSO_ANCHOR.MIDDLE)
    _add_text(slide, bod, left=body_l + 4.2, top=y, width=body_w - 4.2, height=rh,
              size=15, color_rgb=INK_RGB, font=SANS_FONT, anchor=MSO_ANCHOR.MIDDLE)
"""
NOTE["Model and data processing"] = ("The reproducibility slide -- read it as a recipe, not trivia. Five steps: a pretrained CAFormer fine-tuned to a 2-class head; every image resized to 224x224; normalized to the backbone range; the task framed as malignant vs benign; and a patient-level split whose held-out AUC we then audit by skin tone. Step 5 is where the equity work lives -- the split gives an honest number and the recorded tone lets us break it apart. Transition: so does the screen actually work?")

# S8 -- BIG-STAT HERO (no figure): the working AUC
CODE["A working screen"] = f"""
_add_text(slide, '0.83', left=body_l, top=body_top + 0.35, width=body_w, height=1.7,
          size=115, color_rgb=AMBER_RGB, font=MONO_FONT, bold=True, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
_add_text(slide, 'AUC on held-out patients', left=body_l, top=body_top + 2.3, width=body_w, height=0.5,
          size=20, color_rgb=INK_RGB, font=MONO_FONT, bold=True, align=PP_ALIGN.CENTER)
_add_text(slide, 'AUC is the area under the ROC curve: 0.5 is a coin flip, 1.0 is perfect. We grade on AUC, not accuracy, because a rare-cancer dataset rewards a lazy "always benign" guess.',
          left=body_l + 1.4, top=body_top + 3.05, width=body_w - 2.8, height=1.1,
          size=16, color_rgb=INK_RGB, font=SANS_FONT, align=PP_ALIGN.CENTER)
_add_text(slide, 'A genuinely working screen -- which is exactly what makes the fairness audit worth doing.',
          left=body_l + 1.4, top=body_top + 4.2, width=body_w - 2.8, height=0.8,
          size=17, color_rgb=BLUEVIOLET_RGB, font=SANS_FONT, italic=True, align=PP_ALIGN.CENTER)
"""
NOTE["A working screen"] = ("Lead with the headline, but say what it means. 0.83 AUC on patients the model never saw -- a real, working malignant-vs-benign screen. Name the metric in one breath: AUC sweeps every cut-off, 0.5 is a coin flip, 1.0 is perfect, and we use it instead of accuracy because a lazy always-benign guess would score high accuracy and catch nothing. The point of proving it works: auditing a broken model would tell us nothing. Transition: now the real question -- is it fair across skin tones?")

# S9 -- FIGURE TOP WIDE + STAT BAND: fairness where measurable
CODE["Fair across the tones we can measure"] = f"""
img_h = body_h - 1.55
_fit_image(slide, '{FIG}/equity_before_after.png', left=body_l + body_w * 0.08, top=body_top, max_w=body_w * 0.84, max_h=img_h)
band_y = body_top + img_h + 0.15
band_h = 1.35
gap = 0.35
half = (body_w - gap) / 2
_add_rect(slide, left=body_l, top=band_y, width=half, height=band_h, fill_rgb=TURQUOISE_RGB)
tc = _text_on(TURQUOISE_RGB)
_add_text(slide, 'gap 0.01 AUC', left=body_l + 0.3, top=band_y + 0.12, width=half - 0.6, height=0.55,
          size=26, color_rgb=tc, font=MONO_FONT, bold=True)
_add_text(slide, 'light vs medium skin -- already roughly fair where we can actually measure it',
          left=body_l + 0.3, top=band_y + 0.72, width=half - 0.6, height=0.55, size=13.5, color_rgb=tc, font=SANS_FONT)
x2 = body_l + half + gap
_add_rect(slide, left=x2, top=band_y, width=half, height=band_h, fill_rgb=DEEPPINK_RGB)
tc2 = _text_on(DEEPPINK_RGB)
_add_text(slide, 'dark skin: n = 11', left=x2 + 0.3, top=band_y + 0.12, width=half - 0.6, height=0.55,
          size=26, color_rgb=tc2, font=MONO_FONT, bold=True)
_add_text(slide, 'a tone-balanced retrain cannot rebalance data that is not there -- the gap stays invisible',
          left=x2 + 0.3, top=band_y + 0.72, width=half - 0.6, height=0.55, size=13.5, color_rgb=tc2, font=SANS_FONT)
"""
NOTE["Fair across the tones we can measure"] = ("The honest fairness result. Because the data records tone, we compute AUC separately per group. Light 0.836 vs medium 0.829 -- a gap of only about 0.01, so the screen is already roughly fair where we can measure. The turquoise band says so plainly. The pink band is the catch: the dark group has 11 patients, and a common fairness fix -- oversampling the rare group -- did nothing here, because you cannot rebalance data that does not exist. Do NOT claim we fixed a gap. Transition: look at just how few those 11 are.")

# S10 -- FIGURE LEFT + GIANT CALLOUT: the 11-patient crisis
CODE["The real crisis: 11 patients"] = f"""
fw = body_w * 0.60
_fit_image(slide, '{FIG}/tone_distribution.png', left=body_l, top=body_top, max_w=fw, max_h=body_h)
ax = body_l + fw + 0.4
aw = body_w - fw - 0.4
_add_text(slide, '11', left=ax, top=body_top + 0.2, width=aw, height=1.6,
          size=115, color_rgb=DEEPPINK_RGB, font=MONO_FONT, bold=True, anchor=MSO_ANCHOR.MIDDLE)
_add_text(slide, 'dark-skin patients, out of 1,494', left=ax, top=body_top + 2.0, width=aw, height=0.5,
          size=17, color_rgb=INK_RGB, font=MONO_FONT, bold=True)
_add_text(slide, 'You cannot validate a screen for a whole group of people from 11 of them -- no matter how good the overall average looks.',
          left=ax, top=body_top + 2.65, width=aw, height=1.5, size=16, color_rgb=INK_RGB, font=SANS_FONT)
_add_text(slide, 'The under-representation is not a footnote. It is the finding.',
          left=ax, top=body_top + 4.2, width=aw, height=0.8, size=16, color_rgb=BLUEVIOLET_RGB, font=SANS_FONT, italic=True)
"""
NOTE["The real crisis: 11 patients"] = ("The emotional peak of the deck. The average AUC looked fine, and the measurable tones looked fair -- but here is what the average hides: 11 dark-skin patients out of 1,494. Say the giant 11 out loud. You cannot certify a screen for dark skin from 11 people; the confidence interval alone would be enormous. This is the honest, constructive core of the project: the crisis is not a broken model, it is missing data. Transition: one more check on the model itself -- is it looking at the right thing?")

# S11 -- FILMSTRIP + CAPTION BAND: Grad-CAM
CODE["Where the model looks"] = f"""
img_h = body_h - 1.0
_fit_image(slide, '{FIG}/gradcam.png', left=body_l, top=body_top, max_w=body_w, max_h=img_h)
band_y = body_top + img_h + 0.12
_add_rect(slide, left=body_l, top=band_y, width=body_w, height=0.78, fill_rgb=_rgb('#EDEAE1'))
_add_text(slide, 'Grad-CAM = feature importance for an image: the bright areas sit ON the lesion, the cues a clinician reads. It shows the model looks in the right PLACE -- not that it works on skin it never trained on.',
          left=body_l + 0.3, top=band_y + 0.1, width=body_w - 0.6, height=0.6,
          size=14.5, color_rgb=INK_RGB, font=SANS_FONT, anchor=MSO_ANCHOR.MIDDLE)
"""
NOTE["Where the model looks"] = ("The image version of feature importance. For a table we would ask SHAP which columns mattered; for a photo we use Grad-CAM, which highlights WHERE the model looked. The heat lands on the pigmented lesion, not a pen mark or a hair -- reassuring evidence the screen reads the spot itself. But be precise about what this does and does not prove: it shows the model looks in the right place, not that it generalizes to skin tones absent from training. Only data can answer that. Transition: so where does this leave us, honestly?")

# S12 -- CARDS ROW: what the project shows
CODE["What this project shows"] = f"""
cards = [('IT WORKS, AND WE COULD AUDIT IT', 'AUC 0.83, and because PAD-UFES-20 records skin tone we could actually measure fairness -- which most skin datasets make impossible.', 'AUC 0.83 · auditable', TURQUOISE_RGB),
         ('THE GAP IS WHO IS MISSING', 'Only 11 dark-skin patients -- too few to validate. And no race or country is recorded, so those gaps stay invisible here.', '11 dark-skin patients', DEEPPINK_RGB),
         ('THE FIX IS DATA, NOT A TRICK', 'Rebalancing cannot invent dark-skin patients. Fixing dark-skin performance means collecting representative data -- DDI, Fitzpatrick 17k.', 'collect, do not resample', BLUEVIOLET_RGB)]
gap = 0.35
cw = (body_w - gap * 2) / 3
ch = body_h - 0.1
by = body_top + 0.6
for i, (lab, bod, tag, col) in enumerate(cards):
    x = body_l + i * (cw + gap)
    _add_rect(slide, left=x + 0.05, top=body_top + 0.06, width=cw, height=ch, fill_rgb=_rgb('#E4E1D9'))
    _add_rect(slide, left=x, top=body_top, width=cw, height=ch, fill_rgb=WHITE_RGB)
    _add_rect(slide, left=x, top=body_top, width=cw, height=0.1, fill_rgb=col)
    _add_text(slide, lab, left=x + 0.28, top=by, width=cw - 0.56, height=0.85,
              size=16, color_rgb=col, font=MONO_FONT, bold=True)
    _add_text(slide, bod, left=x + 0.28, top=by + 1.05, width=cw - 0.56, height=ch - 2.6,
              size=16, color_rgb=INK_RGB, font=SANS_FONT)
    _add_rect(slide, left=x, top=body_top + ch - 1.05, width=cw, height=0.02, fill_rgb=_rgb('#E4E1D9'))
    _add_text(slide, tag, left=x + 0.28, top=body_top + ch - 0.9, width=cw - 0.56, height=0.6,
              size=15, color_rgb=col, font=MONO_FONT, bold=True, anchor=MSO_ANCHOR.MIDDLE)
"""
NOTE["What this project shows"] = ("Model the honesty the rubric rewards, in three cards. Green: it genuinely works AND we could audit it -- AUC 0.83, and auditing was only possible because the data records skin tone. Pink: the real gap is who is missing -- 11 dark-skin patients, and no race or country recorded at all. Purple: the fix is data, not a resampling trick -- point students to DDI and Fitzpatrick 17k, which exist precisely to collect representative dark-skin images. This is a working solution with an honest edge, not a complaint. Transition: the papers behind it.")

# S13 -- REFERENCES 2-COL
CODE["References"] = """
colw = (body_w - 0.6) / 2
left_refs = ('[1] Fitzpatrick 1988, Arch Dermatol -- the six-point skin-type scale.\\n\\n'
             '[2] Groh et al. 2021, CVPR Workshops -- Fitzpatrick 17k; accuracy tracks representation.\\n\\n'
             '[3] Daneshjou et al. 2022, Sci Adv -- DDI; top models drop on dark skin.\\n\\n'
             '[4] Obermeyer et al. 2019, Science -- a fair-looking algorithm that underserved Black patients.')
right_refs = ('[5] Wen et al. 2022, Lancet Digital Health -- most public skin images are light skin.\\n\\n'
              '[6] Adamson and Smith 2018, JAMA Dermatology -- AI can widen dermatology care gaps.\\n\\n'
              '[7] Buster, Stevens and Elmets 2012, Dermatologic Clinics -- disparities and access.')
_add_text(slide, 'GAPS & LANDMARKS', left=body_l, top=body_top + 0.05, width=colw, height=0.4,
          size=16, color_rgb=TURQUOISE_RGB, font=MONO_FONT, bold=True)
_add_text(slide, left_refs, left=body_l, top=body_top + 0.6, width=colw, height=body_h - 0.7,
          size=14.5, color_rgb=INK_RGB, font=SANS_FONT)
rx = body_l + colw + 0.6
_add_text(slide, 'DATA & ACCESS', left=rx, top=body_top + 0.05, width=colw, height=0.4,
          size=16, color_rgb=DEEPPINK_RGB, font=MONO_FONT, bold=True)
_add_text(slide, right_refs, left=rx, top=body_top + 0.6, width=colw, height=body_h - 0.7,
          size=14.5, color_rgb=INK_RGB, font=SANS_FONT)
"""
NOTE["References"] = ("Give credit and show the trail. Left column names the scale and the gap-measuring studies: Fitzpatrick [1], Groh Fitzpatrick 17k [2], Daneshjou DDI [3], and the Obermeyer proxy-bias case [4]. Right column is data and access: the review showing public images skew light [5], the warning that AI can widen care gaps [6], and the disparities-and-access background [7]. Point out that [2] is the study we redrew and [3] is the collect-better-data fix we recommend. Transition: the one sentence to remember.")

# S14 -- CLOSING QUOTE
CODE["The honest bottom line"] = """
q_top = body_top + body_h * 0.22
_add_text(slide, 'A model can only be fair to the people the data remembers.',
          left=body_l + 0.5, top=q_top, width=body_w - 1.0, height=1.4,
          size=30, color_rgb=TURQUOISE_RGB, font=MONO_FONT, bold=True, align=PP_ALIGN.CENTER)
rule_w = 2.4
_add_rect(slide, left=body_l + (body_w - rule_w) / 2, top=q_top + 1.7, width=rule_w, height=0.05, fill_rgb=TURQUOISE_RGB)
_add_text(slide, 'This screen works, and we could audit it only because the data recorded skin tone. Record who the patient is -- then go collect the data the model is missing.',
          left=body_l + 1.0, top=q_top + 2.25, width=body_w - 2.0, height=1.3,
          size=19, color_rgb=INK_RGB, font=SANS_FONT, align=PP_ALIGN.CENTER)
"""
NOTE["The honest bottom line"] = ("Land the deck on one idea: a model can only be fair to the people the data remembers. Our screen works and we could audit it -- but only because PAD-UFES recorded skin tone, and even then 11 dark-skin patients is a gap no algorithm can close. Leave them with the constructive rule: record who the patient is, then go collect the data the model is missing. That is how you actually make dermatology AI fair. Thank them.")

# ---- apply ----
for s in d["slides"]:
    p = s["params"]
    if s["kind"] == "section-divider":
        acc, note = DIV[p["label"]]
        p["accent_hex"] = acc
        p["notes"] = note
    else:
        title = p["title"]
        assert title in CODE, f"no code for slide: {title!r}"
        p["_provenance"] = "agent"
        p["code"] = CODE[title].strip("\n")
        p["notes"] = NOTE[title]

json.dump(d, open(LJ, "w"), indent=2)
print("authored", len(d["slides"]), "slides ->", LJ.name)
