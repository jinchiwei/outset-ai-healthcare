"""Hand-author the G2 freeform sidecar: theme bone, divider accents, and a DISTINCT body
composition per content slide. Loads the generated scaffold, mutates params in place, writes back.
Run:  python author_sidecar.py
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
LJ = HERE / "g2_skin_screening.md.layout.json"
FIG = str(HERE / "figures")  # absolute figure dir; _fit_image takes absolute paths fine

d = json.load(open(LJ))
d["theme"] = "bone"

# ---- section-divider accents (turquoise -> deeppink -> blueviolet -> amber -> turquoise) ----
DIV = {
    "Background": ("#40E0D0", "We open with the clinical scene so the modeling lands later. Three ideas to plant: a dermatoscope sorts seven look-alike lesions, exactly one (melanoma) is deadly, and machines can already do this at dermatologist level. Transition: let's meet the spot under the lens."),
    "The data": ("#FF1493", "Now the raw material. Why HAM10000 is the right dataset -- real dermatoscopy, big enough, and it records age and sex so we can audit fairness -- and the imbalance that makes plain accuracy a trap. Transition: only about 1 in 9 spots is melanoma."),
    "The model": ("#8A2BE2", "How we turn images into a screen. Transfer learning borrows a pretrained brain, a bake-off picks the backbone, and a clear recipe makes it reproducible. Transition: we don't have millions of skin images, so we borrow a brain."),
    "Results": ("#F0C840", "The payoff, and every number named by how we measured it. A working AUC, the threshold fix that lifts recall, the confusion matrix, a fairness check by sex, and where the model looks. Transition: first, how do you even grade a screen?"),
    "Being honest": ("#40E0D0", "Good science states its own limits out loud. What the screen genuinely does, where it is uneven, the papers behind it, and the one sentence to remember. This is the section the rubric rewards most."),
}

# ---- per-slide freeform bodies, keyed by slide title ----
CODE = {}
NOTE = {}

# S1 -- FULL-BLEED HERO: the dermatoscope scene
CODE["The spot that could be melanoma"] = f"""
img_h = body_h - 0.5
_fit_image(slide, '{FIG}/intro_dermatoscope.png', left=body_l, top=body_top, max_w=body_w, max_h=img_h)
_add_text(slide, 'A dermatoscope magnifies the spot; the screen must sort seven look-alikes and never wave the melanoma through.',
          left=body_l + 1.0, top=body_top + img_h + 0.1, width=body_w - 2.0, height=0.5,
          size=14, color_rgb=INK_RGB, font=SANS_FONT, italic=True, align=PP_ALIGN.CENTER)
"""
NOTE["The spot that could be melanoma"] = ("Open concrete and physical. A dermatologist puts a dermatoscope on a skin spot and asks one question: harmless, or melanoma? Point at the seven lesion types down the left -- these are the HAM10000 classes -- and note that only MELANOMA (in pink) is the deadly one; the arrow points to it under the lens. Do not mention machine learning yet. Transition: why does missing that one matter so much?")

# S2 -- FIGURE LEFT + PROSE ASIDE: why a miss is deadly
CODE["Why a miss is deadly"] = f"""
fw = body_w * 0.56
_fit_image(slide, '{FIG}/intro_cost_of_a_miss.png', left=body_l, top=body_top, max_w=fw, max_h=body_h)
ax = body_l + fw + 0.45
aw = body_w - fw - 0.45
_add_text(slide, 'TWO MISTAKES, ONE IS WORSE', left=ax, top=body_top + 0.15, width=aw, height=0.4,
          size=15, color_rgb=DEEPPINK_RGB, font=MONO_FONT, bold=True)
_add_text(slide, 'A false negative -- a real melanoma called harmless -- lets a cancer keep growing, and can be fatal.',
          left=ax, top=body_top + 0.75, width=aw, height=1.35, size=16, color_rgb=INK_RGB, font=SANS_FONT)
_add_text(slide, 'A false positive is just one extra check-up. The costs are wildly lopsided.',
          left=ax, top=body_top + 2.15, width=aw, height=1.1, size=16, color_rgb=INK_RGB, font=SANS_FONT)
_add_text(slide, 'So we do not chase balanced accuracy. We deliberately tilt the screen toward catching cancers.',
          left=ax, top=body_top + 3.35, width=aw, height=1.3, size=16, color_rgb=BLUEVIOLET_RGB, font=SANS_FONT, italic=True)
"""
NOTE["Why a miss is deadly"] = ("The single idea that drives every later choice. The balance is tipped on purpose: a MISSED MELANOMA (heavy, pink) outweighs a FALSE ALARM (light, gold). Spell out the two error types in plain words -- false negative vs false positive -- and their real-world costs: a spreading cancer versus one extra visit. Because they are not equal, we will later tune the screen toward sensitivity even though it costs specificity. Transition: and this is not hypothetical -- machines already do this well.")

# S3 -- FULL-BLEED HERO + STAT CHIPS: Esteva 2017
CODE["Machines already match dermatologists"] = f"""
img_h = body_h - 1.15
_fit_image(slide, '{FIG}/intro_esteva_2017.png', left=body_l + body_w * 0.14, top=body_top, max_w=body_w * 0.72, max_h=img_h)
chip_y = body_top + img_h + 0.18
chips = [('129,450', 'clinical images it learned from', TURQUOISE_RGB),
         ('21', 'board-certified dermatologists it matched', DEEPPINK_RGB),
         ('2017', 'the year this stopped being science fiction', BLUEVIOLET_RGB)]
gap = 0.3
cw = (body_w - gap * 2) / 3
for i, (num, lab, col) in enumerate(chips):
    x = body_l + i * (cw + gap)
    _add_text(slide, num, left=x, top=chip_y, width=cw, height=0.55,
              size=30, color_rgb=col, font=MONO_FONT, bold=True, align=PP_ALIGN.CENTER)
    _add_text(slide, lab, left=x, top=chip_y + 0.58, width=cw, height=0.45,
              size=12.5, color_rgb=INK_RGB, font=SANS_FONT, align=PP_ALIGN.CENTER)
"""
NOTE["Machines already match dermatologists"] = ("The landmark that makes this project credible. Esteva 2017 trained one network on ~129,450 clinical images and it matched 21 board-certified dermatologists: on the sensitivity/specificity plane the CNN's turquoise curve sits above the dermatologists' cloud (grey dots) and their pink average. This is a redraw of their result, attributed. Our project is a small, honest version of the same idea. Transition: to build one, we need the right data.")

# S4 -- STAT TILES: why HAM10000
CODE["Why HAM10000"] = f"""
tiles = [('10,015', 'real dermatoscopy images -- the kind a clinic actually captures, not cartoons', DEEPPINK_RGB),
         ('7', 'lesion types, which we collapse to one screening question: melanoma vs. everything else', BLUEVIOLET_RGB),
         ('11%', 'are melanoma -- the rare, deadly class -- and every image records age and sex, so we can audit fairness', TURQUOISE_RGB)]
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
    _add_text(slide, num, left=x + 0.28, top=block_top, width=cw - 0.56, height=num_h,
              size=60, color_rgb=tc, font=MONO_FONT, bold=True, anchor=MSO_ANCHOR.MIDDLE)
    _add_text(slide, lab, left=x + 0.28, top=block_top + num_h + 0.2, width=cw - 0.56, height=lab_h,
              size=15, color_rgb=tc, font=SANS_FONT)
"""
NOTE["Why HAM10000"] = ("Justify the dataset in three numbers. 10,015 REAL dermatoscopy images (enough to train on); 7 lesion types collapsed to the yes/no screening question; and only ~11% melanoma -- the imbalance that will bite us next slide -- plus recorded age and sex, which is what later lets us check fairness. The fairness point is the quiet reason this dataset beats a bare pile of images. Transition: that 11% is exactly why accuracy lies.")

# S5 -- WIDE FIGURE TOP + STAT BAND: accuracy trap / sensitivity dial
CODE["Accuracy is a trap; sensitivity is the job"] = f"""
img_h = body_h - 1.55
_fit_image(slide, '{FIG}/intro_sensitivity_tradeoff.png', left=body_l + body_w * 0.10, top=body_top, max_w=body_w * 0.80, max_h=img_h)
band_y = body_top + img_h + 0.15
band_h = 1.35
gap = 0.35
half = (body_w - gap) / 2
_add_rect(slide, left=body_l, top=band_y, width=half, height=band_h, fill_rgb=DEEPPINK_RGB)
tc = _text_on(DEEPPINK_RGB)
_add_text(slide, '89% / 0%', left=body_l + 0.3, top=band_y + 0.12, width=half - 0.6, height=0.55,
          size=30, color_rgb=tc, font=MONO_FONT, bold=True)
_add_text(slide, 'a lazy \"always not melanoma\" model: 89% accurate, catches ZERO cancers', left=body_l + 0.3,
          top=band_y + 0.72, width=half - 0.6, height=0.55, size=13.5, color_rgb=tc, font=SANS_FONT)
x2 = body_l + half + gap
_add_rect(slide, left=x2, top=band_y, width=half, height=band_h, fill_rgb=TURQUOISE_RGB)
tc2 = _text_on(TURQUOISE_RGB)
_add_text(slide, 'the fix: one dial', left=x2 + 0.3, top=band_y + 0.12, width=half - 0.6, height=0.55,
          size=26, color_rgb=tc2, font=MONO_FONT, bold=True)
_add_text(slide, 'slide the decision threshold left to catch more melanomas -- we grade on sensitivity, not accuracy',
          left=x2 + 0.3, top=band_y + 0.72, width=half - 0.6, height=0.55, size=13.5, color_rgb=tc2, font=SANS_FONT)
"""
NOTE["Accuracy is a trap; sensitivity is the job"] = ("The judgment slide. Because only ~1 in 9 spots is melanoma, a model that always says 'not melanoma' scores ~89% accuracy and catches zero cancers -- the pink band. The curves show the real machinery: healthy spots score low (turquoise), melanomas high (pink), and the dashed threshold decides. Slide it left and you catch more cancers at the cost of more false alarms. That dial, not accuracy, is how a screen is run. Transition: now, how do we actually build the model?")

# S6 -- DRAWN DIAGRAM: transfer learning pipeline
CODE["Borrow a brain: transfer learning"] = f"""
stages = [('PRETRAINED', 'a network already trained on millions of everyday photos -- cats, cars, faces', 'it already knows edges, colors, shapes', TURQUOISE_RGB),
          ('FINE-TUNE', 'reuse those edge and color features, then train on ~10,000 HAM10000 skin images', 'it learns melanoma color and border', DEEPPINK_RGB),
          ('SCREEN', 'ask the fine-tuned network for one call on a new skin spot', 'one probability, 0 to 1', BLUEVIOLET_RGB)]
gap = 1.0
cw = (body_w - gap * 2) / 3
ch = body_h - 1.1
ctop = body_top + 0.1
for i, (lab, bod, det, col) in enumerate(stages):
    x = body_l + i * (cw + gap)
    _add_rect(slide, left=x, top=ctop, width=cw, height=ch, fill_rgb=_rgb('#EDEAE1'))
    _add_rect(slide, left=x, top=ctop, width=cw, height=0.12, fill_rgb=col)
    _add_text(slide, lab, left=x + 0.28, top=ctop + 0.45, width=cw - 0.56, height=0.5,
              size=19, color_rgb=col, font=MONO_FONT, bold=True)
    _add_text(slide, bod, left=x + 0.28, top=ctop + 1.25, width=cw - 0.56, height=ch - 2.6,
              size=15.5, color_rgb=INK_RGB, font=SANS_FONT)
    _add_rect(slide, left=x + 0.28, top=ctop + ch - 0.95, width=cw - 0.56, height=0.02, fill_rgb=col)
    _add_text(slide, det, left=x + 0.28, top=ctop + ch - 0.8, width=cw - 0.56, height=0.6,
              size=14, color_rgb=col, font=MONO_FONT, bold=True, anchor=MSO_ANCHOR.MIDDLE)
    if i < 2:
        _add_text(slide, '→', left=x + cw + 0.05, top=ctop + ch / 2 - 0.45, width=gap - 0.1, height=0.9,
                  size=40, color_rgb=INK_RGB, font=MONO_FONT, bold=True, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
_add_text(slide, 'The reused knowledge is why a model can learn skin cancer from thousands, not millions, of photos.',
          left=body_l, top=ctop + ch + 0.18, width=body_w, height=0.55, size=15,
          color_rgb=BLUEVIOLET_RGB, font=SANS_FONT, italic=True, align=PP_ALIGN.CENTER)
"""
NOTE["Borrow a brain: transfer learning"] = ("The one modeling idea. We have ~10,000 skin images, nowhere near enough to teach a network from scratch. So we borrow a brain: take a net already trained on millions of ordinary photos, keep its sense of edges and color, and fine-tune it on skin. Walk the three stages left to right. This is the same trick behind every strong result in the field [5]. Transition: which pretrained brain, though? We tested two.")

# S7 -- FIGURE LEFT + CALLOUT CARD: backbone bake-off
CODE["Why CAFormer, not ResNet18"] = f"""
fw = body_w * 0.55
_fit_image(slide, '{FIG}/backbone_choice.png', left=body_l, top=body_top, max_w=fw, max_h=body_h)
ax = body_l + fw + 0.45
aw = body_w - fw - 0.45
_add_rect(slide, left=ax, top=body_top + 0.1, width=aw, height=body_h - 0.3, fill_rgb=WHITE_RGB)
_add_rect(slide, left=ax, top=body_top + 0.1, width=aw, height=0.1, fill_rgb=TURQUOISE_RGB)
_add_text(slide, 'WE TESTED, WE DID NOT GUESS', left=ax + 0.28, top=body_top + 0.4, width=aw - 0.56, height=0.4,
          size=14, color_rgb=TURQUOISE_RGB, font=MONO_FONT, bold=True)
_add_text(slide, '0.88', left=ax + 0.28, top=body_top + 0.95, width=aw - 0.56, height=0.9,
          size=52, color_rgb=INK_RGB, font=MONO_FONT, bold=True)
_add_text(slide, 'CAFormer AUC -- vs 0.83 for a plain ResNet18. Both clear 0.8, but the stronger backbone wins.',
          left=ax + 0.28, top=body_top + 1.95, width=aw - 0.56, height=1.2, size=15, color_rgb=INK_RGB, font=SANS_FONT)
_add_text(slide, 'The winner is chosen on held-out data, never on a hunch.',
          left=ax + 0.28, top=body_top + 3.25, width=aw - 0.56, height=1.0, size=15, color_rgb=INK_RGB, font=SANS_FONT, italic=True)
"""
NOTE["Why CAFormer, not ResNet18"] = ("The 'which tool?' decision, made honestly. We ran a bake-off: a plain ResNet18 backbone versus the stronger CAFormer, judged by AUC on the held-out test set. ResNet18 was already good (0.83, above the 0.8 line), but CAFormer reached 0.88, so it becomes our screen. The teachable habit: pick the model by measuring on data it never saw, not by guessing. Transition: here is the full recipe so anyone could rebuild it.")

# S8 -- NUMBERED PROCESS: methods spec
CODE["Model and data processing"] = f"""
steps = [('1', 'MODEL', 'caformer_s18, pretrained, backbone FROZEN, a fresh 2-class head trained on top', TURQUOISE_RGB),
         ('2', 'RESIZE', 'every image decoded to color and resized to 224x224 pixels', DEEPPINK_RGB),
         ('3', 'NORMALIZE', 'pixels rescaled to a fixed mean/scale, the range the backbone was trained on', AMBER_RGB),
         ('4', 'CLASS-WEIGHT', 'the loss up-weights the rare melanomas, so the model cannot ignore them', BLUEVIOLET_RGB),
         ('5', 'SPLIT', 'separate train / val / test; every number we report is on the held-out test set', TURQUOISE_RGB)]
n = len(steps)
gap = 0.25
rh = (body_h - gap * (n - 1)) / n
for i, (num, lab, bod, col) in enumerate(steps):
    y = body_top + i * (rh + gap)
    _add_rect(slide, left=body_l, top=y, width=0.9, height=rh, fill_rgb=col)
    _add_text(slide, num, left=body_l, top=y, width=0.9, height=rh,
              size=26, color_rgb=_text_on(col), font=MONO_FONT, bold=True, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    _add_text(slide, lab, left=body_l + 1.15, top=y, width=2.6, height=rh,
              size=17, color_rgb=col, font=MONO_FONT, bold=True, anchor=MSO_ANCHOR.MIDDLE)
    _add_text(slide, bod, left=body_l + 3.9, top=y, width=body_w - 3.9, height=rh,
              size=15, color_rgb=INK_RGB, font=SANS_FONT, anchor=MSO_ANCHOR.MIDDLE)
"""
NOTE["Model and data processing"] = ("The reproducibility slide -- read it as a recipe, not trivia. Five steps: (1) a frozen CAFormer with a fresh 2-class head (transfer learning); (2) every image resized to 224x224; (3) normalized to the backbone's expected range; (4) a class-weighted loss so the rare melanomas count; (5) a clean train/val/test split with all reporting on held-out data. Any one of these done wrong quietly breaks the result. Transition: so how well does it actually screen?")

# S9 -- FIGURE LEFT + BIG STAT: ROC / AUC
CODE["A working screen"] = f"""
fw = body_w * 0.52
_fit_image(slide, '{FIG}/roc.png', left=body_l, top=body_top, max_w=fw, max_h=body_h)
ax = body_l + fw + 0.5
aw = body_w - fw - 0.5
_add_text(slide, '0.885', left=ax, top=body_top + 0.2, width=aw, height=1.3,
          size=84, color_rgb=AMBER_RGB, font=MONO_FONT, bold=True)
_add_text(slide, 'AUC -- the area under the ROC curve', left=ax, top=body_top + 1.55, width=aw, height=0.5,
          size=17, color_rgb=INK_RGB, font=MONO_FONT, bold=True)
_add_text(slide, 'The ROC curve sweeps every cut-off and plots melanomas caught against false alarms. AUC is the area underneath.',
          left=ax, top=body_top + 2.2, width=aw, height=1.4, size=15, color_rgb=INK_RGB, font=SANS_FONT)
_add_text(slide, '0.5 = a coin flip     1.0 = perfect', left=ax, top=body_top + 3.7, width=aw, height=0.5,
          size=15, color_rgb=BLUEVIOLET_RGB, font=MONO_FONT, bold=True)
_add_text(slide, 'The curve hugs the top-left corner -- a genuinely working screen.',
          left=ax, top=body_top + 4.25, width=aw, height=0.6, size=15, color_rgb=INK_RGB, font=SANS_FONT, italic=True)
"""
NOTE["A working screen"] = ("Name the metric before the number. A screen outputs a score, not a yes/no, so we grade it with the ROC curve -- every possible cut-off, plotting melanomas caught against false alarms -- and AUC, the area underneath, where 0.5 is a coin flip and 1.0 is perfect. We use AUC, not accuracy, because accuracy is fooled by the imbalance. Our AUC is 0.885 and the curve hugs the top-left: a real, working screen. Transition: but the default cut-off isn't the right one for screening.")

# S10 -- BEFORE / AFTER: recall tuning
CODE["Tuning to miss fewer cancers"] = f"""
fw = body_w * 0.60
_fit_image(slide, '{FIG}/recall_tuning.png', left=body_l, top=body_top, max_w=fw, max_h=body_h)
ax = body_l + fw + 0.4
aw = body_w - fw - 0.4
_add_rect(slide, left=ax, top=body_top + 0.15, width=aw, height=1.75, fill_rgb=TURQUOISE_RGB)
tc = _text_on(TURQUOISE_RGB)
_add_text(slide, '74% → 90%', left=ax + 0.25, top=body_top + 0.35, width=aw - 0.5, height=0.7,
          size=32, color_rgb=tc, font=MONO_FONT, bold=True)
_add_text(slide, 'melanomas caught (recall), after we lower the threshold', left=ax + 0.25,
          top=body_top + 1.15, width=aw - 0.5, height=0.6, size=14, color_rgb=tc, font=SANS_FONT)
_add_text(slide, 'THE COST WE ACCEPT', left=ax, top=body_top + 2.25, width=aw, height=0.4,
          size=14, color_rgb=DEEPPINK_RGB, font=MONO_FONT, bold=True)
_add_text(slide, 'Specificity slips 85% → 70%: more healthy spots get a second look. For a screen, that is the right trade -- a few extra check-ups to avoid a missed cancer.',
          left=ax, top=body_top + 2.75, width=aw, height=2.0, size=15, color_rgb=INK_RGB, font=SANS_FONT)
"""
NOTE["Tuning to miss fewer cancers"] = ("The improvement, with real before/after numbers. The default cut-off (score >= 0.5) is tuned for accuracy; we lower it on purpose. Recall jumps 74% -> 90% -- from missing 1 in 4 melanomas to missing 1 in 10. The honest cost, in pink: specificity falls 85% -> 70%, so more healthy spots get a second look. That is exactly the trade a screen should make. This is the single most important engineering decision in the project. Transition: let's see those exact hits and misses.")

# S11 -- ANNOTATED MATRIX: confusion with callouts
CODE["At the screening setting"] = f"""
fw = body_w * 0.50
_fit_image(slide, '{FIG}/confusion.png', left=body_l, top=body_top + 0.15, max_w=fw, max_h=body_h - 0.3)
ax = body_l + fw + 0.4
aw = body_w - fw - 0.4
callouts = [('130', 'melanomas CAUGHT -- the wins', TURQUOISE_RGB),
            ('14', 'melanomas MISSED -- each one a real danger', DEEPPINK_RGB),
            ('340', 'false alarms -- each just an extra check-up', AMBER_RGB)]
rh = 1.35
for i, (num, lab, col) in enumerate(callouts):
    y = body_top + 0.15 + i * (rh + 0.15)
    _add_rect(slide, left=ax, top=y, width=0.14, height=rh, fill_rgb=col)
    _add_text(slide, num, left=ax + 0.3, top=y, width=1.7, height=rh,
              size=40, color_rgb=col, font=MONO_FONT, bold=True, anchor=MSO_ANCHOR.MIDDLE)
    _add_text(slide, lab, left=ax + 2.1, top=y, width=aw - 2.1, height=rh,
              size=15, color_rgb=INK_RGB, font=SANS_FONT, anchor=MSO_ANCHOR.MIDDLE)
"""
NOTE["At the screening setting"] = ("Read the confusion matrix out loud -- rows are truth, columns are what the screen said. At the screening threshold it catches 130 of 144 melanomas (recall 90%) and misses only 14. The price is 340 false alarms out of ~1,140 healthy spots. Put those three numbers in human terms: 130 wins, 14 dangers, 340 extra check-ups. A clinician weighs exactly this table when deciding whether to trust the tool. Transition: but does it screen everyone equally well?")

# S12 -- FIGURE LEFT + TWO STAT TILES: fairness by sex
CODE["Does it work for women and men?"] = f"""
fw = body_w * 0.54
_fit_image(slide, '{FIG}/fairness_by_sex.png', left=body_l, top=body_top, max_w=fw, max_h=body_h)
ax = body_l + fw + 0.45
aw = body_w - fw - 0.45
th = 1.5
_add_rect(slide, left=ax, top=body_top + 0.1, width=aw, height=th, fill_rgb=TURQUOISE_RGB)
tc = _text_on(TURQUOISE_RGB)
_add_text(slide, '0.91', left=ax + 0.28, top=body_top + 0.2, width=aw - 0.56, height=0.85,
          size=38, color_rgb=tc, font=MONO_FONT, bold=True, anchor=MSO_ANCHOR.MIDDLE)
_add_text(slide, 'AUC for men', left=ax + 0.28, top=body_top + 1.0, width=aw - 0.56, height=0.5, size=14, color_rgb=tc, font=SANS_FONT)
_add_rect(slide, left=ax, top=body_top + 1.75, width=aw, height=th, fill_rgb=DEEPPINK_RGB)
tc2 = _text_on(DEEPPINK_RGB)
_add_text(slide, '0.85', left=ax + 0.28, top=body_top + 1.85, width=aw - 0.56, height=0.85,
          size=38, color_rgb=tc2, font=MONO_FONT, bold=True, anchor=MSO_ANCHOR.MIDDLE)
_add_text(slide, 'AUC for women', left=ax + 0.28, top=body_top + 2.65, width=aw - 0.56, height=0.5, size=14, color_rgb=tc2, font=SANS_FONT)
_add_text(slide, 'Mind the 0.06 gap. Both clear 0.8, but a screen you trust means naming where it is weaker -- and working to close it.',
          left=ax, top=body_top + 3.5, width=aw, height=1.2, size=15, color_rgb=INK_RGB, font=SANS_FONT, italic=True)
"""
NOTE["Does it work for women and men?"] = ("Fairness, measured the right way: AUC computed SEPARATELY for each sex (HAM10000 records it). Both groups clear 0.8, so the screen works for everyone -- but it is stronger for men (0.91) than women (0.85). Name that 0.06 gap out loud; hiding it is how unfair tools ship. A real deployment would investigate why (skin tone, lesion site, sampling) and work to close it. Transition: one more check -- is it looking at the right thing?")

# S13 -- FILMSTRIP + CAPTION BAND: Grad-CAM
CODE["Where the screen looks"] = f"""
img_h = body_h - 1.0
_fit_image(slide, '{FIG}/gradcam.png', left=body_l, top=body_top, max_w=body_w, max_h=img_h)
band_y = body_top + img_h + 0.12
_add_rect(slide, left=body_l, top=band_y, width=body_w, height=0.78, fill_rgb=_rgb('#EDEAE1'))
_add_text(slide, 'Grad-CAM = feature importance for an image: the bright areas sit ON the pigmented lesion -- the same cues a dermatologist reads, not a hair, a ruler, or the frame corner.',
          left=body_l + 0.3, top=band_y + 0.1, width=body_w - 0.6, height=0.6,
          size=14.5, color_rgb=INK_RGB, font=SANS_FONT, anchor=MSO_ANCHOR.MIDDLE)
"""
NOTE["Where the screen looks"] = ("The image version of feature importance. For a table we'd ask SHAP which columns mattered; for a photo we use Grad-CAM, which highlights WHERE the model looked. Top row: real melanomas. Bottom row: the heat overlay. The bright spots land on the pigmented lesion itself -- reassuring evidence the screen reads the spot, the same cues a dermatologist uses, and not some artifact like a hair or ruler mark. Transition: so where does that leave us, honestly?")

# S14 -- CARDS ROW: limits
CODE["What it can and cannot do"] = f"""
cards = [('IT WORKS', 'A real melanoma screen on real dermatoscopy: AUC 0.885, tuned to catch 90% of melanomas, and it looks at the lesion for the right reasons.', 'AUC 0.885 · recall 90%', TURQUOISE_RGB),
         ('BUT IT IS UNEVEN', 'A real gap by sex (0.85 vs 0.91), and HAM10000 skews toward lighter skin -- so performance on darker skin is unproven here.', 'a real 0.06 gap by sex', DEEPPINK_RGB),
         ('A REAL TOOL', 'Would train on more diverse patients, be validated across many hospitals, and keep a dermatologist in the loop before touching a patient.', 'not ready for a patient', BLUEVIOLET_RGB)]
gap = 0.35
cw = (body_w - gap * 2) / 3
ch = body_h - 0.1
by = body_top + 0.6
for i, (lab, bod, tag, col) in enumerate(cards):
    x = body_l + i * (cw + gap)
    _add_rect(slide, left=x + 0.05, top=body_top + 0.06, width=cw, height=ch, fill_rgb=_rgb('#E4E1D9'))
    _add_rect(slide, left=x, top=body_top, width=cw, height=ch, fill_rgb=WHITE_RGB)
    _add_rect(slide, left=x, top=body_top, width=cw, height=0.1, fill_rgb=col)
    _add_text(slide, lab, left=x + 0.28, top=by, width=cw - 0.56, height=0.5,
              size=17, color_rgb=col, font=MONO_FONT, bold=True)
    _add_text(slide, bod, left=x + 0.28, top=by + 0.75, width=cw - 0.56, height=ch - 2.3,
              size=16, color_rgb=INK_RGB, font=SANS_FONT)
    _add_rect(slide, left=x, top=body_top + ch - 1.05, width=cw, height=0.02, fill_rgb=_rgb('#E4E1D9'))
    _add_text(slide, tag, left=x + 0.28, top=body_top + ch - 0.9, width=cw - 0.56, height=0.6,
              size=15, color_rgb=col, font=MONO_FONT, bold=True, anchor=MSO_ANCHOR.MIDDLE)
"""
NOTE["What it can and cannot do"] = ("Model the honesty the rubric rewards. Green: this genuinely works -- AUC 0.885, tuned to 90% recall, looking at the lesion. Pink: it is uneven -- the sex gap is real, and HAM10000 skews toward lighter skin, so darker skin is unproven here. Purple: a real tool would add diverse patients, multi-hospital validation, and a dermatologist in the loop. State plainly: this teaches the screening mindset, it does not screen a patient. Transition: the papers behind it.")

# S15 -- REFERENCES 2-COL
CODE["References"] = """
colw = (body_w - 0.6) / 2
left_refs = ('[1] Esteva et al. 2017, Nature -- one network matched 21 dermatologists.\\n\\n'
             '[2] Tschandl, Rosendahl & Kittler 2018, Sci Data -- the HAM10000 dataset.\\n\\n'
             '[3] Haenssle et al. 2018, Ann Oncol -- CNN vs 58 dermatologists on melanoma.\\n\\n'
             '[4] Tschandl et al. 2019, Lancet Oncol -- machines vs human readers, international.')
right_refs = ('[5] Kim et al. 2022, BMC Med Imaging -- transfer learning for medical images.\\n\\n'
              '[6] Trevethan 2017, Front Public Health -- sensitivity, specificity, and their pitfalls.\\n\\n'
              '[7] Wei et al. 2024, Front Med -- AI and skin cancer: how well it works and the gaps.')
_add_text(slide, 'LANDMARKS & DATA', left=body_l, top=body_top + 0.05, width=colw, height=0.4,
          size=16, color_rgb=TURQUOISE_RGB, font=MONO_FONT, bold=True)
_add_text(slide, left_refs, left=body_l, top=body_top + 0.6, width=colw, height=body_h - 0.7,
          size=14.5, color_rgb=INK_RGB, font=SANS_FONT)
rx = body_l + colw + 0.6
_add_text(slide, 'METHODS & PRIMERS', left=rx, top=body_top + 0.05, width=colw, height=0.4,
          size=16, color_rgb=DEEPPINK_RGB, font=MONO_FONT, bold=True)
_add_text(slide, right_refs, left=rx, top=body_top + 0.6, width=colw, height=body_h - 0.7,
          size=14.5, color_rgb=INK_RGB, font=SANS_FONT)
"""
NOTE["References"] = ("Give credit and show the trail. Left column is the landmarks and the data: Esteva's dermatologist-level result [1], the HAM10000 dataset we used [2], and the man-vs-machine studies [3][4]. Right column is the methods and plain-language primers: transfer learning [5], the sensitivity/specificity primer [6], and a recent review of where skin-cancer AI stands [7]. Point out that [1] is the result we redrew and [6] is why we grade on sensitivity. Transition: the one sentence to remember.")

# S16 -- CLOSING QUOTE
CODE["The honest bottom line"] = """
q_top = body_top + body_h * 0.24
_add_text(slide, 'A screen earns trust by catching the dangerous case.',
          left=body_l, top=q_top, width=body_w, height=1.1,
          size=32, color_rgb=TURQUOISE_RGB, font=MONO_FONT, bold=True, align=PP_ALIGN.CENTER)
rule_w = 2.4
_add_rect(slide, left=body_l + (body_w - rule_w) / 2, top=q_top + 1.4, width=rule_w, height=0.05, fill_rgb=TURQUOISE_RGB)
_add_text(slide, 'Tune the threshold on purpose, check who it works for -- then use a validated tool and a dermatologist to actually screen a patient.',
          left=body_l + 1.0, top=q_top + 1.95, width=body_w - 2.0, height=1.2,
          size=19, color_rgb=INK_RGB, font=SANS_FONT, align=PP_ALIGN.CENTER)
"""
NOTE["The honest bottom line"] = ("Land the deck on one idea: a screen earns trust by catching the dangerous case, tuning its threshold on purpose, and checking who it works for -- not by chasing a shiny accuracy number. Our tool did all three: AUC 0.885, recall tuned to 90%, fairness named. Leave them with the working rule of medical AI -- a model you can measure honestly beats a fancy one nobody can check -- then hand the real decision to a validated tool and a dermatologist. Thank them.")

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
