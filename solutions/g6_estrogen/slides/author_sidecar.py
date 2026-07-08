"""Hand-author the G6 freeform sidecar: theme bone, divider accents, and a DISTINCT body
composition per content slide (a causal-inference story). Loads the generated scaffold, mutates
params in place, writes back.   Run:  python author_sidecar.py
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
LJ = HERE / "g6_estrogen.md.layout.json"
FIG = str(HERE / "figures")  # absolute figure dir; _fit_image takes absolute paths fine

d = json.load(open(LJ))
d["theme"] = "bone"

# ---- section-divider accents (turquoise -> deeppink -> blueviolet -> amber -> turquoise) ----
DIV = {
    "Background": ("#40E0D0", "We open with the clinical scene so the statistics land later. Three ideas to plant: a real symptom (brain fog) that an objective test can miss, a drug whose answer flipped when researchers finally ran a trial, and the hidden reason why. Transition: meet the patient the number can't see."),
    "The data": ("#FF1493", "Now the raw material: a real national survey, why it is the right (and only) thing we have, and the one honest question a survey lets us ask. Transition: we can't randomize estrogen, so what CAN we ask?"),
    "The model": ("#8A2BE2", "The key teaching point of the whole project. A causal question forces an interpretable model, because you must READ how the estrogen effect changes as you adjust. A black box predicts but can't answer. Transition: prediction is not causation."),
    "The results": ("#F0C840", "The payoff, and the naive answer getting corrected in real time. The tempting headline, the 67% that was confounding, what actually matters, an honest predictive score, and fairness. Transition: start with the headline anyone would first compute."),
    "The takeaway": ("#40E0D0", "Good science states its own limits out loud. What a survey can and cannot do, the papers behind it, and the one habit to carry to any study you ever read. This is the section the rubric rewards most."),
}

CODE = {}
NOTE = {}

# S1 -- FULL-BLEED HERO: the doctor's-office scene
CODE["The visit no test explains"] = f"""
img_h = body_h - 0.5
_fit_image(slide, '{FIG}/intro_doctor_office.png', left=body_l, top=body_top, max_w=body_w, max_h=img_h)
_add_text(slide, 'She describes real, daily brain fog. The objective test says \"normal.\" Both can be true -- and the gap is the whole project.',
          left=body_l + 0.8, top=body_top + img_h + 0.12, width=body_w - 1.6, height=0.5,
          size=14, color_rgb=INK_RGB, font=SANS_FONT, italic=True, align=PP_ALIGN.CENTER)
"""
NOTE["The visit no test explains"] = ("Open concrete and human. A woman in her sixties says she loses words mid-sentence and forgets why she walked into a room -- real, daily brain fog. Then the objective digit-symbol memory test comes back NORMAL. The number on the chart cannot see what she is living, and women's symptoms get waved off as stress far too often. Do not mention models yet. Plant the idea that a test result and a patient's account can BOTH be real. Transition: measured across many women, how often do the two disagree?")

# S2 -- FIGURE RIGHT + ASIDE LEFT (three stacked label rows)
CODE["The tests say fine; she says not fine"] = f"""
fw = body_w * 0.56
_fit_image(slide, '{FIG}/intro_objective_vs_subjective.png', left=body_l + body_w - fw, top=body_top, max_w=fw, max_h=body_h)
ax = body_l
aw = body_w - fw - 0.45
rows = [('THE MISMATCH', 'Objective scores mostly land in the normal band, yet self-reported fog is high and scattered. The lines cross.', DEEPPINK_RGB),
        ('WHY IT MATTERS', 'A test that reads \"fine\" can miss a symptom a woman lives with every day.', BLUEVIOLET_RGB),
        ('THE STANCE', 'Take both as real evidence: the number AND the account. Neither is noise.', TURQUOISE_RGB)]
y = body_top + 0.15
for lab, bod, col in rows:
    _add_text(slide, lab, left=ax, top=y, width=aw, height=0.4, size=15, color_rgb=col, font=MONO_FONT, bold=True)
    _add_text(slide, bod, left=ax, top=y + 0.45, width=aw, height=1.0, size=15, color_rgb=INK_RGB, font=SANS_FONT)
    y += 1.55
"""
NOTE["The tests say fine; she says not fine"] = ("Generalize the opening scene with data. Each connecting line is one woman: her objective cognitive score on the left, her self-reported brain fog on the right. Most objective scores sit in the shaded 'normal' band, but the fog reports are high and scattered, and the lines cross -- the test and the symptom disagree. The mature analyst holds both: the number is real AND the reported experience is real. This is a redraw of the Maki 2024 idea, attributed. Transition: now the drug itself, and a twenty-year plot twist.")

# S3 -- FULL-BLEED HERO + 3 STAT CHIPS: the effect flip (WHIMS)
CODE["Twenty years of watching said estrogen helps"] = f"""
img_h = body_h - 1.15
_fit_image(slide, '{FIG}/intro_effect_flip.png', left=body_l + body_w * 0.13, top=body_top, max_w=body_w * 0.74, max_h=img_h)
chip_y = body_top + img_h + 0.18
chips = [('0.66x', 'the risk observational studies saw -- looks protective', TURQUOISE_RGB),
         ('2.05x', 'the risk the coin-flip trial found -- actual harm', DEEPPINK_RGB),
         ('2003', 'the year WHIMS overturned twenty years of headlines', BLUEVIOLET_RGB)]
gap = 0.3
cw = (body_w - gap * 2) / 3
for i, (num, lab, col) in enumerate(chips):
    x = body_l + i * (cw + gap)
    _add_text(slide, num, left=x, top=chip_y, width=cw, height=0.55, size=30, color_rgb=col, font=MONO_FONT, bold=True, align=PP_ALIGN.CENTER)
    _add_text(slide, lab, left=x, top=chip_y + 0.58, width=cw, height=0.5, size=12.5, color_rgb=INK_RGB, font=SANS_FONT, align=PP_ALIGN.CENTER)
"""
NOTE["Twenty years of watching said estrogen helps"] = ("This is the most important background slide -- read it slowly. Left bar, 'just watching': hormone users looked protected, dementia risk about two-thirds of non-users. Right bar, the coin-flip randomized trial (WHIMS): risk more than doubled. The amber arrow is the sign flip; the dashed line is 'no effect.' The point for a fifteen-year-old: the STUDY DESIGN, not the drug, decided the answer -- watching can lie, a coin flip cannot. This redraws Tang/Zandi vs Shumaker, attributed. Transition: so WHY does watching lie here?")

# S4 -- FIGURE LEFT + ASIDE CARD (paper surface, three rows)
CODE["Why watching lied: healthy-user bias"] = f"""
fw = body_w * 0.58
_fit_image(slide, '{FIG}/intro_confounding_triangle.png', left=body_l, top=body_top, max_w=fw, max_h=body_h)
ax = body_l + fw + 0.4
aw = body_w - fw - 0.4
_add_rect(slide, left=ax, top=body_top + 0.05, width=aw, height=body_h - 0.15, fill_rgb=PAPER_RGB)
_add_rect(slide, left=ax, top=body_top + 0.05, width=aw, height=0.1, fill_rgb=DEEPPINK_RGB)
rows = [('WHO CHOSE THE PILL', 'Users were richer, more active, and saw doctors more often to begin with.'),
        ('THE HIDDEN FACTOR', 'That same head start drives BOTH the drug use AND the better memory.'),
        ('SO THE DIRECT LINK', 'only looks like cause. It is a mirage -- confounding, a.k.a. healthy-user bias.')]
y = 0.4
for lab, bod in rows:
    _add_text(slide, lab, left=ax + 0.28, top=body_top + y, width=aw - 0.56, height=0.35, size=14, color_rgb=DEEPPINK_RGB, font=MONO_FONT, bold=True)
    _add_text(slide, bod, left=ax + 0.28, top=body_top + y + 0.4, width=aw - 0.56, height=1.0, size=14.5, color_rgb=INK_RGB, font=SANS_FONT)
    y += 1.5
"""
NOTE["Why watching lied: healthy-user bias"] = ("Name the villain: confounding, or 'healthy-user bias.' The women who chose hormone therapy were, on average, richer, more active, and saw doctors more. That hidden head start -- the pink box at the top -- pushes down on BOTH lower boxes: it makes them more likely to take the pill AND more likely to have good memory. So the dashed bottom arrow, 'takes pill -> better memory,' only LOOKS like cause. Think of ice-cream sales and shark attacks both rising in summer: the heat drives both, neither causes the other. This one diagram is the whole statistical idea of the talk. Transition: to catch it in our own data, we need the right dataset.")

# S6 -- 4 DESCRIPTOR CARDS ROW
CODE["Why NHANES"] = f"""
cards = [('739 WOMEN', 'all age 60+, drawn from the NHANES 2013-14 US national health survey', TURQUOISE_RGB),
         ('USED ESTROGEN?', 'whether she ever took female hormones -- the treatment we are studying', DEEPPINK_RGB),
         ('HER BACKGROUND', 'age, education, and family income -- the confounders we must adjust for', AMBER_RGB),
         ('MEMORY SCORE', 'a digit-symbol test; higher is sharper. We study who lands in the weaker half', BLUEVIOLET_RGB)]
gap = 0.3
cw = (body_w - gap * 3) / 4
ch = body_h - 0.1
by = body_top + 0.55
for i, (lab, bod, col) in enumerate(cards):
    x = body_l + i * (cw + gap)
    _add_rect(slide, left=x, top=body_top, width=cw, height=ch, fill_rgb=WHITE_RGB)
    _add_rect(slide, left=x, top=body_top, width=cw, height=0.1, fill_rgb=col)
    _add_text(slide, lab, left=x + 0.22, top=by, width=cw - 0.44, height=0.9, size=17, color_rgb=col, font=MONO_FONT, bold=True)
    _add_text(slide, bod, left=x + 0.22, top=by + 1.15, width=cw - 0.44, height=ch - 1.9, size=14.5, color_rgb=INK_RGB, font=SANS_FONT)
"""
NOTE["Why NHANES"] = ("Justify the dataset in four cards. 739 real women aged 60+ from NHANES, a genuine US government survey -- not something we invented. For each we know whether she used estrogen (the treatment), her age/education/income (the possible confounders we will adjust for), and a digit-symbol memory score where higher is sharper. That combination -- a treatment, an outcome, AND the confounders in one table -- is exactly what lets us test for the confounding trap. Transition: but a survey can't randomize, so what can it honestly tell us?")

# S7 -- PROSE + HIGHLIGHTED BOX + TWO CARDS
CODE["The honest question we can actually ask"] = f"""
_add_text(slide, 'We cannot flip a coin to assign estrogen in a survey -- so we cannot prove cause. Instead we ask a narrower, honest question:',
          left=body_l, top=body_top, width=body_w, height=0.6, size=18, color_rgb=INK_RGB, font=SANS_FONT)
_add_rect(slide, left=body_l, top=body_top + 0.8, width=body_w, height=1.15, fill_rgb=PAPER_RGB)
_add_rect(slide, left=body_l, top=body_top + 0.8, width=0.14, height=1.15, fill_rgb=DEEPPINK_RGB)
_add_text(slide, 'Does the estrogen effect SURVIVE once we account for age, education, and income?',
          left=body_l + 0.4, top=body_top + 0.8, width=body_w - 0.7, height=1.15,
          size=22, color_rgb=INK_RGB, font=MONO_FONT, bold=True, anchor=MSO_ANCHOR.MIDDLE)
gap = 0.4
cw = (body_w - gap) / 2
cy = body_top + 2.3
ch = body_h - 2.9
_add_card(slide, label='CRUDE', body='Estrogen\\'s effect all by itself -- ignoring everything else about the woman.',
          left=body_l, top=cy, width=cw, height=ch, accent_rgb=DEEPPINK_RGB)
_add_card(slide, label='ADJUSTED', body='Estrogen\\'s effect AFTER accounting for age, education, and income.',
          left=body_l + cw + gap, top=cy, width=cw, height=ch, accent_rgb=TURQUOISE_RGB)
_add_text(slide, 'If the effect shrinks from crude to adjusted, the headline was mostly confounding -- not cause.',
          left=body_l, top=cy + ch + 0.15, width=body_w, height=0.5, size=14, color_rgb=MUTED_RGB, font=SANS_FONT, italic=True)
"""
NOTE["The honest question we can actually ask"] = ("Set expectations honestly -- this is the intellectual pivot. We can't randomly assign estrogen in a survey, so we can't prove cause. But we CAN ask the boxed question: does the estrogen effect survive once we account for age, education, and income? We will measure estrogen's effect two ways -- CRUDE (alone) and ADJUSTED (after the confounders). If it shrinks, the crude headline was mostly confounding. That crude-versus-adjusted move is the entire method, and it needs a model whose effect we can actually read. Transition: which is exactly why the model choice matters so much.")

# S9 -- FIGURE LEFT + VERDICT ASIDE (two AUC chips + big statement): prediction vs causation
CODE["Prediction is not causation"] = f"""
fw = body_w * 0.55
_fit_image(slide, '{FIG}/intro_prediction_vs_causation.png', left=body_l, top=body_top + 0.1, max_w=fw, max_h=body_h - 0.2)
ax = body_l + fw + 0.4
aw = body_w - fw - 0.4
_add_text(slide, 'A CAUSAL QUESTION FORCES THE MODEL', left=ax, top=body_top + 0.1, width=aw, height=0.4,
          size=14, color_rgb=BLUEVIOLET_RGB, font=MONO_FONT, bold=True)
_add_text(slide, 'To answer it you must READ the estrogen coefficient and watch it change as you adjust. Only an interpretable model gives you that number.',
          left=ax, top=body_top + 0.6, width=aw, height=1.5, size=15.5, color_rgb=INK_RGB, font=SANS_FONT)
th = 0.95
_add_rect(slide, left=ax, top=body_top + 2.15, width=(aw - 0.25) / 2, height=th, fill_rgb=TURQUOISE_RGB)
tc = _text_on(TURQUOISE_RGB)
_add_text(slide, '0.78', left=ax, top=body_top + 2.22, width=(aw - 0.25) / 2, height=0.55, size=28, color_rgb=tc, font=MONO_FONT, bold=True, align=PP_ALIGN.CENTER)
_add_text(slide, 'logistic AUC', left=ax, top=body_top + 2.75, width=(aw - 0.25) / 2, height=0.3, size=12, color_rgb=tc, font=SANS_FONT, align=PP_ALIGN.CENTER)
x2 = ax + (aw - 0.25) / 2 + 0.25
_add_rect(slide, left=x2, top=body_top + 2.15, width=(aw - 0.25) / 2, height=th, fill_rgb=_rgb('#26262E'))
_add_text(slide, '0.76', left=x2, top=body_top + 2.22, width=(aw - 0.25) / 2, height=0.55, size=28, color_rgb=WHITE_RGB, font=MONO_FONT, bold=True, align=PP_ALIGN.CENTER)
_add_text(slide, 'black box AUC', left=x2, top=body_top + 2.75, width=(aw - 0.25) / 2, height=0.3, size=12, color_rgb=_rgb('#B8B4A6'), font=SANS_FONT, align=PP_ALIGN.CENTER)
_add_text(slide, 'The black box does not even predict better here -- and it can never hand you the effect size. So we choose logistic regression.',
          left=ax, top=body_top + 3.35, width=aw, height=1.3, size=15, color_rgb=INK_RGB, font=SANS_FONT, italic=True)
"""
NOTE["Prediction is not causation"] = ("The single most important teaching slide -- slow down here. Our question is causal, so we need a model whose estrogen effect we can READ as one number and watch shrink as we add confounders. A black box (CatBoost, TabPFN, random forest) can predict who scores low, but the estrogen effect is locked inside -- padlock on the left panel. Logistic regression is one readable equation whose coefficient IS the effect size. And the kicker, measured on our data: the black box (0.76) does not even predict better than logistic (0.78). So there is zero reason to reach for it. Prediction is not causation. Transition: here is the exact recipe.")

# S10 -- NUMBERED PROCESS: methods recipe
CODE["Why logistic regression, and the exact recipe"] = f"""
steps = [('1', 'MODEL', 'LogisticRegression -- one linear equation whose estrogen coefficient is the effect size', TURQUOISE_RGB),
         ('2', 'FEATURES', 'age, education, income, used_estrogen, years_estrogen -- each standardized so coefficients compare', DEEPPINK_RGB),
         ('3', 'THE CAUSAL MOVE', 'fit the estrogen effect TWICE: crude (alone) then adjusted (+ confounders); the change is the confounding', AMBER_RGB),
         ('4', 'PREDICTIVE SCORE', 'AUC via 5-fold cross-validation, so no woman is graded by a model that trained on her', BLUEVIOLET_RGB),
         ('5', 'FAIRNESS', 'a held-out train/test split, with accuracy reported separately for each group', TURQUOISE_RGB)]
n = len(steps)
gap = 0.22
rh = (body_h - gap * (n - 1)) / n
for i, (num, lab, bod, col) in enumerate(steps):
    y = body_top + i * (rh + gap)
    _add_rect(slide, left=body_l, top=y, width=0.9, height=rh, fill_rgb=col)
    _add_text(slide, num, left=body_l, top=y, width=0.9, height=rh, size=24, color_rgb=_text_on(col), font=MONO_FONT, bold=True, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    _add_text(slide, lab, left=body_l + 1.15, top=y, width=2.9, height=rh, size=16, color_rgb=col, font=MONO_FONT, bold=True, anchor=MSO_ANCHOR.MIDDLE)
    _add_text(slide, bod, left=body_l + 4.2, top=y, width=body_w - 4.2, height=rh, size=14.5, color_rgb=INK_RGB, font=SANS_FONT, anchor=MSO_ANCHOR.MIDDLE)
"""
NOTE["Why logistic regression, and the exact recipe"] = ("The reproducibility slide -- read it as a recipe, not trivia. (1) A logistic regression, one equation whose estrogen coefficient we read. (2) Five features, each standardized so the coefficients are comparable. (3) The causal move: fit the estrogen effect twice, crude then adjusted, and the difference is the confounding. (4) Predictive AUC by five-fold cross-validation so nothing is graded on data it trained on. (5) Fairness by a held-out split, accuracy per group. Any one of these done wrong quietly breaks the story. Transition: now the results, starting with the headline anyone would first compute.")

# S12 -- FIGURE LEFT + BIG STAT ASIDE (+8.1): raw means
CODE["The tempting headline"] = f"""
fw = body_w * 0.56
_fit_image(slide, '{FIG}/raw_means.png', left=body_l, top=body_top + 0.1, max_w=fw, max_h=body_h - 0.2)
ax = body_l + fw + 0.4
aw = body_w - fw - 0.4
_add_text(slide, '+8.1', left=ax, top=body_top + 0.3, width=aw, height=1.2, size=76, color_rgb=TURQUOISE_RGB, font=MONO_FONT, bold=True)
_add_text(slide, 'points -- the raw gap, users over non-users', left=ax, top=body_top + 1.7, width=aw, height=0.5, size=15, color_rgb=INK_RGB, font=MONO_FONT, bold=True)
_add_text(slide, '51.8 versus 43.7. Taken at face value, the exciting headline writes itself: \"estrogen users think better.\" It is the same clue the 1990s studies chased -- but we never checked whether the two groups were the same kind of people to begin with.',
          left=ax, top=body_top + 2.35, width=aw, height=2.2, size=15, color_rgb=INK_RGB, font=SANS_FONT)
"""
NOTE["The tempting headline"] = ("The tempting headline, exactly as anyone would first compute it. Average the memory score for users versus non-users: 51.8 versus 43.7, an eight-point head start for the estrogen group. If you stopped here you would write 'estrogen users think better' -- the same clue the 1990s observational studies chased. The whole point of the next slide is that we did NOT stop here; we asked who these women were. Transition: adjust for their background and watch what happens.")

# S13 -- BIG STAT LEFT (67%) + FIGURE RIGHT: confounding
CODE["Most of the gap was confounding"] = f"""
fw = body_w * 0.54
_fit_image(slide, '{FIG}/confounding.png', left=body_l + body_w - fw, top=body_top + 0.1, max_w=fw, max_h=body_h - 0.2)
ax = body_l
aw = body_w - fw - 0.45
_add_text(slide, '67%', left=ax, top=body_top + 0.35, width=aw, height=1.4, size=92, color_rgb=DEEPPINK_RGB, font=MONO_FONT, bold=True)
_add_text(slide, 'of the apparent effect was confounding', left=ax, top=body_top + 1.85, width=aw, height=0.5, size=16, color_rgb=INK_RGB, font=MONO_FONT, bold=True)
_add_text(slide, 'The crude effect (-0.33) shrank to just -0.11 once we adjusted for age, education, and income. That missing two-thirds was never the estrogen -- it was that users were healthier and better-off to start with.',
          left=ax, top=body_top + 2.5, width=aw, height=2.0, size=15.5, color_rgb=INK_RGB, font=SANS_FONT)
"""
NOTE["Most of the gap was confounding"] = ("The punchline of the entire project. Same data, honest test. Crude effect: about -0.33 (negative means 'less likely to score low,' i.e. looks protective). Adjusted for age, education, and income: it collapses to -0.11. That is a 67% shrink -- two-thirds of the 'benefit' was never the estrogen, it was that users were healthier and better-off to begin with. This is confounding caught in the act, a miniature of exactly why WHIMS overturned the observational headlines. If they remember one number from this talk, make it 67%. Transition: the same model shows what DID matter.")

# S14 -- FIGURE LEFT + RANKED BULLETS ASIDE: feature importance
CODE["What actually predicts cognition"] = f"""
fw = body_w * 0.58
_fit_image(slide, '{FIG}/feature_importance.png', left=body_l, top=body_top + 0.1, max_w=fw, max_h=body_h - 0.2)
ax = body_l + fw + 0.4
aw = body_w - fw - 0.4
_add_text(slide, 'FROM THE SAME MODEL', left=ax, top=body_top + 0.1, width=aw, height=0.4, size=14, color_rgb=AMBER_RGB, font=MONO_FONT, bold=True)
ranks = [('education', 'the biggest driver', TURQUOISE_RGB),
         ('age', 'close behind', TURQUOISE_RGB),
         ('income', 'still matters', TURQUOISE_RGB),
         ('estrogen', 'the smallest bar', DEEPPINK_RGB)]
y = body_top + 0.65
for name, note, col in ranks:
    _add_text(slide, name, left=ax, top=y, width=aw * 0.5, height=0.42, size=17, color_rgb=col, font=MONO_FONT, bold=True)
    _add_text(slide, note, left=ax + aw * 0.5, top=y, width=aw * 0.5, height=0.42, size=13.5, color_rgb=INK_RGB, font=SANS_FONT, anchor=MSO_ANCHOR.MIDDLE)
    y += 0.62
_add_text(slide, 'Feature importance is free from an interpretable model: it is the standardized coefficients. Prediction and causation agree, and neither points at the pill.',
          left=ax, top=y + 0.15, width=aw, height=1.6, size=15, color_rgb=INK_RGB, font=SANS_FONT, italic=True)
"""
NOTE["What actually predicts cognition"] = ("Feature importance, and it costs us nothing because we chose an interpretable model -- these bars ARE the standardized coefficients of the very model that answered the causal question. Education and age dominate, income follows, and used_estrogen is the SMALLEST bar (in pink). So the same model shows visually what the adjustment proved: once you know a woman's background, estrogen barely moves cognition. Prediction and causation point the same way, and neither points at the pill. Transition: fair question -- can this model even predict at all?")

# S15 -- BIG STAT, NO FIGURE: predictive AUC 0.78
CODE["Can it even predict?"] = f"""
_add_text(slide, '0.78', left=body_l, top=body_top + 0.35, width=body_w * 0.5, height=2.2, size=150, color_rgb=TURQUOISE_RGB, font=MONO_FONT, bold=True)
ax = body_l + body_w * 0.52
aw = body_w - body_w * 0.52
_add_text(slide, 'predictive AUC', left=ax, top=body_top + 0.55, width=aw, height=0.6, size=26, color_rgb=INK_RGB, font=MONO_FONT, bold=True)
_add_text(slide, '5-fold cross-validation. 0.5 would be a coin flip, so 0.78 is a real signal -- but it is driven by age and education, not estrogen.',
          left=ax, top=body_top + 1.25, width=aw, height=1.5, size=16, color_rgb=INK_RGB, font=SANS_FONT)
_add_rect(slide, left=body_l, top=body_top + body_h - 1.1, width=body_w, height=0.9, fill_rgb=PAPER_RGB)
_add_text(slide, 'An honest secondary number -- and a reminder: a black box scored 0.76 here, no better, and could give us no effect size at all.',
          left=body_l + 0.35, top=body_top + body_h - 1.1, width=body_w - 0.7, height=0.9,
          size=15, color_rgb=BLUEVIOLET_RGB, font=SANS_FONT, italic=True, anchor=MSO_ANCHOR.MIDDLE)
"""
NOTE["Can it even predict?"] = ("Be honest about the secondary number. The deliverable is the causal answer, but it is fair to ask how well the model predicts low cognition at all. Five-fold cross-validated AUC is 0.78 -- 0.5 would be a coin flip, so this is a genuine signal, and it comes mostly from age and education, not estrogen. The bottom band closes the loop with the model slide: a black box scored 0.76 here, no better, and it could not hand us an effect size anyway. So the readable model wins on both counts. Transition: one more audit any people-model needs -- fairness.")

# S16 -- FIGURE LEFT + TWO STAT TILES: fairness
CODE["Does it work for both groups?"] = f"""
fw = body_w * 0.56
_fit_image(slide, '{FIG}/fairness_by_group.png', left=body_l, top=body_top, max_w=fw, max_h=body_h)
ax = body_l + fw + 0.45
aw = body_w - fw - 0.45
th = 1.45
_add_rect(slide, left=ax, top=body_top + 0.1, width=aw, height=th, fill_rgb=TURQUOISE_RGB)
tc = _text_on(TURQUOISE_RGB)
_add_text(slide, '0.73', left=ax + 0.28, top=body_top + 0.2, width=aw - 0.56, height=0.8, size=36, color_rgb=tc, font=MONO_FONT, bold=True, anchor=MSO_ANCHOR.MIDDLE)
_add_text(slide, 'accuracy for estrogen-USERS', left=ax + 0.28, top=body_top + 0.98, width=aw - 0.56, height=0.5, size=13.5, color_rgb=tc, font=SANS_FONT)
_add_rect(slide, left=ax, top=body_top + 1.7, width=aw, height=th, fill_rgb=MUTED_RGB)
tc2 = _text_on(MUTED_RGB)
_add_text(slide, '0.65', left=ax + 0.28, top=body_top + 1.8, width=aw - 0.56, height=0.8, size=36, color_rgb=tc2, font=MONO_FONT, bold=True, anchor=MSO_ANCHOR.MIDDLE)
_add_text(slide, 'accuracy for NON-users', left=ax + 0.28, top=body_top + 2.58, width=aw - 0.56, height=0.5, size=13.5, color_rgb=tc2, font=SANS_FONT)
_add_text(slide, 'A tool reliably right for one group and shakier for another can quietly do harm if a clinic trusts it equally. Naming the gap is the honest move.',
          left=ax, top=body_top + 3.4, width=aw, height=1.2, size=14.5, color_rgb=INK_RGB, font=SANS_FONT, italic=True)
"""
NOTE["Does it work for both groups?"] = ("A fairness audit, the habit from earlier this week. We trained one model to predict low cognition, then split its accuracy by group: 0.73 for estrogen-users, 0.65 for non-users. A model can look fine overall yet work unequally across groups -- and a clinic that trusts it equally could quietly do more harm to the group it is worse at. The mature move is not to hide the gap; it is to name it and, in a real deployment, work to close it. Transition: so what can this project honestly claim, and what can it not?")

# S18 -- 3 CARDS ROW: limits
CODE["What a survey can and cannot do"] = f"""
cards = [('WHAT IT CAN DO', 'Show a tempting headline is mostly a mirage, and that adjusting for background shrinks the effect by 67%.', 'the mirage, exposed', TURQUOISE_RGB),
         ('WHAT IT CANNOT DO', 'Prove cause. We only corrected for confounders we measured and thought of; something unrecorded could still be doing the work.', 'a hint, not a proof', DEEPPINK_RGB),
         ('WHAT SETTLES IT', 'A randomized trial: a coin flip, not the patient, assigns the drug, so the groups start the same. That is why WHIMS could overturn 20 years of headlines.', 'only an experiment', BLUEVIOLET_RGB)]
gap = 0.35
cw = (body_w - gap * 2) / 3
ch = body_h - 0.1
by = body_top + 0.55
for i, (lab, bod, tag, col) in enumerate(cards):
    x = body_l + i * (cw + gap)
    _add_rect(slide, left=x, top=body_top, width=cw, height=ch, fill_rgb=WHITE_RGB)
    _add_rect(slide, left=x, top=body_top, width=cw, height=0.1, fill_rgb=col)
    _add_text(slide, lab, left=x + 0.26, top=by, width=cw - 0.52, height=0.5, size=16, color_rgb=col, font=MONO_FONT, bold=True)
    _add_text(slide, bod, left=x + 0.26, top=by + 0.7, width=cw - 0.52, height=ch - 2.15, size=15, color_rgb=INK_RGB, font=SANS_FONT)
    _add_rect(slide, left=x, top=body_top + ch - 1.0, width=cw, height=0.02, fill_rgb=_rgb('#E4E1D9'))
    _add_text(slide, tag, left=x + 0.26, top=body_top + ch - 0.85, width=cw - 0.52, height=0.6, size=14, color_rgb=col, font=MONO_FONT, bold=True, anchor=MSO_ANCHOR.MIDDLE)
"""
NOTE["What a survey can and cannot do"] = ("Be scrupulously honest about scope -- this is what earns trust. CAN DO: show a tempting headline is mostly a mirage and that adjusting shrinks it by 67%. CANNOT DO: prove cause -- we can only correct for confounders we measured and thought of; something unrecorded could still be doing the work. SETTLES IT: a randomized trial, where a coin flip (not the patient) assigns the drug, so the groups start identical. That is why WHIMS could do what twenty years of watching could not. State the line plainly: this teaches the reasoning, it does not settle the medicine. Transition: the sources.")

# S19 -- REFERENCES 2-COL
CODE["References"] = """
colw = (body_w - 0.6) / 2
left_refs = ('[1] Tang et al. 1996, Lancet -- observational; users got Alzheimer\\'s less/later.\\n\\n'
             '[2] Zandi et al. 2002, JAMA (Cache County) -- observational; long-term users lower risk.\\n\\n'
             '[3] Shumaker et al. 2003, JAMA -- WHIMS RCT; hormones DOUBLED dementia risk in 65+.\\n\\n'
             '[4] Rapp et al. 2003, JAMA -- WHIMS RCT; no cognitive benefit, scores trended worse.')
right_refs = ('[5] Wharton et al. 2009, Maturitas -- healthy-user bias; users start out healthier.\\n\\n'
              '[6] Vandenbroucke 2009, Lancet -- why observational studies and RCTs disagreed.\\n\\n'
              '[7] Maki 2024, Harvard Health -- menopause brain fog vs. normal test scores.\\n\\n'
              '[8] Chen & Shafir 2025, Harvard Health -- the dismissal of women\\'s symptoms.')
_add_text(slide, 'OBSERVATIONAL & TRIAL', left=body_l, top=body_top + 0.05, width=colw, height=0.4, size=16, color_rgb=TURQUOISE_RGB, font=MONO_FONT, bold=True)
_add_text(slide, left_refs, left=body_l, top=body_top + 0.6, width=colw, height=body_h - 0.7, size=14, color_rgb=INK_RGB, font=SANS_FONT)
rx = body_l + colw + 0.6
_add_text(slide, 'BIAS & SYMPTOMS', left=rx, top=body_top + 0.05, width=colw, height=0.4, size=16, color_rgb=DEEPPINK_RGB, font=MONO_FONT, bold=True)
_add_text(slide, right_refs, left=rx, top=body_top + 0.6, width=colw, height=body_h - 0.7, size=14, color_rgb=INK_RGB, font=SANS_FONT)
"""
NOTE["References"] = ("Give credit and show the trail, in the order of the story. Left column: the two observational studies that raised the exciting clue [1,2], then the WHIMS randomized results that overturned it [3,4]. Right column: the healthy-user-bias explanation [5,6] and the brain-fog / dismissed-symptoms pieces [7,8]. All were checked against PubMed or the journal page. Point out that [3] is the trial we redrew and [5,6] are the source of the confounding idea. Transition: the one habit to take with you.")

# S20 -- CLOSING QUOTE
CODE["The one habit to carry with you"] = """
q_top = body_top + body_h * 0.16
_add_rect(slide, left=body_l, top=q_top, width=0.16, height=1.5, fill_rgb=TURQUOISE_RGB)
_add_text(slide, 'Suspect the chooser, not the treatment.',
          left=body_l + 0.5, top=q_top - 0.15, width=body_w - 0.6, height=1.5,
          size=38, color_rgb=INK_RGB, font=MONO_FONT, bold=True)
_add_text(slide, 'When a group that CHOSE a treatment looks healthier, the head start usually belongs to the person, not the pill -- and only a coin-flip experiment can promote a correlation to a cause.',
          left=body_l + 0.5, top=q_top + 1.7, width=body_w - 1.0, height=1.3, size=19, color_rgb=INK_RGB, font=SANS_FONT)
_add_text(slide, 'And take the patient\\'s account seriously -- even when the test says \"fine.\"',
          left=body_l + 0.5, top=q_top + 3.1, width=body_w - 1.0, height=0.7, size=19, color_rgb=DEEPPINK_RGB, font=MONO_FONT, bold=True)
"""
NOTE["The one habit to carry with you"] = ("Close on the transferable lesson, not the estrogen specifics. Suspect the chooser, not the treatment: whenever a group that CHOSE something looks healthier, the head start usually belongs to the person, and only a coin-flip experiment can turn a correlation into a cause. That is why we insisted on a model we could read, and why a survey can hint but never prove. And -- because this is medicine, about real women -- take the patient's account seriously even when the test says 'fine.' That pair, statistical humility plus listening, is the whole point of the project. Thank them.")

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
