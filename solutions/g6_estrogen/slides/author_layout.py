"""Hand-author the G6 freeform sidecar: bespoke code + agent provenance + speaker notes.

Run with the outset python. Edits g6_estrogen.md.layout.json in place: sets theme=bone,
divider accents, and writes a per-slide `code` body + rich `notes` for every content slide.
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
FIG = str(HERE / "figures") + "/"          # absolute figure dir (sandbox add_picture needs a real path)
LJ = HERE / "g6_estrogen.md.layout.json"

# ---- helpers to keep the emitted snippets tidy -----------------------------
def img(name, *, l="body_l", t="body_top+0.05", w="body_w", h="body_h-0.10"):
    return f"_fit_image(slide, {json.dumps(FIG + name)}, left={l}, top={t}, max_w={w}, max_h={h})"


# ---- per-slide body code ---------------------------------------------------
# Slide 1: the detective story -- two cards + a bottom takeaway line.
S1 = f"""
gap = 0.4
cw = (body_w - gap) / 2
ch = body_h - 1.15
_add_card(slide, label="THE EXCITING CLUE  (1996-2002)",
          body="Studies that simply WATCHED women found that estrogen users developed Alzheimer's less often and later [1, 2]. It looked like solid proof the pills protected the brain.",
          left=body_l, top=body_top, width=cw, height=ch, accent_rgb=TURQUOISE_RGB)
_add_card(slide, label="THE PLOT TWIST  (2003)",
          body="Then the WHIMS trial let a COIN FLIP decide who got hormones. The benefit vanished, and dementia risk actually ROSE in women 65+ [3, 4].",
          left=body_l+cw+gap, top=body_top, width=cw, height=ch, accent_rgb=DEEPPINK_RGB)
_add_rect(slide, left=body_l, top=body_top+ch+0.28, width=body_w, height=0.62, fill_rgb=AMBER_RGB)
_add_text(slide, "Same drug, opposite answer. The rest of this talk is one question: HOW?",
          left=body_l+0.25, top=body_top+ch+0.28, width=body_w-0.5, height=0.62,
          size=18, color_rgb=INK_RGB, font=MONO_FONT, bold=True, anchor=MSO_ANCHOR.MIDDLE)
"""

# Slide 2: effect-flip figure, full-bleed hero.
S2 = img("intro_effect_flip.png") + """
_add_text(slide, "Just watching (turquoise) said hormones protect; the coin-flip trial (pink) said they harm.",
          left=body_l, top=body_bottom-0.30, width=body_w, height=0.30,
          size=13, color_rgb=MUTED_RGB, font=SANS_FONT, italic=True, align=PP_ALIGN.CENTER)
"""

# Slide 3: confounding triangle on the left, explanatory aside on the right.
S3 = f"""
fw = body_w * 0.60
{img("intro_confounding_triangle.png", w="fw", h="body_h-0.2", t="body_top+0.10")}
px = body_l + fw + 0.35
pw = body_w - fw - 0.35
_add_rect(slide, left=px, top=body_top+0.05, width=pw, height=body_h-0.15, fill_rgb=PAPER_RGB)
_add_rect(slide, left=px, top=body_top+0.05, width=pw, height=0.10, fill_rgb=DEEPPINK_RGB)
rows = [
  ("Who chose the pill?", "Users were richer, more active, and saw doctors more often to begin with [5]."),
  ("The hidden factor", "That same head start drives BOTH the drug use AND the good memory."),
  ("So the bottom link", "only LOOKS like cause. It is a mirage -- confounding, a.k.a. healthy-user bias [6]."),
]
y = 0.36
for lab, bod in rows:
    _add_text(slide, lab, left=px+0.25, top=body_top+y, width=pw-0.5, height=0.32,
              size=15, color_rgb=DEEPPINK_RGB, font=MONO_FONT, bold=True)
    _add_text(slide, bod, left=px+0.25, top=body_top+y+0.36, width=pw-0.5, height=1.0,
              size=13, color_rgb=INK_RGB, font=SANS_FONT)
    y += 1.55
"""

# Slide 4: ice-cream / sharks analogy -- no figure, two mapped cards + takeaway.
S4 = f"""
_add_text(slide, "On hot days, ice-cream sales and shark attacks BOTH rise together. Does ice cream summon sharks?",
          left=body_l, top=body_top, width=body_w, height=0.55,
          size=17, color_rgb=INK_RGB, font=SANS_FONT)
gap = 0.4
cw = (body_w - gap) / 2
ct = body_top + 0.75
ch = body_h - 1.95
_add_card(slide, label="THE ANALOGY",
          body="No -- the hot SUMMER WEATHER drives both: heat sells ice cream and sends people into the ocean. Neither one causes the other.",
          left=body_l, top=ct, width=cw, height=ch, accent_rgb=BLUEVIOLET_RGB)
_add_card(slide, label="OUR VERSION",
          body="Estrogen use and good memory rise together. Being HEALTHIER to start with drives both -- the pill may summon nothing.",
          left=body_l+cw+gap, top=ct, width=cw, height=ch, accent_rgb=TURQUOISE_RGB)
_add_text(slide, "When two things move together, hunt for the hidden third factor before you believe one caused the other.",
          left=body_l, top=ct+ch+0.22, width=body_w, height=0.55,
          size=15, color_rgb=DEEPPINK_RGB, font=MONO_FONT, bold=True)
"""

# Slide 6: the data, four descriptor cards.
S6 = f"""
gap = 0.35
cw = (body_w - 3*gap) / 4
ch = body_h - 0.4
cards = [
  ("739 WOMEN", "all age 60+, drawn from the NHANES 2013-14 US national health survey.", TURQUOISE_RGB),
  ("USED ESTROGEN?", "whether she ever took female hormones -- yes or no, self-reported.", DEEPPINK_RGB),
  ("HER BACKGROUND", "age, education, and family income -- the confounders we must check.", AMBER_RGB),
  ("MEMORY SCORE", "a digit-symbol test; higher = sharper. We predict the weaker-scoring half.", BLUEVIOLET_RGB),
]
for i, (lab, bod, acc) in enumerate(cards):
    x = body_l + i*(cw+gap)
    _add_card(slide, label=lab, body=bod, left=x, top=body_top+0.1,
              width=cw, height=ch, accent_rgb=acc)
"""

# Slide 7: the honest question -- prose + a highlighted line + two chips.
S7 = f"""
_add_text(slide, "We cannot randomly assign estrogen in a survey. So we ask a narrower, honest question:",
          left=body_l, top=body_top, width=body_w, height=0.55, size=18, color_rgb=INK_RGB, font=SANS_FONT)
_add_rect(slide, left=body_l, top=body_top+0.75, width=body_w, height=1.15, fill_rgb=PAPER_RGB)
_add_rect(slide, left=body_l, top=body_top+0.75, width=0.14, height=1.15, fill_rgb=DEEPPINK_RGB)
_add_text(slide, "Does the estrogen effect SURVIVE once we account for age, education, and income?",
          left=body_l+0.4, top=body_top+0.75, width=body_w-0.7, height=1.15,
          size=22, color_rgb=INK_RGB, font=MONO_FONT, bold=True, anchor=MSO_ANCHOR.MIDDLE)
gap = 0.4
cw = (body_w - gap) / 2
cy = body_top + 2.25
ch = body_h - 2.85
_add_card(slide, label="CRUDE",
          body="Estrogen's effect all by itself -- ignoring everything else about the woman.",
          left=body_l, top=cy, width=cw, height=ch, accent_rgb=DEEPPINK_RGB)
_add_card(slide, label="ADJUSTED",
          body="Estrogen's effect AFTER accounting for age, education, and income.",
          left=body_l+cw+gap, top=cy, width=cw, height=ch, accent_rgb=TURQUOISE_RGB)
_add_text(slide, "If the effect shrinks from crude to adjusted, the headline was mostly confounding -- not cause.",
          left=body_l, top=cy+ch+0.18, width=body_w, height=0.5,
          size=14, color_rgb=MUTED_RGB, font=SANS_FONT, italic=True)
"""

# Slide 9: raw means figure + big stat aside.
S9 = f"""
fw = body_w * 0.56
{img("raw_means.png", w="fw", h="body_h-0.2", t="body_top+0.10")}
px = body_l + fw + 0.4
pw = body_w - fw - 0.4
_add_text(slide, "+8.1", left=px, top=body_top+0.3, width=pw, height=1.2,
          size=66, color_rgb=TURQUOISE_RGB, font=MONO_FONT, bold=True)
_add_text(slide, "points -- the raw gap, users over non-users", left=px, top=body_top+1.55,
          width=pw, height=0.5, size=15, color_rgb=INK_RGB, font=MONO_FONT, bold=True)
_add_text(slide, "51.8 versus 43.7. Taken at face value, the exciting headline writes itself: 'estrogen users think better.' But we never checked whether the two groups were the same kind of people to begin with.",
          left=px, top=body_top+2.2, width=pw, height=2.2, size=15, color_rgb=INK_RGB, font=SANS_FONT)
"""

# Slide 10: confounding figure + big 67% stat aside.
S10 = f"""
fw = body_w * 0.56
{img("confounding.png", w="fw", h="body_h-0.2", t="body_top+0.10")}
px = body_l + fw + 0.4
pw = body_w - fw - 0.4
_add_text(slide, "67%", left=px, top=body_top+0.3, width=pw, height=1.3,
          size=76, color_rgb=DEEPPINK_RGB, font=MONO_FONT, bold=True)
_add_text(slide, "of the effect was confounding", left=px, top=body_top+1.65,
          width=pw, height=0.5, size=15, color_rgb=INK_RGB, font=MONO_FONT, bold=True)
_add_text(slide, "The crude effect (-0.33) shrank to just -0.11 once we adjusted for age, education, and income. That missing two-thirds was never the estrogen -- it was that users were healthier and better-off to start with.",
          left=px, top=body_top+2.3, width=pw, height=2.2, size=15, color_rgb=INK_RGB, font=SANS_FONT)
"""

# Slide 11: fairness figure + aside bullets.
S11 = f"""
fw = body_w * 0.56
{img("fairness_by_group.png", w="fw", h="body_h-0.2", t="body_top+0.10")}
px = body_l + fw + 0.4
pw = body_w - fw - 0.4
rows = [
  ("0.73", "test accuracy for estrogen-USERS", TURQUOISE_RGB),
  ("0.65", "test accuracy for NON-users", MUTED_RGB),
]
y = 0.2
for num, lab, col in rows:
    _add_text(slide, num, left=px, top=body_top+y, width=pw, height=0.75,
              size=44, color_rgb=col, font=MONO_FONT, bold=True)
    _add_text(slide, lab, left=px, top=body_top+y+0.8, width=pw, height=0.4,
              size=14, color_rgb=INK_RGB, font=MONO_FONT, bold=True)
    y += 1.5
_add_text(slide, "A tool that is reliably right for one group and shakier for another can quietly do harm if a clinic trusts it equally. Naming the gap out loud is the honest move.",
          left=px, top=body_top+y+0.1, width=pw, height=1.6, size=14, color_rgb=INK_RGB, font=SANS_FONT)
"""

# Slide 13: objective-vs-subjective figure + aside.
S13 = f"""
fw = body_w * 0.58
{img("intro_objective_vs_subjective.png", w="fw", h="body_h-0.2", t="body_top+0.10")}
px = body_l + fw + 0.4
pw = body_w - fw - 0.4
_add_rect(slide, left=px, top=body_top+0.05, width=pw, height=body_h-0.15, fill_rgb=PAPER_RGB)
_add_rect(slide, left=px, top=body_top+0.05, width=pw, height=0.10, fill_rgb=BLUEVIOLET_RGB)
rows = [
  ("The mismatch", "Many women report real, daily brain fog even when the objective test looks perfectly normal [7]."),
  ("Why it matters", "Women's symptoms are too often waved off as stress or emotion instead of investigated [8]."),
  ("Hold both ideas", "The confounding is real AND the patient's account is real evidence -- not noise to dismiss."),
]
y = 0.36
for lab, bod in rows:
    _add_text(slide, lab, left=px+0.25, top=body_top+y, width=pw-0.5, height=0.32,
              size=15, color_rgb=BLUEVIOLET_RGB, font=MONO_FONT, bold=True)
    _add_text(slide, bod, left=px+0.25, top=body_top+y+0.36, width=pw-0.5, height=1.05,
              size=13, color_rgb=INK_RGB, font=SANS_FONT)
    y += 1.55
"""

# Slide 14: honest limits -- three cards can/cannot/settles.
S14 = f"""
gap = 0.4
cw = (body_w - 2*gap) / 3
ch = body_h - 0.4
cards = [
  ("WHAT IT CAN DO", "Show a tempting headline is mostly a mirage, and that adjusting shrinks the effect by ~67%.", TURQUOISE_RGB),
  ("WHAT IT CANNOT DO", "Prove cause. We can only correct for confounders we actually measured and thought of.", DEEPPINK_RGB),
  ("WHAT SETTLES IT", "A randomized trial: a coin flip, not the patient, assigns the drug. That is why WHIMS could overturn 20 years of headlines.", AMBER_RGB),
]
for i, (lab, bod, acc) in enumerate(cards):
    x = body_l + i*(cw+gap)
    _add_card(slide, label=lab, body=bod, left=x, top=body_top+0.1,
              width=cw, height=ch, accent_rgb=acc)
"""

# Slide 15: references, two columns.
REFS_L = [
  "[1] Tang et al. 1996, Lancet -- observational; estrogen users got Alzheimer's less/later.",
  "[2] Zandi et al. 2002, JAMA (Cache County) -- observational; long-term users lower risk.",
  "[3] Shumaker et al. 2003, JAMA -- WHIMS RCT; hormones DOUBLED dementia risk in 65+.",
  "[4] Rapp et al. 2003, JAMA -- WHIMS RCT; no cognitive benefit, scores trended worse.",
]
REFS_R = [
  "[5] Wharton et al. 2009, Maturitas -- healthy-user bias; users start out healthier.",
  "[6] Vandenbroucke 2009, Lancet -- why observational studies and RCTs disagreed.",
  "[7] Maki 2024, Harvard Health -- menopause 'brain fog' vs. normal test scores.",
  "[8] Chen & Shafir 2025, Harvard Health -- the dismissal of women's symptoms.",
]
def refcol(items, xexpr):
    out = [f"y = 0.15"]
    for it in items:
        out.append(f"_add_text(slide, {json.dumps(it)}, left={xexpr}, top=body_top+y, "
                   f"width=(body_w-0.4)/2, height=1.0, size=13, color_rgb=INK_RGB, font=SANS_FONT)")
        out.append("y += 1.15")
    return "\n".join(out)
S15 = refcol(REFS_L, "body_l") + "\n" + refcol(REFS_R, "body_l+(body_w-0.4)/2+0.4")

# Slide 16: closing habit -- big statement.
S16 = f"""
_add_rect(slide, left=body_l, top=body_top+0.2, width=0.16, height=2.4, fill_rgb=TURQUOISE_RGB)
_add_text(slide, "Suspect the chooser, not the treatment.",
          left=body_l+0.5, top=body_top+0.2, width=body_w-0.6, height=1.5,
          size=38, color_rgb=INK_RGB, font=MONO_FONT, bold=True)
_add_text(slide, "When a group that CHOSE a treatment looks healthier, the head start usually belongs to the person, not the pill -- and only a coin-flip experiment can promote a correlation to a cause.",
          left=body_l+0.5, top=body_top+1.85, width=body_w-1.0, height=1.2,
          size=18, color_rgb=INK_RGB, font=SANS_FONT)
_add_text(slide, "And take the patient's account seriously -- even when the test says 'fine.'",
          left=body_l+0.5, top=body_top+3.1, width=body_w-1.0, height=0.8,
          size=18, color_rgb=DEEPPINK_RGB, font=MONO_FONT, bold=True)
"""

# ---- speaker notes ---------------------------------------------------------
NOTES = {
 1: "Open like a mystery, because it was one. For twenty years, observational studies -- just watching women, not assigning anything -- kept finding that estrogen users had sharper memory and less Alzheimer's [1, 2]. Doctors nearly started prescribing hormones for the brain. Then in 2003 the WHIMS randomized trial flipped a coin to decide who got hormones, and the benefit didn't just vanish, it reversed: dementia risk actually rose [3, 4]. Same drug, opposite answer. Set up the whole talk as: how can two honest studies disagree this badly? Transition: here it is in one picture.",
 2: "This is our most important slide -- read it slowly. Left bar, 'just watching': hormone users looked protected, risk about two-thirds of non-users. Right bar, the coin-flip trial: risk more than doubled. The amber arrow is the sign flip. The dashed line is 'no effect.' The point for a 15-year-old: the STUDY DESIGN, not the drug, decided the answer. Watching can lie; a coin flip cannot. Transition: so why does watching lie here?",
 3: "Now name the villain: confounding, or 'healthy-user bias.' The women who chose hormone therapy were, on average, richer, more active, and saw doctors more [5]. That hidden head start -- the pink box at the top -- pushes down on BOTH boxes: it makes them more likely to take the pill AND more likely to have good memory. So the dashed bottom arrow, 'takes pill -> better memory,' only LOOKS like cause. It's a mirage created by the third factor [6]. This one diagram is the whole statistical idea of the talk. Transition: you already know this trap from summer.",
 4: "Make it click with the classic. Ice-cream sales and shark attacks rise together every summer -- but ice cream doesn't summon sharks; hot weather drives both. Map it: estrogen use is the ice cream, good memory is the shark attacks, and 'being healthier to start with' is the summer heat. The takeaway line is the transferable habit: when two things move together, hunt for the hidden third factor before you believe one caused the other. Transition: now to our own toy data.",
 6: "Quick tour of the dataset so the results land. 739 women, all 60+, from NHANES -- a real US government health survey, not something we made up. For each woman we know: did she ever use estrogen; her age, education, and income (the possible confounders); and a memory/attention test score, where higher is sharper. Our model tries to predict who lands in the weaker-scoring half. Keep it light -- they don't need every column, just the shape. Transition: what can we actually ask of survey data?",
 7: "Set expectations honestly. We can't randomly assign estrogen in a survey -- so we can't prove cause. But we CAN ask the narrower question in the box: does the estrogen effect survive once we account for age, education, and income? We'll measure estrogen's effect two ways -- crude (alone) and adjusted (after the confounders). If it shrinks, the crude headline was mostly confounding. That crude-vs-adjusted move is the entire method. Transition: start with the naive, tempting look.",
 9: "The tempting headline, exactly as anyone would first compute it. Average the memory score for users versus non-users: 51.8 versus 43.7, an 8-point head start for the estrogen group. If you stopped here, you'd write 'estrogen users think better' -- the same clue the 1990s studies chased. The whole point of the next slide is that we did NOT stop here. Transition: now adjust for who these women were.",
 10: "The punchline slide. Same data, honest test. Crude effect: about -0.33 (negative = looks protective). Adjusted for age, education, and income: it collapses to -0.11. That's a 67% shrink -- two-thirds of the 'benefit' was never the estrogen, it was that users were healthier and better-off to begin with. This is confounding caught in the act, a miniature of exactly why WHIMS overturned the observational headlines. If they remember one number from this talk, make it 67%. Transition: one more check any people-model needs.",
 11: "A fairness audit, the habit from earlier this week. We trained one model to predict low cognition, then split its accuracy by group: 0.73 for estrogen-users, 0.65 for non-users. A model can look fine overall yet work unequally across groups -- and a clinic that trusts it equally could quietly do more harm to the group it's worse at. The mature move isn't to hide the gap; it's to name it. Transition: and there's something no score in this dataset can see at all.",
 13: "Step outside the numbers. Our data is a test SCORE. But many women in menopause report real, daily 'brain fog' even when their objective test looks normal [7] -- the slope lines cross exactly there. And women's symptoms get waved off as stress far too often [8]. The grown-up data-scientist stance holds two true things at once: the confounding is real (the pill may not be the hero), AND the patient's lived report is real evidence, not noise. Transition: so what can this project honestly claim?",
 14: "Be scrupulously honest about scope -- this is what earns trust. CAN DO: show a tempting headline is mostly a mirage and that adjusting shrinks it ~67%. CANNOT DO: prove cause -- we can only correct for confounders we measured and thought of; something unrecorded could still be doing the work. SETTLES IT: a randomized trial, where a coin flip (not the patient) assigns the drug, so the groups start identical. That's why WHIMS could do what twenty years of watching couldn't. Transition: the sources.",
 15: "The paper trail, in order of the story. Top-left: the two observational studies that raised the exciting clue [1, 2]. Bottom of the left column into the trial: the WHIMS randomized results that overturned it [3, 4]. Right column: the healthy-user-bias explanation [5, 6] and the brain-fog / symptom-dismissal pieces [7, 8]. All were checked against PubMed or the journal page. Transition: one habit to take with you.",
 16: "Close on the transferable lesson, not the estrogen specifics. Suspect the chooser, not the treatment: whenever a group that CHOSE something looks healthier, the head start usually belongs to the person, and only a coin-flip experiment can turn a correlation into a cause. And -- because this is medicine, about real women -- take the patient's account seriously even when the test says 'fine.' That pair, statistical humility plus listening, is the whole point of the project. Thank you.",
}

CODE = {1:S1,2:S2,3:S3,4:S4,6:S6,7:S7,9:S9,10:S10,11:S11,13:S13,14:S14,15:S15,16:S16}
DIVIDER_ACCENT = {0:"#40E0D0", 5:"#FF1493", 8:"#F0C840", 12:"#8A2BE2"}

d = json.loads(LJ.read_text())
d["theme"] = "bone"
for i, s in enumerate(d["slides"]):
    p = s.setdefault("params", {})
    if s["kind"] == "section-divider":
        if i in DIVIDER_ACCENT:
            p["accent_hex"] = DIVIDER_ACCENT[i]
        continue
    p["_provenance"] = "agent"
    p["lede"] = ""
    if i in CODE:
        p["code"] = CODE[i].strip("\n")
    if i in NOTES:
        p["notes"] = NOTES[i]

LJ.write_text(json.dumps(d, indent=2, ensure_ascii=False))
print("authored", LJ)
print("content slides authored:", sorted(CODE))
