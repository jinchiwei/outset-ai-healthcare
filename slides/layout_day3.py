"""Handcraft the Day 3 (capstone) expressive layout. Reuses the Day 1/2 design
vocabulary via helpers. Run after build.py --plan-only, then patches the sidecar.

Then:  python slides/layout_day3.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SIDE = ROOT / "slides/day3.md.layout.json"
FIG = str(ROOT / "slides/figures")

SLIDES = {}


def slide(key, code, notes):
    SLIDES[key] = (code.strip("\n"), notes.strip())


def fig(key, fname, notes):
    slide(key, f"_fit_image(slide, '{FIG}/{fname}', left=body_l, top=body_top + 0.05, "
           f"max_w=body_w, max_h=body_h - 0.10)", notes)


def cards3(key, items, notes):
    code = f"""
cards = {items!r}
n = len(cards)
gap = 0.30
cw = (body_w - (n - 1) * gap) / n
for i, (lab, bod, c) in enumerate(cards):
    col = {{'turquoise': TURQUOISE_RGB, 'deeppink': DEEPPINK_RGB, 'amber': AMBER_RGB, 'blueviolet': BLUEVIOLET_RGB}}[c]
    x = body_l + i * (cw + gap)
    _add_rect(slide, left=x, top=body_top, width=cw, height=body_h, fill_rgb=PAPER_RGB)
    _add_rect(slide, left=x, top=body_top, width=cw, height=0.90, fill_rgb=col)
    _add_text(slide, lab, left=x + 0.26, top=body_top, width=cw - 0.52, height=0.90,
              size=17, color_rgb=_text_on(col), font=MONO_FONT, bold=True, anchor=MSO_ANCHOR.MIDDLE)
    _add_text(slide, bod, left=x + 0.26, top=body_top + 1.20, width=cw - 0.52, height=body_h - 1.5,
              size=15, color_rgb=INK_RGB, font=SANS_FONT)
"""
    slide(key, code, notes)


def twobytwo(key, items, notes):
    code = f"""
items = {items!r}
gap = 0.30
cw = (body_w - gap) / 2
ch = (body_h - gap) / 2
cmap = {{'turquoise': TURQUOISE_RGB, 'deeppink': DEEPPINK_RGB, 'amber': AMBER_RGB, 'blueviolet': BLUEVIOLET_RGB}}
for i, (lab, bod, c) in enumerate(items):
    col = cmap[c]
    r = i // 2
    cc = i % 2
    x = body_l + cc * (cw + gap)
    y = body_top + r * (ch + gap)
    _add_rect(slide, left=x, top=y, width=cw, height=ch, fill_rgb=col)
    _add_text(slide, lab, left=x + 0.26, top=y + 0.22, width=cw - 0.52, height=0.42,
              size=18, color_rgb=_text_on(col), font=MONO_FONT, bold=True)
    _add_text(slide, bod, left=x + 0.26, top=y + 0.78, width=cw - 0.52, height=ch - 1.0,
              size=15, color_rgb=_text_on(col), font=SANS_FONT)
"""
    slide(key, code, notes)


def darkpanel(key, intro, rows, notes):
    code = f"""
_add_rect(slide, left=body_l, top=body_top, width=body_w, height=body_h, fill_rgb=DARK_BG_RGB)
_add_text(slide, {intro!r}, left=body_l + 0.5, top=body_top + 0.35, width=body_w - 1.0, height=0.5,
          size=17, color_rgb=WHITE_RGB, font=SANS_FONT)
rows = {rows!r}
cmap = {{'turquoise': TURQUOISE_RGB, 'deeppink': DEEPPINK_RGB, 'amber': AMBER_RGB, 'blueviolet': BLUEVIOLET_RGB}}
ry = body_top + 1.15
rh = (body_h - 1.45) / 3
for i, (lab, bod, c) in enumerate(rows):
    col = cmap[c]
    y = ry + i * rh
    _add_rect(slide, left=body_l + 0.5, top=y + 0.10, width=0.14, height=rh - 0.30, fill_rgb=col)
    _add_text(slide, lab, left=body_l + 0.85, top=y + 0.06, width=4.2, height=rh - 0.2,
              size=18, color_rgb=col, font=MONO_FONT, bold=True, anchor=MSO_ANCHOR.MIDDLE)
    _add_text(slide, bod, left=body_l + 5.1, top=y + 0.06, width=body_w - 5.6, height=rh - 0.2,
              size=15, color_rgb=WHITE_RGB, font=SANS_FONT, anchor=MSO_ANCHOR.MIDDLE)
"""
    slide(key, code, notes)


def statement(key, big, small, notes):
    code = f"""
_add_text(slide, {big!r}, left=body_l, top=body_top + body_h * 0.32, width=body_w, height=1.2,
          size=32, color_rgb=accent_rgb, font=MONO_FONT, bold=True, align=PP_ALIGN.CENTER)
_add_text(slide, {small!r}, left=body_l + 1.0, top=body_top + body_h * 0.60, width=body_w - 2.0, height=1.0,
          size=17, color_rgb=INK_RGB, font=SANS_FONT, align=PP_ALIGN.CENTER)
"""
    slide(key, code, notes)


# --------------------------------------------------------------------------- #
fig("three-afternoons-one-arc", "d3_journey.png", """
Open with pride and momentum. In three afternoons they went from "what is a pixel" to
building real medical AI. Day 1: the ladder on eye scans, end-to-end deep learning. Day 2:
combining image, text, and data with a foundation model. Today: their own build. Make them
feel the distance traveled. Transition: a quick recap of the two big lessons.
""")

fig("what-day-1-taught", "result_ladder.png", """
Recap Day 1 in one idea. The five-model ladder was not about memorizing five architectures;
it was about the jump. A network pretrained on a million everyday photos, reused for eyes,
beat anything trained from scratch on our small dataset. The transferable habit: when in
doubt, start from something pretrained. Transition: and Day 2's lesson.
""")

fig("what-day-2-taught", "d2_two_paradigms.png", """
Recap Day 2. Two things: you can turn any mix of signals (image, text, demographics) into one
table and let a foundation model handle it; and you must distrust a too-good result and hunt
for leakage. Those two habits, reuse pretrained models and stay skeptical, are most of what a
working ML engineer does. Transition: here is everything you can now do.
""")

fig("everything-in-your-toolkit", "d3_panel_toolkit.png", """
This is the confidence slide. Read it out loud, because students underestimate how much they
just learned. They have real models, a real method for training and validating, and, most
importantly, the judgment to evaluate honestly. This is the actual toolkit of a junior
medical-AI engineer. Transition: now put it to work.
""")

# --------------------------------------------------------------------------- #
# Where the field stands (lit-review intro)
# --------------------------------------------------------------------------- #
fig("ai-is-already-in-the-clinic", "d3_field_map.png", """
Zoom out before they build. AI in medicine is not speculative; roughly a thousand AI/ML
devices have FDA clearance, spread across radiology, ophthalmology, pathology, cardiology,
clinical text, and drug discovery. Every example on this map is a real, deployed system. The
point: the thing they are about to attempt is a real and growing field, not a toy. Transition:
a few flagships by name.
""")

fig("four-systems-treating-patients-now", "d3_flagships.png", """
Make it concrete with four real systems, each mapping onto something they built. IDx-DR: an
autonomous screening classifier (Day 1's task, deployed). Viz.ai: a triage detector that pages
the stroke team. Apple Watch: a wearable signal at population scale. Med-PaLM: a clinical LLM
that passed the medical licensing bar. They now understand the shape of all four. Transition:
where is this heading?
""")

fig("the-frontier-multimodal-medical-ai", "d3_frontier.png", """
Name the trajectory. The field is moving from one narrow model per task toward large,
pretrained, multimodal models that ingest image plus text plus signals together, the same
late-fusion idea from Day 2, scaled to foundation models like Med-Gemini and GPT-4. Reassure
them: they already grasp the core idea behind the frontier. Transition: but deployed is not the
same as flawless.
""")

fig("where-it-still-breaks", "d3_panel_breaks.png", """
Ground the hype with a real cautionary tale: a widely deployed hospital sepsis model (live in
hundreds of hospitals) was found on external validation to miss most cases and over-alert
(Wong et al., JAMA Internal Medicine 2021). The failure modes are exactly the ones they already
met: dataset shift, the wrong metric, and leakage. This is why today's rubric rewards honest
evaluation over a high number. Transition: now, your mission.
""")

fig("the-capstone-format", "d3_panel_capstone.png", """
Set the logistics clearly so the sprint runs smoothly. Pairs, because building together is
how real work happens and it keeps everyone unstuck. Ninety minutes to build, with two
check-ins so nobody silently spins. Three-minute presentations at the end. Emphasize: the
goal is a real model you understand, not a perfect one. Transition: pick your problem.
""")

fig("pick-one-or-pitch", "d3_options.png", """
Present the three starter problems. Pneumonia on chest X-rays (closest to Day 2, a clean
binary). Skin-lesion classification including melanoma (multi-class, high-stakes feel).
Or choose any MedMNIST dataset and surprise yourself. All use MedMNIST so data is a
non-issue, it downloads in seconds. And if a pair has their own idea, they can pitch it in
the first ten minutes. Transition: how to choose between them.
""")

fig("how-to-choose", "d3_panel_choose.png", """
Lower the stakes of the choice; any of the three is a great afternoon, so pick on gut in
two minutes. Frame the three reasons: familiarity (pneumonia mirrors Day 2), interest
(melanoma is visceral), or exploration (a dozen MedMNIST sets to poke at). The worst move is
spending fifteen minutes deciding. Transition: once you have picked, how do you actually
build it?
""")

fig("the-build-loop", "d3_workflow.png", """
Teach the single most useful meta-skill: building is a loop, not a straight line. Load and
look at the data. Get the simplest possible thing running. Improve one change at a time.
Evaluate honestly. Then loop. The classic beginner mistake is to skip the baseline and
over-engineer; the pros do the opposite, get something working first. We will walk each
step. Transition: step one.
""")

fig("step-1-get-a-baseline", "d3_panel_step1.png", """
Hammer the baseline-first discipline, it is the most common thing beginners skip. The first
goal is NOT a good model; it is a model that runs end to end and prints a number, however
bad. That number is the safety net every later change is measured against. Resist making it
fancy before it works at all. Transition: now improve, but carefully.
""")

fig("step-2-improve", "d3_panel_step2.png", """
Teach scientific iteration. Change ONE thing, re-measure, keep it only if it genuinely
helped. Train longer, unfreeze the backbone, add augmentation, bigger images, one at a time.
If you change five things and the number moves, you have learned nothing about why. And
logging each result ("augmentation: +3 points") literally writes their presentation for
them. Transition: things will break; here is how to handle it.
""")

fig("step-3-debug", "d3_panel_step3.png", """
Normalize errors; they are not failure, they are the job. The skill is reading them, not
avoiding them. Read the actual message (it usually names the problem), print tensor shapes
(most bugs are shape mismatches), change one thing, retry. And when stuck, Claude is right
there, but understand the fix, do not paste blindly. Transition: speaking of Claude, today
it is a full engineer.
""")

darkpanel("claude-is-your-engineer",
          "Days 1 and 2, Claude filled in blanks. Today it is a full pair programmer.",
          [("DESCRIBE THE GOAL", "give it context and what you want, not just 'make it work'", "turquoise"),
           ("BUILD TOGETHER", "it drafts, you steer, you run it, you both iterate", "amber"),
           ("OWN EVERY LINE", "the rule all week: you must be able to explain your code", "deeppink")],
          """
This is the real-world workflow slide, the one that most reflects how software actually gets
built now. Today Claude is not a hint tool, it is a pair programmer: you describe the goal,
it drafts code, you read and run and fix it together. That is genuinely how engineers work
in 2026. The one unchanged rule, repeated all week: own every line, be able to explain it.
Transition: build freely, but evaluate honestly.
""")

twobytwo("the-four-traps",
         [("ACCURACY LIES", "on imbalanced data, 'always say no' can score high and be useless", "turquoise"),
          ("TARGET LEAKAGE", "a feature that secretly encodes the answer; the Day 2 trap", "deeppink"),
          ("OVERFITTING", "great on training data, bad on new data; watch validation", "amber"),
          ("CHERRY-PICKING", "reporting your one lucky run instead of the typical result", "blueviolet")],
         """
Consolidate the skepticism from two days into one checklist they can hold in their head
during the sprint. Accuracy lies on imbalanced data. Target leakage was the Day 2 gut-punch.
Overfitting is memorizing, caught by validation. And cherry-picking, reporting the one lucky
seed, is the subtle, tempting one under time pressure. Tell them: if your number looks
amazing, suspect one of these four first. Transition: so what does good evaluation look like?
""")

fig("what-makes-a-good-evaluation", "d3_panel_eval.png", """
Define what an honest evaluation is, positively this time. Test on held-out data the model
never trained on. Choose the metric that matches the stakes (sensitivity for screening,
since a miss is worse than a false alarm). And look at the failures, not just the score, the
wrong cases are both the most instructive and the most compelling thing to present.
Transition: which connects to how the talk is judged.
""")

fig("how-your-talk-is-judged", "d3_rubric.png", """
Walk the rubric so expectations are crystal clear, and point out what is NOT on it: highest
accuracy. Five points: it runs, you evaluated honestly, you found a failure, each partner
defends a decision, both contributed. The meta-message: we reward understanding and honesty
over a leaderboard number. A simple model you can fully explain beats a fancy one you cannot.
Transition: and the talk itself is short, so here is how to nail it.
""")

fig("the-three-minute-talk", "d3_panel_talk.png", """
Give them a foolproof three-minute structure so nobody freezes. Show the model doing
something (live or one clear result), give the single best finding, name one honest
limitation. That is a complete, professional short talk. Three minutes is less time than it
sounds, so show, do not narrate. Transition: and both of you talk.
""")

statement("both-of-you-out-loud",
          "Both of you present.",
          "Decide who covers what, and make sure each of you can answer a question about the other's part. The best teams understood the whole thing, not half each.",
          """
Reinforce the pair expectation, which the rubric grades. Both partners speak, and each
should be able to field a question about the other's half, because the strongest teams truly
co-built rather than splitting the work in two. Keep this one short; it is a reminder, not a
lecture. Transition: and after today?
""")

fig("where-to-go-from-here", "d3_whats_next.png", """
Give them an on-ramp so the momentum does not die at 5pm. Keep building (Kaggle, fast.ai,
their own ideas). Go deeper (Andrew Ng's courses, the MICCAI community for medical imaging).
And the real message: medical AI is a genuine, hiring field, not a hobby; people make
careers of exactly what they did this week. Transition: a final word.
""")

statement("you-did-real-work-this-week",
          "Go build something people want.",
          "Three afternoons ago some of you had never written Python. Today you build and evaluate medical AI on real data, and reason about when to trust it. That is the real job, not a toy version of it.",
          """
Close with sincerity and a little ambition. Name the distance traveled: from never writing
Python to building and critically evaluating medical AI in three afternoons. Stress that this
was not a watered-down version, it is the actual work. End on the send-off: go build
something people want, and thank you. Let it land; this is the last thing they hear.
""")


# --------------------------------------------------------------------------- #
def main():
    data = json.loads(SIDE.read_text())
    patched = 0
    for s in data["slides"]:
        if s["kind"] != "freeform":
            continue
        sid = s["slide_id"]
        match = next((k for k in SLIDES if k in sid), None)
        if not match:
            print("NO HANDCRAFT FOR:", sid)
            continue
        code, notes = SLIDES[match]
        s["params"]["code"] = code
        s["params"]["notes"] = notes
        s["params"]["_provenance"] = "agent"
        patched += 1
    SIDE.write_text(json.dumps(data, indent=1))
    print(f"patched {patched} content slides; provenance=agent")


if __name__ == "__main__":
    main()
