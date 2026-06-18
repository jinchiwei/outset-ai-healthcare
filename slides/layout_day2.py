"""Handcraft the Day 2 expressive layout: bespoke freeform code + speaker notes
per content slide, patched into the plan sidecar. Reuses the Day 1 design
vocabulary (figure-dominant, colored-header cards, chips, dark panel) via helpers.

Run AFTER:  build.py --input slides/day2.md ... --plan-only   (writes sidecar)
Then:       python slides/layout_day2.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SIDE = ROOT / "slides/day2.md.layout.json"
FIG = str(ROOT / "slides/figures")

SLIDES = {}


def slide(key, code, notes):
    SLIDES[key] = (code.strip("\n"), notes.strip())


def fig(key, fname, notes):
    """Figure-dominant slide: figure fills the body (figures carry their own title)."""
    slide(key, f"_fit_image(slide, '{FIG}/{fname}', left=body_l, top=body_top + 0.05, "
           f"max_w=body_w, max_h=body_h - 0.10)", notes)


def cards3(key, items, notes):
    """3 cards in a row: colored header band (contrast text) + paper body (ink)."""
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
              size=18, color_rgb=_text_on(col), font=MONO_FONT, bold=True, anchor=MSO_ANCHOR.MIDDLE)
    _add_text(slide, bod, left=x + 0.26, top=body_top + 1.20, width=cw - 0.52, height=body_h - 1.5,
              size=15, color_rgb=INK_RGB, font=SANS_FONT)
"""
    slide(key, code, notes)


# --------------------------------------------------------------------------- #
# Where we are
# --------------------------------------------------------------------------- #
fig("yesterday-and-today", "d2_recap_days.png", """
Set the frame for the day. Yesterday was one image into one network. Today two changes at
once: chest X-rays, which uniquely come paired with a written report, and the idea of
combining several signals about one patient. The throughline to stress: real medicine is
multimodal, a doctor never looks at only the scan. Transition: let us look at the scan first.
""")

fig("one-chest-x-ray", "d2_cxr_findings.png", """
Ground them in the modality. The chest X-ray is the most common imaging test on earth,
done hundreds of millions of times a year. From one image a radiologist reads several
findings: is the heart enlarged (cardiomegaly), is there fluid around the lung (effusion),
a hazy patch of infection (pneumonia/opacity), a collapsed lung (pneumothorax). Today's
task will be one of these yes/no calls. Transition: but the image is only half the data.
""")

fig("what-is-cardiomegaly", "d2_cardiomegaly.png", """
Make the target concrete before any modeling. Cardiomegaly just means an enlarged heart, and
radiologists decide it with a real measurement: the cardiothoracic ratio, heart width over
chest width. Under ~0.50 is normal; above it the heart is enlarged. The figure shows a normal
heart (CTR 0.45) beside an enlarged one (CTR 0.62). The point for students: our yes/no label
isn't arbitrary, it reflects a physical thing you can see and measure. Transition: and every
scan comes with a written report describing it.
""")

fig("every-scan-comes-with-text", "d2_report.png", """
This is the key difference from Day 1. Every chest X-ray in the clinic comes with a
radiology report, the radiologist's free-text findings and impression. That text is data,
and historically it is where the richest clinical signal lives. If a model can read it,
we unlock a whole second channel beyond the pixels. The figure shows the move we will make:
free text in, structured yes/no findings out. Transition: reading text is a job for a
language model, so what is one?
""")

# --------------------------------------------------------------------------- #
# What is a language model
# --------------------------------------------------------------------------- #
fig("the-bridge-from-yesterday", "concept_bridge.png", """
Reassure them that today is not a fresh mountain to climb. Yesterday's Vision Transformer
chopped an image into patches and used attention to decide which patches mattered. A large
language model does the identical thing with words instead of patches. The bottom row of
this figure IS an LLM. Same machinery, different input. Everything else today is just
consequences of that one idea. Transition: so how does text become something attention can
chew on?
""")

fig("tokenization", "d2_tokenization.png", """
Mirror yesterday's "an image is just numbers" with "text is just numbers too". A model
cannot read letters, so text is split into tokens, whole words or word-pieces like "en" +
"larged", and each token is looked up as a number. Once text is a sequence of numbers, the
same attention machinery from the ViT applies directly. Keep it light, the point is the
parallel to images. Transition: and attention is what finds the meaning.
""")

fig("attention-which-words", "d2_attention.png", """
Show what attention does on text. Given a report, the model does not treat every word
equally; it learns that "heart" and "enlarged" carry the finding while "the" and "is" are
filler. Bigger and bolder in the figure means more attention. This is the same mechanism
that let the ViT focus on the lesion in an image, now focusing on the meaningful words.
Transition: stack a lot of this up and you get something that can do real work, and also
make things up.
""")

fig("what-an-llm-actually-does", "d2_llm_does.png", """
Demystify the LLM in one breath. Under all the capability, the training objective is just:
predict the next token, given everything so far. Do that at massive scale and you get a
system that can summarize, answer, and extract structure from messy text. But the objective
rewards plausibility, not truth, which is exactly why it will confidently make things up.
Land that tension; the next slide is the payoff. Transition: here is what that looks like.
""")

fig("hallucination", "d2_hallucination.png", """
The most important safety idea about LLMs. When the model does not know something, it does
not stop or say "unknown", it generates a fluent, confident, fabricated answer. Ask it for a
medication the report never mentions and it will invent a plausible drug and dose. For a
chatbot that is annoying; in medicine a confident wrong answer is the dangerous kind, because
people believe it. The rule for the lab and for life: use the LLM, but verify. Transition:
with that caution noted, let us build the multimodal model.
""")

# --------------------------------------------------------------------------- #
# The multimodal stack
# --------------------------------------------------------------------------- #
fig("three-signals", "d2_three_signals.png", """
Lay out the build. For each patient we have three different signals: the X-ray pixels, the
report text, and basic demographics like age and smoking history. Each says something on its
own; combined they should be stronger, which is the whole bet of multimodal modeling. The
challenge, and the rest of this section, is getting three very different things into one form
a single model can use. Transition: start with the image.
""")

fig("image-to-a-vote", "d2_stacking.png", """
First signal, and a deliberate callback to Day 1. How do we get the X-ray into the table?
You could hand-craft a hundred radiomics numbers, or dump a 512-number embedding, but both
swamp a tabular model. Cleaner move: train an actual image model, exactly the transfer-
learning recipe from yesterday, and feed the table only its prediction, one probability,
img_pred. That is the image's vote. This pattern has a name: late fusion, or stacking. Each
modality produces one opinion; the tabular model combines the opinions. Transition: but there
is an honesty catch.
""")

fig("one-honest-catch", "d2_oof.png", """
A subtle but real trap in stacking, and a chance to reinforce the day's theme before the big
leakage reveal. If the image model scores a patient it trained on, that score is too
optimistic; it has effectively peeked. So when the instructor pre-computed every img_pred,
each one came from a model trained only on the other patients, out-of-fold. Same fairness
instinct that runs through the whole day: never let a model grade itself. Plant this now, so
when the text leakage lands in a few slides, students already have the mental hook.
Transition: now the second signal, the text.
""")

fig("text-to-numbers", "d2_text_to_num.png", """
Second signal. We hand each radiology report to a language model and ask for structured
findings as JSON: cardiomegaly yes/no, effusion yes/no, and so on. Those flags become
numbers. The practical wrinkle students will feel: calling an LLM hundreds of times costs
money and time, so the instructor pre-ran it and saved the answers. In the lab you load the
cache, no API key needed; the one live call is the demo. Transition: image numbers, text
numbers, plus demographics, and now the magic step.
""")

fig("everything-becomes-one-tabular", "d2_tabular_row.png", """
The big idea of the day, and it is satisfyingly simple. The image model's vote is a number,
text features are numbers, demographics are numbers. Lay them all side by side and every
patient becomes one row in a spreadsheet, with a label to predict. We have taken a scary-
sounding "multimodal" problem and reduced it to a table. That reframing is most of the
insight. Transition: and there is a very modern way to model a table.
""")

fig("tabpfn", "d2_tabpfn.png", """
Who predicts from the table? Not a network you train for an hour. TabPFN is a foundation
model pretrained on millions of synthetic tables; you "fit" it (it just studies your rows)
and "predict", both in seconds, no tuning. Connect it to yesterday: this is the same
pretraining-and-reuse shift you saw with ImageNet and ResNet, now for tabular data. The
2020s pattern: reach for a pretrained foundation model first. Transition: so, does stacking
all three signals actually help?
""")

# --------------------------------------------------------------------------- #
# Did it work?
# --------------------------------------------------------------------------- #
fig("adding-text-looks-amazing", "d2_result.png", """
Show the result and let them feel the excitement first. The image vote plus demographics
alone gets about 72%, a real honest signal straight from the pixels. Add the LLM-extracted
text features and it leaps to about 98%. One extra signal, a huge jump. If we stopped here we
would celebrate. But set the hook: a jump that big should make a careful person suspicious,
not happy. Transition: so let us be suspicious.
""")

fig("wait-that-is-too-good", "d2_leakage.png", """
The single most important lesson of Day 2. The report we extracted text features from
literally states the diagnosis we are trying to predict, the impression says "Cardiomegaly".
So the LLM's "cardiomegaly: yes" matches the label almost perfectly, correlation around 0.96.
The model is not detecting disease, it is copying the answer off the report. The 89% is a
mirage. Make sure this lands hard, because it generalizes far beyond this lab. Transition: so
what would have been a fair test?
""")

fig("what-a-fair-test-looks-like", "d2_fair_test.png", """
Turn the gotcha into a transferable skill. Target leakage is when a feature secretly carries
the answer, and catching it is what separates a careful modeler from a fooled one. Three
habits: never let a feature encode the label; prefer inputs that exist before the label is
known (pixels and demographics are fair, the radiologist's impression is not); and always
report the honest, de-leaked number even though it is lower. Transition: you will expose this
yourself in the lab.
""")

# --------------------------------------------------------------------------- #
# The lab
# --------------------------------------------------------------------------- #
slide("how-the-lab-works", """
steps = [('# TODO', 'Build the table', 'assemble the three feature groups, call TabPFN fit and predict', TURQUOISE_RGB),
         ('-/-', 'Run the ablation', 'drop the text features, refit, compare; watch the gap shrink', AMBER_RGB),
         ('?', 'Discuss', 'was that a fair test? what would you change?', DEEPPINK_RGB)]
n = len(steps)
gap = 0.30
cw = (body_w - (n - 1) * gap) / n
for i, (tag, head, bod, c) in enumerate(steps):
    x = body_l + i * (cw + gap)
    _add_rect(slide, left=x, top=body_top, width=cw, height=body_h, fill_rgb=PAPER_RGB)
    _add_rect(slide, left=x, top=body_top, width=cw, height=0.10, fill_rgb=c)
    cw2 = 2.0
    _add_rect(slide, left=x + 0.24, top=body_top + 0.40, width=cw2, height=0.78, fill_rgb=c)
    _add_text(slide, tag, left=x + 0.24, top=body_top + 0.40, width=cw2, height=0.78,
              size=24, color_rgb=_text_on(c), font=MONO_FONT, bold=True, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    _add_text(slide, head, left=x + 0.24, top=body_top + 1.45, width=cw - 0.48, height=0.5,
              size=20, color_rgb=INK_RGB, font=MONO_FONT, bold=True)
    _add_text(slide, bod, left=x + 0.24, top=body_top + 2.15, width=cw - 0.48, height=body_h - 2.4,
              size=15, color_rgb=INK_RGB, font=SANS_FONT)
""", """
Same shape as Day 1: one notebook, fill the TODO blanks. They assemble the three feature
groups into a table, call TabPFN's fit and predict, read the accuracy, then run the ablation
that drops the text features and watch the leak disappear. End on the discussion prompt, the
"was that fair?" question, because the thinking matters more than the number. Transition: a
word on Claude and cost.
""")

slide("using-claude", """
_add_rect(slide, left=body_l, top=body_top, width=body_w, height=body_h, fill_rgb=DARK_BG_RGB)
_add_text(slide, 'Claude is your pair programmer again. One new wrinkle: real API calls cost money.',
          left=body_l + 0.5, top=body_top + 0.35, width=body_w - 1.0, height=0.5,
          size=17, color_rgb=WHITE_RGB, font=SANS_FONT)
rows = [('PAIR PROGRAMMER', 'stuck on a blank? ask Claude, then make sure you understand the answer', TURQUOISE_RGB),
        ('COST DISCIPLINE', 'pre-caching is how real teams keep LLM bills sane: cache once, reuse many times', AMBER_RGB),
        ('THE RULE, STILL', 'always be able to explain what your code does', DEEPPINK_RGB)]
ry = body_top + 1.15
rh = (body_h - 1.45) / 3
for i, (lab, bod, c) in enumerate(rows):
    y = ry + i * rh
    _add_rect(slide, left=body_l + 0.5, top=y + 0.10, width=0.14, height=rh - 0.30, fill_rgb=c)
    _add_text(slide, lab, left=body_l + 0.85, top=y + 0.06, width=4.2, height=rh - 0.2,
              size=18, color_rgb=c, font=MONO_FONT, bold=True, anchor=MSO_ANCHOR.MIDDLE)
    _add_text(slide, bod, left=body_l + 5.1, top=y + 0.06, width=body_w - 5.6, height=rh - 0.2,
              size=15, color_rgb=WHITE_RGB, font=SANS_FONT, anchor=MSO_ANCHOR.MIDDLE)
""", """
Two quick points on tooling. Claude is the pair programmer again, same rule as yesterday:
use it, but be able to explain your code. The new lesson is cost discipline. Real LLM calls
cost money per request, so calling one on hundreds of reports every run is wasteful; we
pre-cache the answers once and reuse them. That cache-once-reuse-many pattern is exactly how
real teams keep their AI bills sane. Transition: let us close by zooming out.
""")

# --------------------------------------------------------------------------- #
# What you built
# --------------------------------------------------------------------------- #
fig("two-paradigms", "d2_two_paradigms.png", """
The big recap. In two days students have built the two dominant approaches in applied AI.
Day 1: feed raw data to one big network that learns its own features (end-to-end deep
learning). Day 2: hand-engineer features from every signal and let a pretrained foundation
model handle the resulting table. Neither is "better"; knowing which to reach for, given the
data and the constraints, is half the real job. Transition: one last caution about today's
new kind of data.
""")

fig("text-data-is-even-more-sensitive", "d2_text_sensitive.png", """
Close the ethics thread, because today introduced a riskier kind of data. A pixel grid is
fairly anonymous; a clinical note is full of identifying detail, names, dates, places, and
"de-identified" text frequently is not, fully. Worse, LLMs can memorize and regurgitate
training text verbatim, so sensitive notes can leak back out. The discipline: minimize (use
the least data that does the job) and protect it like the medical record it is. Transition:
tomorrow, you build.
""")

slide("tomorrow-build-your-own", """
_add_text(slide, 'You now have the whole toolkit.', left=body_l, top=body_top + body_h * 0.30,
          width=body_w, height=1.0, size=34, color_rgb=accent_rgb, font=MONO_FONT, bold=True,
          align=PP_ALIGN.CENTER)
_add_text(slide, 'End-to-end deep learning  ·  transfer learning  ·  multimodal feature stacks  ·  foundation models',
          left=body_l, top=body_top + body_h * 0.52, width=body_w, height=0.6, size=16,
          color_rgb=INK_RGB, font=SANS_FONT, align=PP_ALIGN.CENTER)
_add_text(slide, 'Tomorrow you pick a problem, grab a dataset, and build it start to finish, with Claude as your engineer.',
          left=body_l, top=body_top + body_h * 0.66, width=body_w, height=0.8, size=17,
          color_rgb=INK_RGB, font=SANS_FONT, align=PP_ALIGN.CENTER)
""", """
Send them off energized. In two afternoons they have gone from "what is a pixel" to building
both dominant paradigms of medical AI on real data. List the toolkit out loud so they feel
how much they have. Tomorrow is the capstone: they choose a problem, grab a dataset, and
build something end to end with Claude as their engineer. That is the whole arc of the
course paying off.
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
