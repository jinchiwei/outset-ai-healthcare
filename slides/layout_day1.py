"""Handcraft the Day 1 expressive layout: bespoke freeform code + speaker notes
for every content slide. Patches the plan sidecar in place (keeps slide_ids and
content_hashes from the --plan-only --shake skeleton; replaces params.code/notes
and stamps _provenance=agent so the bespoke gate passes).

Run AFTER:  build.py --plan-only --shake   (writes day1.md.layout.json)
Then:       python slides/layout_day1.py
Then:       build.py --input slides/day1.md --output slides/build/day1.pptx --qa
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SIDE = ROOT / "slides/day1.md.layout.json"
FIG = str(ROOT / "slides/figures")  # absolute; embedded into code strings

# Each entry: h2-key substring -> (code, notes). Matched against slide_id.
SLIDES = {}


def slide(key, code, notes):
    SLIDES[key] = (code.strip("\n"), notes.strip())


# --------------------------------------------------------------------------- #
# Clinical-problem section: 5 figure-dominant slides + a failure-modes 2x2
# --------------------------------------------------------------------------- #
def fig_slide(key, fname, notes, caption=""):
    cap = ""
    if caption:
        cap = (f"\n_add_text(slide, '{caption}', left=body_l, top=body_top + body_h - 0.48, "
               f"width=body_w, height=0.42, size=16, color_rgb=accent_rgb, font=MONO_FONT, "
               f"bold=True, align=PP_ALIGN.CENTER)")
        max_h = "body_h - 0.65"
    else:
        max_h = "body_h - 0.10"
    code = (f"_fit_image(slide, '{FIG}/{fname}', left=body_l, top=body_top + 0.05, "
            f"max_w=body_w, max_h={max_h}){cap}")
    slide(key, code, notes)


fig_slide("ai-is-already-in-the-clinic", "clinical_ai_in_medicine.png", """
Open by killing the "someday" framing. AI is reading medical data in real clinics right
now, across basically every specialty: radiology finds nodules and bleeds, pathology
grades tumors on biopsy slides, dermatology flags melanoma, cardiology catches abnormal
heart rhythms, and ophthalmology screens the eye. The FDA has cleared well over a thousand
of these. The point for students: this is a real, employable field, and today they get to
work in one corner of it. Transition: that corner is the eye, so let us look at the disease.
""")

fig_slide("what-diabetic-retinopathy-does", "clinical_dr_progression.png", """
Now the actual medicine, and it is worth making vivid. Diabetes keeps blood sugar too high
for years, which slowly damages the microscopic blood vessels lining the retina. Early on
you get tiny bulges and dot hemorrhages. It progresses to bigger bleeds and yellow lipid
deposits. In the worst stage the eye desperately grows fragile new vessels that bleed into
the eye and can detach the retina, causing blindness. The emotional anchor: this is a
leading cause of blindness in working-age adults, AND it is largely preventable if you
catch it early. That tension, devastating but preventable, is the whole reason screening
matters. Transition: so why don't we just screen everyone?
""")

fig_slide("a-screening-problem", "clinical_access_gap.png", """
Here is the systems problem. Every person with diabetes, and there are hundreds of millions
of them, should get their retina checked every single year. But a retina exam needs a
trained eye specialist, and there are nowhere near enough, especially in rural areas and
lower-income countries. So a huge fraction of diabetics never get screened and lose vision
they did not have to. This is the gap: enormous, routine need on one side, scarce expertise
on the other. That is the shape of problem where AI screening genuinely helps, by putting a
specialist-level read anywhere there is a camera. Transition: and this is not hypothetical.
""")

fig_slide("from-research-to-the-clinic", "clinical_deployment.png", """
Walk the timeline, because it is genuinely fast and inspiring. 2016: a Google team publishes
in JAMA that a deep network grades diabetic retinopathy about as well as ophthalmologists.
2018: IDx-DR becomes the first FDA-authorized AUTONOMOUS AI diagnostic, meaning it gives a
screening result with no doctor in the loop, run by a nurse in a primary-care office. Soon
after, screening programs roll out in clinics in India and Thailand where specialists are
scarce. Two-ish years from paper to patients. The lesson for students: the gap between a
good research result and real-world impact can be short. Transition: so what exactly does
the model decide?
""")

fig_slide("reading-the-retina", "clinical_severity_scale.png", """
Connect the medicine to the machine-learning task. Ophthalmologists grade DR on a 0-to-4
scale: none, mild, moderate, severe, and proliferative, the sight-threatening stage. That is
rich clinical detail, but for SCREENING it collapses to one decision: is this grade 2 or
worse? If yes, refer to a specialist. That single yes/no line, "referable DR", is the exact
target the students will train every model on this afternoon. Emphasize: we are not asking
the model to be the doctor, just to decide who needs to see one. Transition: that is the
clinical picture; now the technical question, what is the model actually looking at?
""")

# --------------------------------------------------------------------------- #
# How these systems fail  (2x2 color zones)
# --------------------------------------------------------------------------- #
slide("how-these-systems-fail", """
gap = 0.30
cw = (body_w - gap) / 2
ch = (body_h - gap) / 2
items = [('SHORTCUT LEARNING', 'the model reads the ruler in the image, not the tumor', TURQUOISE_RGB),
         ('DISTRIBUTION SHIFT', 'your hospital is not the training hospital, so accuracy quietly drops', DEEPPINK_RGB),
         ('OVERCONFIDENCE', '"90% sure" can mean nothing; confident and wrong is the dangerous mix', AMBER_RGB),
         ('AUTOMATION BIAS', 'when the AI is usually right, people stop checking, and the rare miss sails through', BLUEVIOLET_RGB)]
for i, (lab, bod, c) in enumerate(items):
    r = i // 2
    col = i % 2
    x = body_l + col * (cw + gap)
    y = body_top + r * (ch + gap)
    _add_rect(slide, left=x, top=y, width=cw, height=ch, fill_rgb=c)
    _add_text(slide, lab, left=x + 0.26, top=y + 0.22, width=cw - 0.52, height=0.42,
              size=18, color_rgb=_text_on(c), font=MONO_FONT, bold=True)
    _add_text(slide, bod, left=x + 0.26, top=y + 0.78, width=cw - 0.52, height=ch - 1.0,
              size=15, color_rgb=_text_on(c), font=SANS_FONT)
""", """
Before the technical content, plant the healthy skepticism that runs all week. These
systems fail in subtle, human ways. Shortcut learning: a tumor detector secretly keys on
the ruler that techs place next to tumors, so it "detects" rulers. Distribution shift: it
was tuned on one hospital's camera and quietly degrades on yours. Overconfidence: it reports
90% on everything because nobody calibrated it. Automation bias: the scariest one socially,
when the AI is right 99% of the time, the human stops double-checking, so the 1% miss goes
straight to the patient. The takeaway: knowing how these break is the real expertise, more
than chasing a higher accuracy number. Transition: to reason about any of it, we start with
what an image actually is.
""")

# --------------------------------------------------------------------------- #
# 3. An image is just numbers  (figure dominant)
# --------------------------------------------------------------------------- #
slide("an-image-is-just-numbers", f"""
_fit_image(slide, '{FIG}/concept_image_numbers.png', left=body_l, top=body_top + 0.05, max_w=body_w, max_h=body_h - 0.75)
_add_text(slide, 'The model never sees a photo. It sees a spreadsheet of about 150,000 numbers.',
          left=body_l, top=body_top + body_h - 0.55, width=body_w, height=0.45,
          size=18, color_rgb=accent_rgb, font=MONO_FONT, bold=True, align=PP_ALIGN.CENTER)
""", """
This is the most important conceptual unlock of the day, and it is simple: a picture is
just a grid of numbers. Each pixel is a brightness from 0 to 255. A grayscale image is
one grid; a color image is three grids stacked, one each for red, green, and blue. So a
224 by 224 color image is about 150,000 numbers. Everything we do after this, every
model, is math on those numbers. Mention augmentation: if we flip or rotate or brighten
the same eye, it is still the same eye but a fresh set of numbers, which is free extra
training variety. Transition: now that the image is numbers, what does it mean for a
model to learn?
""")

# --------------------------------------------------------------------------- #
# 4. A classifier is a function  (input -> f -> output pipeline)
# --------------------------------------------------------------------------- #
slide("a-classifier-is-a-function", """
bw = 3.3
bh = 2.7
gap = (body_w - 3 * bw) / 2
ch = 1.25
stack_h = bh + 0.6 + ch
y = body_top + max(0.0, (body_h - stack_h) / 2)
xs = [body_l, body_l + bw + gap, body_l + 2 * (bw + gap)]
labels = [('INPUT', 'the image, as a pile of numbers', TURQUOISE_RGB),
          ('f(x)', 'the model: knobs we tune', INK_RGB),
          ('OUTPUT', 'a label: refer, or not', DEEPPINK_RGB)]
for x, (lab, bod, c) in zip(xs, labels):
    _add_rect(slide, left=x, top=y, width=bw, height=bh, fill_rgb=c)
    _add_text(slide, lab, left=x + 0.2, top=y + bh * 0.30, width=bw - 0.4, height=0.7,
              size=30, color_rgb=_text_on(c), font=MONO_FONT, bold=True, align=PP_ALIGN.CENTER)
    _add_text(slide, bod, left=x + 0.25, top=y + bh * 0.62, width=bw - 0.5, height=0.8,
              size=15, color_rgb=_text_on(c), font=SANS_FONT, align=PP_ALIGN.CENTER)
for i in range(2):
    ax = xs[i] + bw + 0.12
    _add_flat_shape(slide, MSO_SHAPE.RIGHT_ARROW, left=ax, top=y + bh / 2 - 0.22,
                    width=gap - 0.24, height=0.44, fill_rgb=AMBER_RGB)
cy = y + bh + 0.6
_add_rect(slide, left=body_l, top=cy, width=body_w, height=ch, fill_rgb=AMBER_RGB)
_add_text(slide, 'LEARNING  =  adjust the knobs until the guesses match the truth on examples we already know',
          left=body_l + 0.35, top=cy, width=body_w - 0.7, height=ch,
          size=18, color_rgb=INK_RGB, font=MONO_FONT, bold=True, anchor=MSO_ANCHOR.MIDDLE)
""", """
Demystify the word "model". A classifier is just a function: numbers go in, a label comes
out. For us, the image numbers go in, and out comes "refer this eye" or "do not". The
function has knobs, often millions of them. "Learning" is nothing mystical: it is turning
those knobs until the function's guesses line up with the right answers on examples we
already have labels for. That is it. Every model today does the same thing; what differs
is how much structure the function is allowed to see in the numbers. Transition: so let
us build five of them, each one able to see more than the last.
""")

# --------------------------------------------------------------------------- #
# 5. Five models, one ladder  (ascending staircase)
# --------------------------------------------------------------------------- #
slide("the-ladder-we-build", """
steps = [('1', 'Logistic\\nregression', 'one straight line', MUTED_RGB),
         ('2', 'MLP', 'curved, still blind to space', MUTED_RGB),
         ('3', 'CNN', 'sees shapes', TURQUOISE_RGB),
         ('4', 'ResNet', 'borrows trained vision', DEEPPINK_RGB),
         ('5', 'ViT', 'patches that attend', BLUEVIOLET_RGB)]
n = len(steps)
gap = 0.16
sw = (body_w - (n - 1) * gap) / n
for i, (num, name, sub, c) in enumerate(steps):
    h = body_h * (0.40 + 0.135 * i)
    x = body_l + i * (sw + gap)
    yt = body_top + body_h - h
    _add_rect(slide, left=x, top=yt, width=sw, height=h, fill_rgb=c)
    tc = _text_on(c)
    _add_text(slide, num, left=x + 0.14, top=yt + 0.14, width=sw - 0.28, height=0.6,
              size=30, color_rgb=tc, font=MONO_FONT, bold=True)
    _add_text(slide, name, left=x + 0.14, top=yt + 0.78, width=sw - 0.28, height=0.8,
              size=16, color_rgb=tc, font=MONO_FONT, bold=True)
    _add_text(slide, sub, left=x + 0.14, top=yt + h - 0.95, width=sw - 0.28, height=0.85,
              size=12, color_rgb=tc, font=SANS_FONT)
""", """
This is the spine of the whole day. We solve the same task five times, each model a rung
higher than the last, and we watch the accuracy climb. Logistic regression flattens the
image and draws one straight boundary, no idea pixels form shapes. An MLP bends that
boundary but is still blind to space. A CNN slides filters across the image and finally
sees structure. ResNet borrows vision already learned from a million everyday photos. The
Vision Transformer chops the image into patches that pay attention to each other. The
gray rungs are the flat-pixel models, then the color rungs are where it gets good.
Transition: let us look closely at the three that matter most, starting with the CNN.
""")

# --------------------------------------------------------------------------- #
# 6. The CNN sees structure  (figure dominant, full width)
# --------------------------------------------------------------------------- #
slide("the-cnn-sees-structure", f"""
_fit_image(slide, '{FIG}/concept_cnn.png', left=body_l, top=body_top + 0.10, max_w=body_w, max_h=body_h - 0.20)
""", """
The jump from MLP to CNN is the cleanest idea in computer vision. A plain network treats
every pixel as unrelated to its neighbors, which is absurd: in a real image, nearby pixels
form edges, blobs, lesions. A convolutional network slides one small filter across the
WHOLE image, so if it learns to detect a hemorrhage in the top-left, it detects it
anywhere. That weight-sharing is what "seeing spatial structure" means, and it is why the
CNN beats the flat-pixel models. Point at the highlighted patch in the figure: the filter
looks at a little neighborhood, produces one number in the feature map, then slides over.
Transition: CNNs are great, but training one from scratch on a small medical set is hard,
so the next rung borrows a brain.
""")

# --------------------------------------------------------------------------- #
# 7. ResNet: borrow a trained brain  (figure + 2 chips)
# --------------------------------------------------------------------------- #
slide("resnet-borrows", f"""
fig_h = body_h * 0.60
_fit_image(slide, '{FIG}/concept_resnet.png', left=body_l, top=body_top, max_w=body_w, max_h=fig_h)
chip_y = body_top + fig_h + 0.22
ch = body_h - fig_h - 0.30
gap = 0.30
cw = (body_w - gap) / 2
chips = [('SKIP CONNECTION', 'carry the input forward so even very deep networks train', BLUEVIOLET_RGB),
         ('TRANSFER LEARNING', 'freeze a network pretrained on a million photos, teach one new layer', DEEPPINK_RGB)]
for i, (lab, bod, c) in enumerate(chips):
    x = body_l + i * (cw + gap)
    _add_rect(slide, left=x, top=chip_y, width=cw, height=ch, fill_rgb=c)
    _add_text(slide, lab, left=x + 0.22, top=chip_y + 0.14, width=cw - 0.44, height=0.34,
              size=15, color_rgb=_text_on(c), font=MONO_FONT, bold=True)
    _add_text(slide, bod, left=x + 0.22, top=chip_y + 0.52, width=cw - 0.44, height=ch - 0.62,
              size=14, color_rgb=_text_on(c), font=SANS_FONT)
""", """
ResNet brings two ideas. First, the skip connection: it adds the block's input straight
to its output, which sounds trivial but is what lets networks get 50+ layers deep without
the training falling apart. Second, and the one students will feel in the lab: transfer
learning. Instead of learning vision from scratch, we take a ResNet already trained on a
million everyday photos, freeze it, and train only a small new final layer for our eye
task. It already knows edges and textures; we just teach it which combinations mean
"refer". This is the single biggest practical trick in modern vision, and it is the jump
on the leaderboard. Transition: the last rung swaps convolutions for attention.
""")

# --------------------------------------------------------------------------- #
# 8. Vision Transformer  (figure dominant, full width)
# --------------------------------------------------------------------------- #
slide("vision-transformer", f"""
_fit_image(slide, '{FIG}/concept_vit.png', left=body_l, top=body_top + 0.10, max_w=body_w, max_h=body_h - 0.20)
""", """
The Vision Transformer is the newest rung and the bridge to tomorrow. It throws out
convolutions entirely. Instead it chops the image into a grid of patches, turns each patch
into a vector, and uses "attention" to let the patches weigh how much they matter to each
other, before making a call. Do not over-explain attention here; the one sentence that
lands is: the model decides which parts of the image to focus on. Keep this slide quick,
because the parallel to language models is the payoff and we make it explicit at the end
of the day. Transition: that is the full ladder; now you go build it.
""")

# --------------------------------------------------------------------------- #
# 9. How the lab works  (3 numbered steps, code motif)
# --------------------------------------------------------------------------- #
slide("how-the-lab-works", """
steps = [('# TODO', 'Fill the blanks', 'each model has 2-3 missing lines; write them, run it, read the number', TURQUOISE_RGB),
         ('?>', 'Predict, then check', 'guess each model\\'s accuracy first; being wrong is how intuition forms', AMBER_RGB),
         ('->', 'Ask for help', 'stuck on a blank? ask me, or ask Claude; both are fair game', DEEPPINK_RGB)]
n = len(steps)
gap = 0.30
cw = (body_w - (n - 1) * gap) / n
for i, (tag, head, bod, c) in enumerate(steps):
    x = body_l + i * (cw + gap)
    _add_rect(slide, left=x, top=body_top, width=cw, height=body_h, fill_rgb=PAPER_RGB)
    _add_rect(slide, left=x, top=body_top, width=cw, height=0.10, fill_rgb=c)
    # accent chip holds the code-motif tag (readable: text via _text_on, never accent-on-paper)
    chip_w = 2.0
    _add_rect(slide, left=x + 0.24, top=body_top + 0.40, width=chip_w, height=0.78, fill_rgb=c)
    _add_text(slide, tag, left=x + 0.24, top=body_top + 0.40, width=chip_w, height=0.78,
              size=26, color_rgb=_text_on(c), font=MONO_FONT, bold=True,
              align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    _add_text(slide, head, left=x + 0.24, top=body_top + 1.45, width=cw - 0.48, height=0.5,
              size=20, color_rgb=INK_RGB, font=MONO_FONT, bold=True)
    _add_text(slide, bod, left=x + 0.24, top=body_top + 2.15, width=cw - 0.48, height=body_h - 2.4,
              size=15, color_rgb=INK_RGB, font=SANS_FONT)
""", """
Set up the lab mechanics in thirty seconds. It is one notebook for everyone. Each model
step has a couple of blank lines marked TODO; the student fills them in, runs the cell,
and sees the accuracy. Before running each model, they jot a one-word guess at how well it
will do; the point is not to be right, it is that guessing builds the intuition for why
the next rung is better. And there is no shame in getting stuck: ask the instructor, ask
Claude, both fine. Transition: a word on using Claude well, because that is a real skill.
""")

# --------------------------------------------------------------------------- #
# 10. Using Claude well  (dark terminal-style panel)
# --------------------------------------------------------------------------- #
slide("using-claude-well", """
_add_rect(slide, left=body_l, top=body_top, width=body_w, height=body_h, fill_rgb=DARK_BG_RGB)
_add_text(slide, 'you all have Claude. in the real world it takes an idea to a working result.',
          left=body_l + 0.5, top=body_top + 0.35, width=body_w - 1.0, height=0.5,
          size=17, color_rgb=WHITE_RGB, font=SANS_FONT)
rows = [('ASK CLEARLY', 'give it the goal and the context, not just "fix this"', TURQUOISE_RGB),
        ('READ THE ANSWER', 'I read what it writes; I do not paste blindly', AMBER_RGB),
        ('OWN IT', 'the one rule all week: always be able to explain your code', DEEPPINK_RGB)]
ry = body_top + 1.15
rh = (body_h - 1.45) / 3
for i, (lab, bod, c) in enumerate(rows):
    y = ry + i * rh
    _add_rect(slide, left=body_l + 0.5, top=y + 0.10, width=0.14, height=rh - 0.30, fill_rgb=c)
    _add_text(slide, lab, left=body_l + 0.85, top=y + 0.06, width=4.0, height=rh - 0.2,
              size=20, color_rgb=c, font=MONO_FONT, bold=True, anchor=MSO_ANCHOR.MIDDLE)
    _add_text(slide, bod, left=body_l + 4.9, top=y + 0.06, width=body_w - 5.4, height=rh - 0.2,
              size=16, color_rgb=WHITE_RGB, font=SANS_FONT, anchor=MSO_ANCHOR.MIDDLE)
""", """
This is the philosophy slide for how we use AI in this course, and it is worth slowing
down on. Claude is not the subject of the class and it is not cheating; it is the tool a
working engineer reaches for. Model it live on one TODO: show how you phrase the ask with
the goal and context, show that you READ the response and sanity-check it rather than
pasting on faith, and land the one rule that governs the whole week, including the capstone
rubric on Wednesday: you can use Claude freely, but you must always be able to explain
what your code does. Then get out of the way and let them build. Transition: here is what
they will have built by the end of the hour.
""")

# --------------------------------------------------------------------------- #
# 11. The leaderboard you built  (figure dominant)
# --------------------------------------------------------------------------- #
slide("the-leaderboard-you-built", f"""
_fit_image(slide, '{FIG}/result_ladder.png', left=body_l, top=body_top + 0.05, max_w=body_w, max_h=body_h - 0.65)
_add_text(slide, 'Flat-pixel models near a coin flip. The CNN sees structure. Transfer learning takes the leap.',
          left=body_l, top=body_top + body_h - 0.50, width=body_w, height=0.42,
          size=17, color_rgb=accent_rgb, font=MONO_FONT, bold=True, align=PP_ALIGN.CENTER)
""", """
This is the payoff slide: the actual leaderboard from the lab. Read it left to right.
Logistic regression and the MLP sit in the mid-70s, both flattening pixels, both blind to
space; the gap between them is noise. The from-scratch CNN edges up because it finally
sees structure. Then ResNet and the Vision Transformer jump to the high 80s, and that jump
is almost entirely transfer learning, reusing a network that already knew how to see.
Note honestly that ResNet and ViT are basically tied; ViT is not here to win, it is here
to be the bridge. The real takeaway: borrowed knowledge beats from-scratch on small data.
Transition: and that bridge is tomorrow's whole story.
""")

# --------------------------------------------------------------------------- #
# 12. The bridge to tomorrow  (figure dominant + punch)
# --------------------------------------------------------------------------- #
slide("the-bridge-to-tomorrow", f"""
_fit_image(slide, '{FIG}/concept_bridge.png', left=body_l, top=body_top + 0.05, max_w=body_w, max_h=body_h - 0.65)
_add_text(slide, 'Swap patches for words and you have a language model. See you tomorrow.',
          left=body_l, top=body_top + body_h - 0.50, width=body_w, height=0.42,
          size=18, color_rgb=accent_rgb, font=MONO_FONT, bold=True, align=PP_ALIGN.CENTER)
""", """
End on the bridge, because it makes tomorrow feel inevitable. The Vision Transformer they
just used works by splitting an image into patches and letting attention decide which
patches matter. A large language model does the exact same thing with words instead of
patches. Same machinery, different input; that bottom row in the figure IS an LLM. So
tomorrow, when we point a language model at radiology reports, it is not a new magic
trick, it is the same idea they already built today. Send them off feeling like they have
the key to the next day already in hand.
""")


# --------------------------------------------------------------------------- #
# Added for the hour-long expansion: mechanics + evaluation figure slides
# --------------------------------------------------------------------------- #
fig_slide("augmentation", "concept_augmentation.png", """
We never have enough labeled medical images. Augmentation is the cheapest fix: take each
training image and apply small changes that do not change the right answer, a flip, a
rotation, a brightness shift. The eye is still the same eye, still the same label, but the
pixels are different numbers, so the model effectively sees more variety and is less able to
memorize. Make the point that this is standard practice and basically free. Transition: now
that the image is numbers, what does it actually mean for the model to learn?
""")

fig_slide("split-the-data", "concept_data_split.png", """
This is the single most important habit in machine learning and the easiest to get wrong, so
slow down. We split our data into three piles. The model trains on the biggest one. We use a
validation pile to make choices and watch for trouble. And we lock away a test pile that the
model does not see until the very end, to get an honest grade. The analogy that lands: testing
on data the model already trained on is like giving a student the exact exam questions to
practice, then being amazed they ace the exam. Transition: so how do the knobs actually move?
""")

fig_slide("rolling-downhill", "concept_gradient_descent.png", """
Demystify training itself. The model makes predictions, measures how wrong it is with a number
called the loss, and then nudges every one of its millions of knobs a tiny bit in the direction
that reduces the loss. Do that over and over and the error rolls downhill into a valley, the
best the model can do. That is gradient descent, and it is the engine under literally every
model this week. Keep it intuitive, a ball settling into a bowl; no calculus needed.
Transition: but there is a way this goes wrong that you have to watch for.
""")

fig_slide("overfitting", "concept_overfitting.png", """
Overfitting is the trap that catches everyone. If you train too long, the model stops learning
the general patterns of disease and starts memorizing the exact training images, down to their
quirks. It scores beautifully on data it has seen and falls apart on anything new, which is
useless for real patients. The tell: training accuracy keeps climbing while validation accuracy
stalls or drops. That gap is the warning light, and it is exactly why we kept a separate
validation pile. Transition: with the mechanics in hand, let us build the actual models.
""")

# Rungs 1 and 2 — the two flat-pixel models, side by side
slide("rungs-1-and-2", """
gap = 0.40
cw = (body_w - gap) / 2
ch = body_h * 0.66
models = [('LOGISTIC REGRESSION', 'flatten the image into one long row, then draw a single straight-line boundary between refer and clear', 'the simplest thing that works', TURQUOISE_RGB),
          ('MLP', 'stack a few layers so the boundary can bend into curves', 'more flexible, but still a flat list of numbers', DEEPPINK_RGB)]
for i, (name, what, note, c) in enumerate(models):
    x = body_l + i * (cw + gap)
    _add_rect(slide, left=x, top=body_top, width=cw, height=ch, fill_rgb=PAPER_RGB)
    _add_rect(slide, left=x, top=body_top, width=cw, height=0.95, fill_rgb=c)
    _add_text(slide, name, left=x + 0.28, top=body_top, width=cw - 0.56, height=0.95,
              size=22, color_rgb=_text_on(c), font=MONO_FONT, bold=True, anchor=MSO_ANCHOR.MIDDLE)
    _add_text(slide, what, left=x + 0.28, top=body_top + 1.30, width=cw - 0.56, height=1.5,
              size=16, color_rgb=INK_RGB, font=SANS_FONT)
    _add_text(slide, note, left=x + 0.28, top=body_top + ch - 0.75, width=cw - 0.56, height=0.6,
              size=14, color_rgb=INK_RGB, font=SANS_FONT, italic=True)
cy = body_top + ch + 0.30
_add_rect(slide, left=body_l, top=cy, width=body_w, height=body_h - ch - 0.30, fill_rgb=AMBER_RGB)
_add_text(slide, 'Both throw away the picture. Neither can see that pixels next to each other belong together.',
          left=body_l + 0.35, top=cy, width=body_w - 0.7, height=body_h - ch - 0.30,
          size=17, color_rgb=INK_RGB, font=MONO_FONT, bold=True, anchor=MSO_ANCHOR.MIDDLE)
""", """
Give the two simplest models their due without dwelling. Logistic regression flattens the
image into one long row of numbers and draws a single straight cut between refer and clear; it
is the honest baseline, the dumbest thing that could work. An MLP stacks a few layers so that
cut can bend into curves, more powerful, but it still sees a flat list with no notion of which
pixels are neighbors. The punchline on the amber bar is the whole reason the next rung exists:
both of these threw away the 2D picture. Transition: the CNN is what happens when we stop doing
that.
""")

fig_slide("accuracy-can-lie", "eval_accuracy_lies.png", """
This is the most important evaluation idea, so open the section with the gotcha. In screening,
most people are healthy, so the classes are imbalanced. If 90 out of 100 patients are fine, a
lazy model that just says "healthy, healthy, healthy" to everyone scores 90% accuracy, and yet
it misses every single sick patient, the exact people it was built to catch. A 90% number that
hides total failure. The lesson: in medicine especially, one accuracy number can be worse than
useless. Transition: so we look at all four outcomes instead.
""")

fig_slide("the-confusion-matrix", "eval_confusion.png", """
The confusion matrix replaces one number with four honest outcomes. The model can be right two
ways, correctly refer a sick patient or correctly clear a healthy one. And it can be wrong two
ways that are NOT equal: a false alarm, where it refers a healthy person, costs a needless
specialist visit; a missed case, where it clears a sick person, can cost that person their
sight. Drive the asymmetry home, because it is the entire reason medical AI is evaluated
carefully. Transition: from these four boxes come the two numbers clinicians live by.
""")

fig_slide("the-two-numbers-that-matter", "eval_sens_spec.png", """
Define the two words students will hear constantly in medicine. Sensitivity: of the people who
truly have disease, what fraction did we catch? Specificity: of the people who are truly
healthy, what fraction did we correctly clear? They trade off. For a SCREENING tool we usually
push for high sensitivity, because the cost of missing disease, here blindness, is so much
worse than the cost of a false alarm, a second look. Connect it back: this is why "accuracy"
alone never tells the story. Transition: and you, the builder, get to choose the balance.
""")

fig_slide("you-choose-the-trade-off", "eval_threshold.png", """
End the evaluation section by handing students agency. The model does not output a hard yes or
no; it outputs a confidence, and YOU pick the threshold for action. Set a low bar for referral
and you catch more disease but raise more false alarms; set a high bar and the reverse. There
is no magic point that wins on both, the curves cross. The clinical context decides: a blindness
screener leans toward catching everything. The takeaway: evaluation is a values question, not
just a math question. Transition: enough theory, time to build.
""")

# Where the data comes from / fairness — 3 cards
slide("where-the-data-comes-from", """
cards = [('WHOSE EYES?', 'if the data skews to one country, age, or camera, the model can quietly underperform on everyone else', TURQUOISE_RGB),
         ('CONSENT & PRIVACY', 'these are real patients; who agreed to this use, and how is the data protected?', DEEPPINK_RGB),
         ('FAIRNESS IS TESTABLE', 'report accuracy per subgroup, not just overall; a good average can hide a bad gap', AMBER_RGB)]
n = len(cards)
gap = 0.30
cw = (body_w - (n - 1) * gap) / n
for i, (lab, bod, c) in enumerate(cards):
    x = body_l + i * (cw + gap)
    _add_rect(slide, left=x, top=body_top, width=cw, height=body_h, fill_rgb=PAPER_RGB)
    _add_rect(slide, left=x, top=body_top, width=cw, height=0.90, fill_rgb=c)
    _add_text(slide, lab, left=x + 0.26, top=body_top, width=cw - 0.52, height=0.90,
              size=18, color_rgb=_text_on(c), font=MONO_FONT, bold=True, anchor=MSO_ANCHOR.MIDDLE)
    _add_text(slide, bod, left=x + 0.26, top=body_top + 1.25, width=cw - 0.52, height=body_h - 1.6,
              size=15, color_rgb=INK_RGB, font=SANS_FONT)
""", """
Do not skip the ethics, especially in a healthcare course. A model is only as fair as the data
it learned from, and real deployed systems have been caught working worse on underrepresented
groups, for example performing differently across skin tones or in populations the training set
barely included. Three things to leave them with: ask whose data trained it, because skew
becomes bias; remember these are real patients with a right to consent and privacy; and stress
that fairness is testable, you report accuracy per subgroup, not just one overall number that
can hide a bad gap. Transition: now, finally, what did all this build?
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
