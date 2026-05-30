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
# 1. AI already screens for blindness  (figure: dr_screening + 3 chips)
# --------------------------------------------------------------------------- #
slide("ai-already-screens", f"""
fig_h = body_h * 0.64
_fit_image(slide, '{FIG}/intro_dr_screening.png', left=body_l, top=body_top, max_w=body_w, max_h=fig_h)
chip_y = body_top + fig_h + 0.22
ch = body_h - fig_h - 0.30
gap = 0.26
cw = (body_w - 2 * gap) / 3
chips = [('THE DATA', 'color photos of the retina', TURQUOISE_RGB),
         ('THE TASK', 'refer this eye, yes or no?', DEEPPINK_RGB),
         ('WHY FAMOUS', 'first medical AI at scale', AMBER_RGB)]
for i, (lab, bod, c) in enumerate(chips):
    x = body_l + i * (cw + gap)
    _add_rect(slide, left=x, top=chip_y, width=cw, height=ch, fill_rgb=c)
    _add_text(slide, lab, left=x + 0.20, top=chip_y + 0.14, width=cw - 0.40, height=0.34,
              size=13, color_rgb=_text_on(c), font=MONO_FONT, bold=True)
    _add_text(slide, bod, left=x + 0.20, top=chip_y + 0.52, width=cw - 0.40, height=ch - 0.62,
              size=15, color_rgb=_text_on(c), font=SANS_FONT)
""", """
Diabetic retinopathy is damage to blood vessels in the retina caused by diabetes. Left
unchecked it causes blindness, but if you catch it early and send the patient to a
specialist, that blindness is preventable. The catch is scale: hundreds of millions of
diabetics, nowhere near enough eye doctors to screen them all. That mismatch is exactly
what an automated screener fixes, which is why this was one of the first medical AIs
deployed in real clinics. Frame the day: by the end, every student will have built a
model that makes this refer-or-not call. Transition: but first, what is the model even
looking at?
""")

# --------------------------------------------------------------------------- #
# 2. Real systems, real failure modes  (2x2 color zones)
# --------------------------------------------------------------------------- #
slide("real-systems-real-failure", """
gap = 0.30
cw = (body_w - gap) / 2
ch = (body_h - gap) / 2
items = [('DEPLOYED TODAY', 'DR screening, mammogram triage, sepsis alerts: all in real hospitals now', TURQUOISE_RGB),
         ('SHORTCUT LEARNING', 'the model reads the ruler in the image, not the tumor', DEEPPINK_RGB),
         ('DISTRIBUTION SHIFT', 'your hospital is not the training hospital, so accuracy quietly drops', AMBER_RGB),
         ('OVERCONFIDENCE', '"90% sure" can mean nothing; confident and wrong is the dangerous mix', BLUEVIOLET_RGB)]
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
Medical AI is not science fiction; it is already treating patients. But the interesting
part for students is how it fails, because the failures are subtle and human. Shortcut
learning: a model told to spot tumors instead learns that scans WITH tumors happened to
have a ruler laid next to them, so it just detects rulers. Distribution shift: it was
trained on one hospital's scanner and quietly gets worse on yours. Overconfidence: it
reports 90% confidence on everything because nobody calibrated it. The lesson: knowing
how these break is the real skill, more than chasing a higher accuracy number.
Transition: to understand any of this, we have to start with what an image actually is.
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
slide("five-models-one-ladder", """
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
slide("resnet-borrow-a-trained-brain", f"""
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
slide("vision-transformer-patches", f"""
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
