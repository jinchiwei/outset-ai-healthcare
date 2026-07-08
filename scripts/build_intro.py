"""Build the Day 0 intro notebook (lab + solution).

Dead-simple, accessible primer. Branded plots come from nbfig helpers so the student's
code stays a one-liner. Each idea is a worked EXAMPLE, then YOUR TURN on new points.

Currently: TASK 1 only (draw a line to separate two clusters). More sections to follow.

Run:  python scripts/build_intro.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from nbutil import code, md, new_nb, save  # noqa: E402

LAB, SOL = [], []
def both(cell): LAB.append(cell); SOL.append(cell)
def pair(lab_cell, sol_cell): LAB.append(lab_cell); SOL.append(sol_cell)


# --------------------------------------------------------------------------- #
# Part 0 -- Coding fundamentals (~15 min)
# --------------------------------------------------------------------------- #
both(md("""
# Part 0 -- Coding fundamentals (about 15 minutes)

New to Python? This is the whole toolkit you need for today, in a few cells. Already comfortable?
Skim it and jump to Part 1. Run each cell (**Shift+Enter**) and read what it prints.

**First, run this setup cell** (it loads the course's plotting helpers):
"""))

both(code("""
# grab the course helpers (branded plots). On Colab this clones the repo; locally it's a no-op.
import os, sys                                 # os = talk to the computer; sys = talk to Python
if not os.path.exists("../_shared/nbfig.py"):  # if the helper file isn't here yet (we're on Colab)
    os.system("git clone -q https://github.com/jinchiwei/outset-ai-healthcare.git")  # download the repo
    os.chdir("outset-ai-healthcare/notebooks/day0_intro")  # move into the downloaded folder
sys.path.insert(0, "../_shared")   # tell Python it can import files from the _shared folder
import nbfig                        # load the course's plotting helpers
nbfig.use()                         # turn on the branded plot style
"""))

both(md("""
## Variables and printing

A **variable** is just a name for a value. `print(...)` shows things on screen.
"""))

both(code("""
name = "Ada"      # text is called a string
age = 17          # a number
print(name, "is", age, "years old")   # print several things in a row, separated by spaces
print(age + 1)    # you can do math with number variables
"""))

both(md("""
## Lists

A **list** holds many values in order. Positions count from 0, so `scores[0]` is the first one.
"""))

both(code("""
scores = [90, 82, 75, 88]   # a list of four numbers
print("first score:", scores[0])   # [0] grabs the first item (counting starts at 0)
print("how many:", len(scores))    # len() tells you how many items are in the list
scores.append(100)          # add one to the end
print(scores)               # show the whole list
"""))

both(md("""
## Making decisions: `if` / `else`

Run different code depending on whether something is true. Note the indentation -- it marks what's
inside the `if`.
"""))

both(code("""
temperature = 38.5        # a measurement
if temperature > 37.5:    # is it above the fever threshold?
    print("fever")        # runs only when the test above is true
else:                     # otherwise
    print("normal")       # runs when the test is false
"""))

both(md("""
## Repeating: `for` loops

A **for loop** runs the same code once for each item in a list -- so you don't copy-paste.
"""))

both(code("""
for s in [90, 82, 75]:   # run the body once for each number, calling it s
    print(s, "->", "pass" if s >= 60 else "fail")   # pass if s is at least 60, else fail
"""))

both(md("""
Loops are perfect for adding things up: keep a running total and add each item onto it.
"""))

both(code("""
total = 0                   # start the running total at zero
for n in [4, 8, 15, 16]:    # go through each number in turn
    total = total + n       # add each number onto the running total
print("total:", total)      # show the final sum
"""))

pair(
    code("""
# your turn: finish the loop so it adds up these numbers
total = 0                   # start the running total at zero
for n in [10, 20, 30]:      # go through each number in turn
    total = total           # TODO: add n onto total
print("total:", total)      # should print 60
"""),
    code("""
total = 0                   # start the running total at zero
for n in [10, 20, 30]:      # go through each number in turn
    total = total + n       # add each number onto the running total
print("total:", total)      # should print 60
"""),
)

both(md("""
## Functions

A **function** is a reusable recipe: give it **inputs**, it does some work, and `return`s an
**output**. Think of `mail_letter` -- you hand it a stamp, a receiver address, and a sender address,
and it sends the letter:
"""))

both(code("""
nbfig.mail_letter_diagram()   # draw the stamp/receiver/sender picture of a function
"""))

both(md("""
In code, the inputs in the parentheses are called **arguments**, and `return` hands back the output:
"""))

both(code("""
def mail_letter(stamp, receiver_address, sender_address):   # define a function with 3 inputs
    print("Stamp: ", stamp)              # show the stamp input
    print("To:    ", receiver_address)   # show the receiver input
    print("From:  ", sender_address)     # show the sender input
    return "letter is on its way!"       # hand back this text as the output

result = mail_letter("Forever stamp", "123 Main St", "500 Oak Ave")   # call it, keep the output
print(result)                            # show what the function returned
"""))

both(md("""
The same idea with numbers -- define it once, call it as often as you like:
"""))

both(code("""
def average(numbers):                    # a function that takes a list of numbers
    return sum(numbers) / len(numbers)   # add them all up, divide by how many -> the average

print(average([90, 82, 75, 88]))         # call it on four numbers
print(average([10, 20]))                 # call it again on two numbers
"""))

pair(
    code("""
# your turn: finish this function so it returns the number doubled
def double(x):          # a function that takes one number, x
    return              # TODO: return x times 2

print(double(4))        # should print 8
"""),
    code("""
def double(x):          # a function that takes one number, x
    return x * 2        # hand back x times 2

print(double(4))        # should print 8
"""),
)

both(md("""
That's the whole toolkit -- variables, lists, `if`, `for` loops, and functions. You'll use exactly
these for the rest of the notebook. On to the machine learning.
"""))


# --------------------------------------------------------------------------- #
# Part 1 -- one line
# --------------------------------------------------------------------------- #
both(md("""
# Machine learning starts with a line

Here are a few **healthy** patients (turquoise) and a few **diseased** patients (deeppink). Each
patient is two measurements, an `x` and a `y`. A huge amount of machine learning is just this:
**find a line that separates the groups.** You write the line yourself as `y = m*x + b` -- `m` is
the slope, `b` is the height. A helper draws it and counts how many patients land on the right side.
"""))

both(md("""
## Example -- first, look at the patients

Six healthy patients and six diseased patients are made for you. Each has an `x` (across) and a
`y` (up). Run this to see where they sit and read off their measurements.
"""))

both(code("""
healthy, diseased = nbfig.two_clusters(seed=0)   # a few healthy + a few diseased patients
nbfig.show_points(healthy, diseased)             # scatter + a table of every measurement
"""))

both(md("""
Now draw a line through them. The line is `y = m*x + b`. Here `m = 1` and `b = 0`, so `y = 1*x + 0`.
You build the line's `y` values yourself, then hand the patients and your line to the helper.
"""))

both(code("""
x = nbfig.xgrid(healthy, diseased)         # the x-values to draw your line over
y = 1 * x + 0                              # your line:  y = m*x + b

nbfig.show_line(healthy, diseased, x, y)   # draws your line and scores it
"""))

both(md("""
## Your turn

New patients: healthy sit in the **lower-left**, diseased in the **upper-right**. Look at their
measurements first.
"""))

both(code("""
healthy, diseased = nbfig.two_clusters(seed=9, healthy_center=(1, 1), diseased_center=(6, 6))  # new patients
nbfig.show_points(healthy, diseased)   # scatter + a table of every measurement
"""))

both(md("""
Now write your own line. The example line `y = 1*x + 0` runs straight through both groups and gets
4 patients wrong -- change `m` and `b` (the two numbers) until the line sits between them.
"""))

pair(
    code("""
x = nbfig.xgrid(healthy, diseased)   # the x-values to draw your line over
y = 1 * x + 0        # TODO: change the slope and height so the line separates healthy from diseased

nbfig.show_line(healthy, diseased, x, y)   # draws your line and scores it
"""),
    code("""
x = nbfig.xgrid(healthy, diseased)   # the x-values to draw your line over
y = 0 * x + 3.5      # a flat line at height 3.5 splits healthy (low) from diseased (high)

nbfig.show_line(healthy, diseased, x, y)   # draws your line and scores it
"""),
)


# --------------------------------------------------------------------------- #
# Task 2 -- when one line isn't enough (XOR): two lines / a band
# --------------------------------------------------------------------------- #
both(md("""
# When one line isn't enough

Sometimes no single line can separate the groups. Here the **diseased** patients sit in *two*
corners (lower-left and upper-right) with **healthy** patients in the other two corners. Try to
imagine a single straight line that puts both diseased corners on one side -- you can't.
"""))

both(code("""
healthy, diseased = nbfig.xor_clusters(seed=0)   # two diseased corners + two healthy corners
nbfig.show_points(healthy, diseased, table=False)   # scatter only (table=False hides the number table)
"""))

both(md("""
But **two** lines can. The diseased patients lie along a diagonal stripe. Draw two parallel lines
that bracket that stripe -- diseased patients fall **inside** the band, healthy **outside**. Here
the band is `y = 1*x + 2.5` (top edge) and `y = 1*x - 2.5` (bottom edge).
"""))

both(code("""
x = nbfig.xgrid(healthy, diseased)   # the x-values to draw both lines over
y1 = 1 * x + 2.5     # top edge of the band
y2 = 1 * x - 2.5     # bottom edge of the band

nbfig.show_band(healthy, diseased, x, y1, y2)   # diseased inside the band, healthy outside
"""))

both(md("""
## Your turn

New patients. The diseased group now sits in a **flat horizontal stripe** across the middle
(at `y = 3`), with healthy patients above and below it. Look at them first.
"""))

both(code("""
# diseased now sit in a flat stripe (both at y=3); healthy sit above and below it
healthy, diseased = nbfig.xor_clusters(seed=0,
    diseased_centers=((1, 3), (6, 3)), healthy_centers=((3, 1), (3, 6)))
nbfig.show_points(healthy, diseased, table=False)   # scatter only, no number table
"""))

both(md("""
The example's slanted band won't fit a flat stripe -- you need **different** lines. Both lines
should now be flat (slope `0`). Change the slope and the two heights until the band holds the
diseased patients.
"""))

pair(
    code("""
x = nbfig.xgrid(healthy, diseased)   # the x-values to draw both lines over
y1 = 1 * x + 2.5     # TODO: make both lines flat (slope 0) and bracket the diseased stripe
y2 = 1 * x - 2.5     # the other edge of the band

nbfig.show_band(healthy, diseased, x, y1, y2)   # diseased inside the band, healthy outside
"""),
    code("""
x = nbfig.xgrid(healthy, diseased)   # the x-values to draw both lines over
y1 = 0 * x + 4       # a flat band from y=2 to y=4 holds the diseased stripe at y=3
y2 = 0 * x + 2       # the bottom edge of that flat band

nbfig.show_band(healthy, diseased, x, y1, y2)   # diseased inside the band, healthy outside
"""),
)


# --------------------------------------------------------------------------- #
# Task 3 -- learning: the computer finds m and b by itself
# --------------------------------------------------------------------------- #
both(md(r"""
# Letting the computer find the line

So far *you* picked `m` and `b`. A neural network does it automatically:

1. **Start with a flat line** (`m = 0`, `b = 0`) -- it's bad.
2. **Measure the loss** -- one number for how wrong the line is.
3. **Nudge `m` and `b`** a little to make the loss smaller. This nudge is *gradient descent*.
4. **Repeat.** Each pass over all the patients is one *epoch*.

### How the loss is calculated

For each patient we turn "how far above the line" into a probability of being diseased:

$$p = \text{sigmoid}\big(\,y - (m x + b)\,\big) \qquad \text{sigmoid}(z) = \frac{1}{1 + e^{-z}}$$

`p` near 1 means "confidently above the line" (we guess diseased), near 0 means "below" (healthy).
Then the loss compares `p` to the truth `t` (`t = 1` diseased, `t = 0` healthy), averaged over all N patients:

$$\text{loss} = -\frac{1}{N} \sum \Big[\, t\,\log(p) + (1 - t)\,\log(1 - p) \,\Big]$$

If the line is confident **and right**, `p` matches `t` and that patient adds ~0. If it's confident and
**wrong**, `log` blows up and the loss is large. Learning = making this number small.

**In plain terms:** for each patient we make a guess `p`, penalize being *confidently wrong*, and
average the penalty over everyone. The two halves of the picture below:
"""))

both(code("""
nbfig.loss_explainer()   # left: how the line becomes a probability   right: the penalty
"""))

both(md("""
Read the left panel: the bigger the line's score `z` for a patient, the closer sigmoid pushes `p`
toward 1 (diseased); a very negative `z` gets pushed toward 0 (healthy). Read the right panel:
whatever the true answer is, we look at the chance we gave *that* answer -- confident and right costs
almost nothing, confident and wrong is punished hardest.

Here is the whole learner -- read it top to bottom, it's just the four steps above in a loop:
"""))

both(code("""
import numpy as np                  # numpy: fast math on whole arrays of numbers at once

def sigmoid(z):                      # squash any number into a 0-to-1 probability
    return 1 / (1 + np.exp(-np.clip(z, -30, 30)))   # clip keeps exp() from overflowing

def learn_line(healthy, diseased, lr=0.5, epochs=50):   # lr = step size, epochs = how many passes
    pts    = np.vstack([healthy, diseased])                      # all patients, stacked
    truth  = np.array([0] * len(healthy) + [1] * len(diseased))  # healthy = 0, diseased = 1
    x, y   = pts[:, 0], pts[:, 1]                                # split into x-column and y-column

    m, b = 0.0, 0.0                # 1. start with a flat line
    lines, losses = [], []         # remember the line and loss at each step, to plot later
    for epoch in range(epochs):    # do one full pass per epoch
        p    = sigmoid(y - (m * x + b))                          # our guess for each patient
        loss = -np.mean(truth * np.log(p) + (1 - truth) * np.log(1 - p))   # 2. the loss equation
        lines.append((m, b)); losses.append(loss)   # save this step's line and loss

        error = p - truth              # 3. how wrong we are on each patient
        m = m - lr * np.mean(error * -x)   #    nudge the slope  to lower the loss
        b = b - lr * np.mean(error * -1)   #    nudge the height to lower the loss

    # report the final line and its loss, formatted to 2-3 decimals
    print(f"learned line:  y = {m:.2f}*x + {b:.2f}   (final loss {losses[-1]:.3f})")
    return nbfig.show_learning(healthy, diseased, lines, losses)   # animate the line learning
"""))

both(md("""
Now run it and watch the line swing from its flat start (gray) into the learned line (pink) as the loss drops.
"""))

both(code("""
healthy, diseased = nbfig.two_clusters(seed=0)   # the same easy patients as before
learn_line(healthy, diseased)                    # let the computer find m and b
"""))

both(md("""
## Your turn 1 -- pick a learning rate

Run the learner on new patients, but with a tiny `lr = 0.01`. Watch the loss: it barely moves and the
line never arrives -- the steps are too small. **Make `lr` bigger** until it actually learns.
"""))

pair(
    code("""
healthy, diseased = nbfig.two_clusters(seed=9, healthy_center=(1, 1), diseased_center=(6, 6))  # new patients
learn_line(healthy, diseased, lr=0.01, epochs=50)   # TODO: the loss barely moves -- make lr bigger
"""),
    code("""
healthy, diseased = nbfig.two_clusters(seed=9, healthy_center=(1, 1), diseased_center=(6, 6))  # new patients
learn_line(healthy, diseased, lr=0.5, epochs=50)    # a bigger step size actually learns
"""),
)

both(md("""
## Your turn 2 -- when no learning rate works

A different lab reports the same kind of patients but in different units, so the measurements are in the
**hundreds**. Run the learner with a good `lr = 0.5`:
"""))

both(code("""
# same kind of patients, but measured in the hundreds instead of single digits
healthy, diseased = nbfig.two_clusters(seed=9, healthy_center=(100, 100),
                                       diseased_center=(600, 600), spread=70, round_to=0)
learn_line(healthy, diseased, lr=0.5, epochs=50)   # a good step size that worked before
"""))

both(md("""
It's stuck -- a jagged loss and a line that never separates them. This time **changing `lr` won't help**
(try it): the numbers are so large that every step overshoots. The fix is to **normalize** -- rescale
each measurement to about -1 to 1 by subtracting the average and dividing by the spread. Finish the two
lines below and re-run:
"""))

pair(
    code("""
allpts = np.vstack([healthy, diseased])   # stack all patients to measure them together
mean, spread = allpts.mean(axis=0), allpts.std(axis=0)   # the average and the spread of each column

healthy_n  = healthy       # TODO: change to (healthy  - mean) / spread
diseased_n = diseased      # TODO: change to (diseased - mean) / spread

learn_line(healthy_n, diseased_n, lr=0.5, epochs=50)   # now train on the rescaled patients
"""),
    code("""
allpts = np.vstack([healthy, diseased])   # stack all patients to measure them together
mean, spread = allpts.mean(axis=0), allpts.std(axis=0)   # the average and the spread of each column

healthy_n  = (healthy  - mean) / spread    # normalize: centered near 0, range about -1 to 1
diseased_n = (diseased - mean) / spread    # rescale the diseased patients the same way

learn_line(healthy_n, diseased_n, lr=0.5, epochs=50)   # now train on the rescaled patients
"""),
)

both(md("""
Same data, same learning rate -- it only failed because of the *scale*. Normalizing is one of the first
things real ML pipelines do, for exactly this reason.

That's the whole idea of a neural network. **One** line (like here) is a single neuron. Task 2 showed
some problems need **two** lines -- so real networks stack many neurons, and gradient descent tunes all
of their `m`'s and `b`'s at once.
"""))


# --------------------------------------------------------------------------- #
# Part 4 -- computer vision: images are grids of numbers
# --------------------------------------------------------------------------- #
both(md("""
# Part 4 -- Computer vision

An image is just a **grid of numbers**. Each number is one pixel's brightness, from 0 (black) to
255 (white). So "editing an image" is really just doing math on that grid. Here is a handwritten `5`:
"""))

both(code("""
import numpy as np                # numpy again: it handles the image grid as an array

digit = nbfig.sample_digit()     # a 28x28 grid of brightness values
print("shape:", digit.shape)     # .shape tells you the grid's size: (rows, columns)
print(digit[8:14, 8:14])         # peek at a 6x6 corner -- just numbers
nbfig.show_images([("this is a 5", digit)])   # draw the grid as a picture
"""))

both(md("""
A few classic operations, written as functions you can read (they use the `for` loops and functions
from Part 0):

- **threshold** -- snap every pixel to pure black or white
- **erode** -- shrink the bright strokes (each pixel becomes the *darkest* of its neighbors)
- **dilate** -- grow the bright strokes (each pixel becomes the *brightest* of its neighbors)
- **outline** -- the edges, found by subtracting the eroded image from the dilated one
"""))

both(code("""
def erode(img):                      # shrink: each pixel becomes its DARKEST neighbor
    out = img.copy()                 # start from a copy so we don't overwrite as we go
    for i in range(1, img.shape[0] - 1):   # every row except the very edges
        for j in range(1, img.shape[1] - 1):   # every column except the very edges
            out[i, j] = img[i-1:i+2, j-1:j+2].min(axis=(0, 1))   # axis=(0,1): per color channel
    return out                       # hand back the shrunk image

def dilate(img):                     # grow: each pixel becomes its BRIGHTEST neighbor
    out = img.copy()                 # start from a copy so we don't overwrite as we go
    for i in range(1, img.shape[0] - 1):   # every row except the very edges
        for j in range(1, img.shape[1] - 1):   # every column except the very edges
            out[i, j] = img[i-1:i+2, j-1:j+2].max(axis=(0, 1))   # axis=(0,1): per color channel
    return out                       # hand back the grown image

def threshold(img, cutoff=127):      # force every pixel to pure black or white
    return np.where(img > cutoff, 255, 0)   # above cutoff -> 255 (white), else 0 (black)

def outline(img):                    # edges = what dilation grew, minus what erosion shrank
    return dilate(img).astype(int) - erode(img).astype(int)   # astype(int) avoids negative wrap-around
"""))

both(code("""
nbfig.show_images([                        # draw all five versions side by side, each with a label
    ("original (a 5)", digit),
    ("threshold",      threshold(digit)),
    ("erode",          erode(digit)),
    ("dilate",         dilate(digit)),
    ("outline",        outline(digit)),
])
"""))

both(md("""
## Your turn -- the same operations on a retina

Day 1 works with **retina** photos (the back of the eye) to grade diabetic retinopathy. A retina photo
is a **color** image -- the same grid of numbers, but with three channels (red, green, blue) instead of
one. The four functions above already handle color: the `axis=(0, 1)` in `erode`/`dilate` takes the
min/max of each channel separately. So you can run the **exact same operations** on it.

Here's the retina -- only the loading is done for you:
"""))

both(code("""
retina = nbfig.sample_retina()       # a real APTOS retina photo: 96x96 with 3 color channels
print("shape:", retina.shape)        # note the extra 3 at the end: red, green, blue
nbfig.show_images([("a retina", retina)])   # draw the color photo
"""))

both(md("""
Now **type the operations** yourself -- call `threshold`, `erode`, `dilate`, and `outline` on `retina`,
exactly like you did for the 5.
"""))

pair(
    code("""
nbfig.show_images([                    # draw the retina and each operation side by side
    ("original",  retina),
    ("threshold", threshold(retina)),
    # TODO: add erode(retina), dilate(retina), outline(retina) the same way
])
"""),
    code("""
nbfig.show_images([                    # draw the retina and each operation side by side
    ("original",  retina),
    ("threshold", threshold(retina)),
    ("erode",     erode(retina)),
    ("dilate",    dilate(retina)),
    ("outline",   outline(retina)),
])
"""),
)

both(md("""
Same four operations, now on a real medical image. Look at **outline** -- it traces the rim of the
retina and lights up the bright optic disc and the vessels. Edge-finding like this is a building block
many medical-imaging models use.
"""))

both(md("""
## Segmentation -- labeling every pixel

Editing an image is one thing; **segmentation** is another: deciding *what each pixel is*. In eye care,
programs segment the **blood vessels** and the **optic disc** (the bright spot where the optic nerve
leaves the eye) so they can be measured automatically. We ran a vessel-finding filter and an
optic-disc detector on this retina ahead of time -- run this to see the two maps drawn on top:
"""))

both(code("""
vessels = nbfig.sample_retina_vessels()   # a mask: True where the pixel is a vessel
disc    = nbfig.sample_retina_disc()      # a mask: True where the optic disc is
nbfig.show_segmentation(retina, vessels, disc)   # draw both masks on top of the photo
"""))

both(md("""
Each map is just a grid of True/False -- **one label per pixel** -- laid over the photo (vessels in
turquoise, the optic disc in pink). Producing masks like these automatically is exactly what medical
image-**segmentation** models are trained to do.
"""))


# --------------------------------------------------------------------------- #
# Part 5 -- how a neural network works (standalone)
# --------------------------------------------------------------------------- #
both(md("""
# Part 5 -- How a neural network works

You now have every idea behind a real neural network: **functions** and **imports** from Part 0, the
**neuron** (`y = m*x + b`) from Part 3, and the **training loop** from Part 3. Here's how real code --
the kind behind medical-imaging AI -- snaps those pieces together. Every line is written out and
commented, nothing hidden.
"""))

both(md("""
## Importing a prebuilt package

You wrote your own functions with `def`. A **package** (or library) is a big box of ready-made
functions someone else already wrote and tested -- you `import` it and use its tools. You've been
doing this all along with `numpy`. Neural networks use **PyTorch**:
"""))

both(code("""
import torch                 # PyTorch: the standard toolkit for neural networks
import torch.nn as nn        # nn = the neural-network building blocks (layers, etc.)

print("torch version:", torch.__version__)   # confirm PyTorch loaded, and which version
"""))

both(md("""
## Defining an MLP

In Part 3, one line `y = m*x + b` was a single **neuron**. Put many neurons side by side and you have
a **layer**; stack layers and you have a **multi-layer perceptron (MLP)** -- a network that can bend
into shapes one straight line can't (remember Part 2's XOR).
"""))

both(code("""
nbfig.mlp_diagram()   # a picture: input neurons -> hidden layer -> output
"""))

both(md("""
Here's how you define one in PyTorch. Read the comments top to bottom:
"""))

both(code("""
# nn.Linear(in, out)  is one whole LAYER of neurons (each neuron is a y = m*x + b).
# nn.ReLU()           is the "bend" between layers, so the network can learn curves.
# nn.Sequential(...)  just runs these steps in order, top to bottom.

mlp = nn.Sequential(               # build the network by listing its steps in order
    nn.Linear(3 * 64 * 64, 256),   # layer 1: a flattened image (3 colors x 64 x 64) -> 256 neurons
    nn.ReLU(),                     # bend
    nn.Linear(256, 128),           # layer 2: 256 -> 128 neurons
    nn.ReLU(),                     # bend
    nn.Linear(128, 2),             # output layer: 2 scores, one for "no" and one for "yes"
)

print(mlp)   # printing the model shows its layers
"""))

both(md("""
### Your turn

Add one more hidden layer. Copy the pattern -- an `nn.Linear` followed by an `nn.ReLU` -- and make the
output layer start from your new layer's size.
"""))

pair(
    code("""
my_mlp = nn.Sequential(            # your copy of the network to extend
    nn.Linear(3 * 64 * 64, 256),   # layer 1: flattened image -> 256 neurons
    nn.ReLU(),                     # bend
    nn.Linear(256, 128),           # layer 2: 256 -> 128 neurons
    nn.ReLU(),                     # bend
    # TODO: add  nn.Linear(128, 64),  then  nn.ReLU(),  here
    nn.Linear(128, 2),   # TODO: once you add the layer above, change 128 to 64
)
print(my_mlp)                      # show the layers you built
"""),
    code("""
my_mlp = nn.Sequential(            # your copy of the network to extend
    nn.Linear(3 * 64 * 64, 256),   # layer 1: flattened image -> 256 neurons
    nn.ReLU(),                     # bend
    nn.Linear(256, 128),           # layer 2: 256 -> 128 neurons
    nn.ReLU(),                     # bend
    nn.Linear(128, 64),   # the new hidden layer
    nn.ReLU(),            # bend after the new layer
    nn.Linear(64, 2),     # output now starts from 64
)
print(my_mlp)                      # show the layers you built
"""),
)

both(md("""
## Loading a pretrained model (ResNet)

You don't build a giant model like **ResNet** by hand -- someone already trained it on millions of
images. You **import it** from a public package and reuse what it learned, swapping only the last
layer for your own question. That reuse is called **transfer learning** -- the single most useful
trick in modern computer vision.

We'll also count each model's **parameters** -- the tunable numbers it learns. That count is a rough
measure of how much a model can capture, and it's how you feel *just how much bigger* ResNet is than
the little MLP you just built.

**How is a parameter count calculated?** Each `nn.Linear(in, out)` layer holds an `in x out` grid of
weights, plus one `out` bias -- so `in*out + out` numbers. For your MLP's first layer that's
`12288 * 256 + 256 = 3,145,984`. Add up every layer and you get the model's size:
"""))

both(code("""
from torchvision.models import resnet50   # a famous image model, free to use

resnet = resnet50(weights=None)   # weights=None just builds the shape (fast, no download)
# ^ To reuse what it learned, you pass the PRETRAINED weights instead -- that's the whole trick:
#   resnet50(weights="IMAGENET1K_V2")   # it arrives already knowing how to see

# ResNet's last layer predicts 1000 categories. We replace it with OUR question:
# 2 classes (yes / no). Everything before it is reused as-is.
resnet.fc = nn.Linear(resnet.fc.in_features, 2)   # replace the final layer with a 2-class one

# A "parameter" is one tunable number. An nn.Linear(in, out) holds  in*out  weights
# plus  out  biases. Add that over every layer -- here's our MLP, layer by layer:
for layer in mlp:
    n = sum(p.numel() for p in layer.parameters())   # how many numbers this one layer holds
    if n > 0:                                         # ReLU layers hold none -- skip them
        print(f"{layer}  ->  {n:,} parameters")       # print this layer and its parameter count

mlp_params    = sum(p.numel() for p in mlp.parameters())      # every MLP layer added up
resnet_params = sum(p.numel() for p in resnet.parameters())   # the same count, for ResNet50
print(f"\\nour MLP:  {mlp_params:>12,} parameters")   # \\n = blank line first; :>12, = right-aligned with commas
print(f"ResNet50: {resnet_params:>12,} parameters")   # ResNet's total, lined up under the MLP's
print(f"-> ResNet50 has about {resnet_params // mlp_params}x more knobs to tune")   # // = whole-number divide
"""))

both(md("""
### Your turn

Swap in a different prebuilt model and compare its size. `resnet18` is a smaller cousin of `resnet50`.
"""))

pair(
    code("""
from torchvision.models import resnet18   # a smaller cousin of resnet50

model = resnet18(weights=None)            # TODO: try resnet50 here and compare the count
model.fc = nn.Linear(model.fc.in_features, 2)   # same swap: a new 2-class head
print("params:", sum(p.numel() for p in model.parameters()))   # add up every tunable number
"""),
    code("""
from torchvision.models import resnet50   # the bigger model, for comparison

model = resnet50(weights=None)            # ~25 million params vs resnet18's ~11 million
model.fc = nn.Linear(model.fc.in_features, 2)   # same swap: a new 2-class head
print("params:", sum(p.numel() for p in model.parameters()))   # add up every tunable number
"""),
)

both(md("""
## What a trained network "sees"

A network learns in **stages**. Early layers react to simple things like edges; middle layers combine
those into textures; the deepest layers respond to whole abstract regions. Below are real feature maps
from a **pretrained** ResNet run on our retina (computed ahead of time, so nothing downloads) -- four
channels at each of three depths:
"""))

both(code("""
early, middle, late = nbfig.sample_feature_maps()   # a pretrained ResNet's response, at 3 depths
nbfig.show_feature_maps(early, middle, late)   # draw four channels at each of the three depths
"""))

both(md("""
Notice how the pictures get **coarser and more abstract** with depth: crisp edges near the input, then
textures, then low-resolution blobs standing in for high-level concepts. That layered build-up, simple
parts combining into complex ones, is why deep networks see so well.
"""))

both(md("""
## Training is the loop from Part 3

Training any of these models is the exact loop you wrote by hand in Part 3: **guess -> measure the
loss -> nudge the weights -> repeat.** PyTorch just does the calculus (the gradients) for you. Here's
one real training step on a tiny fake batch so you can watch the loss move:
"""))

both(code("""
# a tiny pretend batch so we can run a real training step (no data download needed)
fake_images = torch.randn(8, 3 * 64 * 64)     # 8 fake flattened images
fake_labels = torch.randint(0, 2, (8,))       # 8 fake yes/no labels

loss_fn   = nn.CrossEntropyLoss()                       # the loss (Part 3's idea, for 2+ classes)
optimizer = torch.optim.SGD(mlp.parameters(), lr=0.01)  # SGD = gradient descent, from Part 3

for step in range(5):                      # 5 training steps
    scores = mlp(fake_images)              # 1. the model guesses
    loss   = loss_fn(scores, fake_labels)  # 2. measure how wrong it is
    optimizer.zero_grad()                  #    clear the old gradients
    loss.backward()                        # 3. PyTorch computes the gradients for us
    optimizer.step()                       # 4. nudge every weight a little downhill
    print("step", step, "  loss", round(loss.item(), 3))   # watch the loss shrink each step
"""))

both(md("""
That's a neural network in a nutshell: **import** a public toolkit, **define** an MLP or **load** a
pretrained model, swap the last layer, and **train** with the same loop from Part 3. The models behind
real medical AI are just bigger versions of exactly these pieces.
"""))


def build():
    out = ROOT / "notebooks/day0_intro"
    out.mkdir(parents=True, exist_ok=True)
    lab = new_nb(); lab.cells = LAB; save(lab, out / "intro.ipynb")
    sol = new_nb(); sol.cells = SOL; save(sol, out / "intro_solution.ipynb")
    print(f"wrote intro.ipynb ({len(LAB)} cells) + intro_solution.ipynb")


if __name__ == "__main__":
    build()
