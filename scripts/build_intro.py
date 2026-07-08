"""Build the Day 0 intro notebook (lab + solution).

A from-scratch primer: pure numpy/matplotlib/sklearn, NO course helpers, NO Claude assumed.
Each concept = a worked EXAMPLE, then YOUR TURN on slightly different data (student writes it,
copying the example). Markdown is deliberately sparse.

Arc: fit a line -> flowers (a neuron is a line) -> many neurons = many lines -> images are tensors.

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
both(md("""
# From a line to a neural network

The whole idea of machine learning: **find numbers that make a model fit data.** Pure
`numpy` + `matplotlib`, nothing hidden. Each idea is a worked **example**, then it's **your
turn** on slightly different data. Copy the example, change what needs changing.
"""))

both(code("""
import numpy as np
import matplotlib.pyplot as plt
"""))

# ---- 1. Fit a line -------------------------------------------------------- #
both(md("""
## 1. Fit a line

A model is `y = w*x + b`. "Learning" = nudging `w` and `b` to shrink the error (how far the
line is from the points).
"""))

both(code("""
# EXAMPLE: make noisy points that follow a line, then learn the line back.
rng = np.random.default_rng(0)
x = np.linspace(0, 10, 40)
y = 2.3 * x + 1.0 + rng.normal(0, 2, size=x.size)     # true slope 2.3, intercept 1, + noise

w, b = 0.0, 0.0                    # start flat
lr = 0.002                         # step size
for step in range(400):
    yhat = w * x + b               # prediction
    err  = yhat - y
    w -= lr * 2 * (err * x).mean() # nudge w downhill
    b -= lr * 2 * err.mean()       # nudge b downhill
print(f"learned:  y = {w:.2f} x + {b:.2f}   (true was 2.3 x + 1.0)")

plt.scatter(x, y, label="data")
plt.plot(x, w * x + b, color="red", label="fitted line")
plt.legend(); plt.show()
"""))

both(md("""
**Your turn.** Here's different data (steeper *down*, noisier). Fit a line the same way.
"""))

pair(
    code("""
x2 = np.linspace(0, 10, 40)
y2 = -1.5 * x2 + 8.0 + rng.normal(0, 3, size=x2.size)

# your turn: copy the loop from the example to learn w, b for (x2, y2),
# then scatter the data and plot your fitted line.
"""),
    code("""
x2 = np.linspace(0, 10, 40)
y2 = -1.5 * x2 + 8.0 + rng.normal(0, 3, size=x2.size)

w, b = 0.0, 0.0
lr = 0.002
for step in range(400):
    err = (w * x2 + b) - y2
    w -= lr * 2 * (err * x2).mean()
    b -= lr * 2 * err.mean()
print(f"learned:  y = {w:.2f} x + {b:.2f}")

plt.scatter(x2, y2, label="data")
plt.plot(x2, w * x2 + b, color="red", label="fitted line")
plt.legend(); plt.show()
"""),
)

# ---- 2. A neuron is a line ------------------------------------------------ #
both(md("""
## 2. A neuron is a line

Load real flowers. A single **neuron** computes `w·x + b`, then squashes it to a yes/no with
`sigmoid`. Its decision boundary is just a straight line.
"""))

both(code("""
from sklearn.datasets import load_iris
iris = load_iris()

# two petal features, two species: versicolor (0) vs virginica (1)
keep = iris.target != 0
X = iris.data[keep][:, [2, 3]]                 # petal length, petal width
y = (iris.target[keep] == 2).astype(int)
X = (X - X.mean(0)) / X.std(0)                 # standardize (helps training)

def sigmoid(z): return 1 / (1 + np.exp(-np.clip(z, -30, 30)))

# EXAMPLE: train one neuron (logistic regression) by gradient descent.
w = np.zeros(2); b = 0.0
for step in range(3000):
    p = sigmoid(X @ w + b)
    w -= 0.1 * X.T @ (p - y) / len(y)
    b -= 0.1 * (p - y).mean()
print(f"one-neuron accuracy: {((sigmoid(X @ w + b) > 0.5).astype(int) == y).mean():.2f}")

# draw the straight-line boundary
xx, yy = np.meshgrid(np.linspace(-2.5, 2.5, 200), np.linspace(-2.5, 2.5, 200))
grid = sigmoid(np.c_[xx.ravel(), yy.ravel()] @ w + b).reshape(xx.shape)
plt.contourf(xx, yy, grid, levels=20, cmap="coolwarm", alpha=0.6)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", edgecolor="k")
plt.xlabel("petal length"); plt.ylabel("petal width"); plt.show()
"""))

both(md("""
**Your turn.** Train the same one neuron on the two **sepal** features instead. Print the accuracy.
"""))

pair(
    code("""
X2 = iris.data[keep][:, [0, 1]]                # sepal length, sepal width
X2 = (X2 - X2.mean(0)) / X2.std(0)

# your turn: train one neuron on (X2, y) with the same gradient-descent loop,
# then print the accuracy. (Is it higher or lower than with the petal features?)
"""),
    code("""
X2 = iris.data[keep][:, [0, 1]]
X2 = (X2 - X2.mean(0)) / X2.std(0)

w, b = np.zeros(2), 0.0
for step in range(3000):
    p = sigmoid(X2 @ w + b)
    w -= 0.1 * X2.T @ (p - y) / len(y)
    b -= 0.1 * (p - y).mean()
print(f"sepal one-neuron accuracy: {((sigmoid(X2 @ w + b) > 0.5).astype(int) == y).mean():.2f}")
"""),
)

# ---- 3. Many neurons = many lines ----------------------------------------- #
both(md("""
## 3. Many neurons = many lines

One line can't bend. A **hidden layer** of `H` neurons draws `H` lines and combines them into
a curved boundary. Here's data one line *can't* separate.
"""))

both(code("""
from sklearn.datasets import make_moons
Xm, ym = make_moons(n_samples=300, noise=0.2, random_state=0)
Xm = (Xm - Xm.mean(0)) / Xm.std(0)

# a tiny neural net, from scratch: one hidden layer of H tanh neurons -> one output.
def train_mlp(X, y, H, steps=4000, lr=0.5, seed=0):
    g = np.random.default_rng(seed)
    W1 = g.normal(0, 0.5, (X.shape[1], H)); b1 = np.zeros(H)
    W2 = g.normal(0, 0.5, (H, 1));          b2 = np.zeros(1)
    Y = y.reshape(-1, 1)
    for _ in range(steps):
        A1 = np.tanh(X @ W1 + b1)              # H neurons = H lines, then bent by tanh
        P  = sigmoid(A1 @ W2 + b2)
        dZ2 = (P - Y) / len(y)
        dW2 = A1.T @ dZ2; db2 = dZ2.sum(0)
        dZ1 = (dZ2 @ W2.T) * (1 - A1 ** 2)
        dW1 = X.T @ dZ1;  db1 = dZ1.sum(0)
        W1 -= lr * dW1; b1 -= lr * db1; W2 -= lr * dW2; b2 -= lr * db2
    return (W1, b1, W2, b2)

def predict(X, p):
    W1, b1, W2, b2 = p
    return sigmoid(np.tanh(X @ W1 + b1) @ W2 + b2).ravel()

def show_boundary(params, X, y, title):
    xx, yy = np.meshgrid(np.linspace(-2.5, 2.5, 200), np.linspace(-2.5, 2.5, 200))
    zz = predict(np.c_[xx.ravel(), yy.ravel()], params).reshape(xx.shape)
    plt.contourf(xx, yy, zz, levels=20, cmap="coolwarm", alpha=0.6)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", edgecolor="k")
    plt.title(title)

# EXAMPLE: 8 neurons -> a boundary that bends around the moons.
p8 = train_mlp(Xm, ym, H=8)
print(f"8 neurons accuracy: {((predict(Xm, p8) > 0.5).astype(int) == ym).mean():.2f}")
show_boundary(p8, Xm, ym, "8 neurons"); plt.show()
"""))

both(md("""
**Your turn.** Train with just `H=1` neuron, then `H=6`. Show both boundaries. Which one bends?
"""))

pair(
    code("""
# your turn: call train_mlp on (Xm, ym) with H=1 and again with H=6,
# then use show_boundary(...) to plot each. Compare them.
"""),
    code("""
for H in [1, 6]:
    p = train_mlp(Xm, ym, H=H)
    acc = ((predict(Xm, p) > 0.5).astype(int) == ym).mean()
    plt.figure(figsize=(4, 4))
    show_boundary(p, Xm, ym, f"{H} neuron(s), acc={acc:.2f}"); plt.show()
"""),
)

# ---- 4. Images are tensors ------------------------------------------------ #
both(md("""
## 4. Images are just numbers

An image is a grid of numbers: **height x width** (x **depth** for color: 3 channels R, G, B).
A network **layer** turns one such grid into another. A neuron doesn't care how many inputs it
has -- flatten the grid and it's the same `w·x + b`.
"""))

both(code("""
from sklearn.datasets import load_digits
digits = load_digits()
img = digits.images[0]                      # one handwritten digit
print("this image is a grid of numbers, shape =", img.shape, "(height x width, depth 1 = gray)")
print(img.astype(int))
plt.imshow(img, cmap="gray"); plt.title(f"label: {digits.target[0]}"); plt.show()

# EXAMPLE: one neuron on flattened images -> tell 0s from 1s.
keep = digits.target < 2                     # digits 0 and 1
Xd = digits.images[keep].reshape(len(digits.images[keep]), -1)   # 8x8 -> 64 numbers
Xd = (Xd - Xd.mean(0)) / (Xd.std(0) + 1e-6)
yd = digits.target[keep]
w = np.zeros(64); b = 0.0
for step in range(2000):
    p = sigmoid(Xd @ w + b)
    w -= 0.1 * Xd.T @ (p - yd) / len(yd)
    b -= 0.1 * (p - yd).mean()
print(f"0-vs-1 accuracy: {((sigmoid(Xd @ w + b) > 0.5).astype(int) == yd).mean():.2f}")
"""))

both(md("""
**Your turn.** Show a different digit and its shape. Then tell 3s from 8s with the same neuron.
"""))

pair(
    code("""
# your turn:
# (a) imshow digits.images[10] and print its shape.
# (b) copy the EXAMPLE, but keep digits 3 and 8 (hint: (digits.target==3)|(digits.target==8),
#     and set the label to (digits.target==8)). Print the accuracy.
# (c) in a comment: what shape is a 28x28 COLOR photo?
"""),
    code("""
plt.imshow(digits.images[10], cmap="gray"); plt.title(f"label {digits.target[10]}"); plt.show()
print("shape:", digits.images[10].shape)

keep = (digits.target == 3) | (digits.target == 8)
Xd = digits.images[keep].reshape(keep.sum(), -1)
Xd = (Xd - Xd.mean(0)) / (Xd.std(0) + 1e-6)
yd = (digits.target[keep] == 8).astype(int)
w = np.zeros(64); b = 0.0
for step in range(2000):
    p = sigmoid(Xd @ w + b)
    w -= 0.1 * Xd.T @ (p - yd) / len(yd)
    b -= 0.1 * (p - yd).mean()
print(f"3-vs-8 accuracy: {((sigmoid(Xd @ w + b) > 0.5).astype(int) == yd).mean():.2f}")

# (c) a 28x28 color photo has shape (28, 28, 3): height x width x depth(3 = R,G,B).
"""),
)


# ---- 5. Messing with images (preprocessing) ------------------------------ #
both(md("""
## 5. Messing with images

Before training we **normalize** pixel values (small, consistent numbers train better) and
**augment** with shifts / shears / erosion so the model sees variety. Same image, different
tools.
"""))

both(code("""
from scipy import ndimage

five = digits.images[digits.target == 5][0]

# normalize: values 0..16 -> 0..1
print("before:", five.min(), five.max(), "  after /16:", round((five / 16).max(), 2))

# augment: shear (skew sideways) and erode (thin the strokes)
sheared = ndimage.affine_transform(five, np.array([[1, 0.4], [0, 1]]), offset=(-1.5, 0), order=1)
eroded  = ndimage.grey_erosion(five, size=(2, 2))

fig, ax = plt.subplots(1, 3, figsize=(7, 2.6))
ax[0].imshow(five, cmap="gray");    ax[0].set_title("original")
ax[1].imshow(sheared, cmap="gray"); ax[1].set_title("sheared")
ax[2].imshow(eroded, cmap="gray");  ax[2].set_title("eroded")
for a in ax: a.axis("off")
plt.show()
"""))

both(md("""
**Your turn.** Erode a **5** and a **1** the same way. The 1 is a thin stroke, the 5 is thicker
-- what happens differently?
"""))

pair(
    code("""
five = digits.images[digits.target == 5][0]
one  = digits.images[digits.target == 1][0]

# your turn: apply ndimage.grey_erosion(im, size=(2,2)) to BOTH digits, and show
# original vs eroded for each (a 2x2 grid). What happens to the thin 1?
"""),
    code("""
five = digits.images[digits.target == 5][0]
one  = digits.images[digits.target == 1][0]

fig, ax = plt.subplots(2, 2, figsize=(5, 5))
for r, (name, im) in enumerate([("5", five), ("1", one)]):
    ax[r, 0].imshow(im, cmap="gray");                               ax[r, 0].set_title(f"{name} original")
    ax[r, 1].imshow(ndimage.grey_erosion(im, size=(2, 2)), cmap="gray"); ax[r, 1].set_title(f"{name} eroded")
for a in ax.ravel(): a.axis("off")
plt.show()
# the thin 1 almost disappears; the thicker 5 mostly survives -> erosion hurts thin strokes most.
"""),
)


def build():
    out = ROOT / "notebooks/day0_intro"
    out.mkdir(parents=True, exist_ok=True)
    lab = new_nb(); lab.cells = LAB; save(lab, out / "intro.ipynb")
    sol = new_nb(); sol.cells = SOL; save(sol, out / "intro_solution.ipynb")
    print(f"wrote intro.ipynb ({len(LAB)} cells) + intro_solution.ipynb")


if __name__ == "__main__":
    build()
