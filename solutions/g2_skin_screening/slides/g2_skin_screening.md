---
title: "Never miss a melanoma"
eyebrow: "Outset · AI in Healthcare · Day 3 · Group 2"
subtitle: "Screening skin spots for cancer, where catching every melanoma matters more than raw accuracy"
name: "Jinchi Wei"
org: "Outset"
date: "2026-07-08"
---

# Background

---

## Two mistakes, and only one of them can be fatal

Dermatology-AI systems look at a photo of a skin spot and estimate how likely it is to be cancer, and landmark studies showed a neural network can match or beat dermatologists at exactly this task (Esteva 2017, Haenssle 2018). But for a screening tool the goal is not to be right on average -- it is to never wave a real melanoma through as harmless, because that missed cancer is the one mistake that can kill.

---

## Screening tunes one dial: sensitivity versus specificity

Every screening model has a decision threshold, and where you set it is a trade-off you cannot escape. Slide it one way and you catch more cancers but raise more false alarms; slide it the other way and you calm the false alarms but start missing real disease. Screening deliberately favors sensitivity -- catching the cancers -- because a false alarm just costs a follow-up visit (Trevethan 2017).

---

## We borrow a brain instead of starting blank

We only have about ten thousand skin images, far too few to teach a network vision from scratch. So we use transfer learning: start from a network already trained on millions of everyday photos, keep everything it learned about edges, colors, and textures, and fine-tune only a small new head on the skin images (Kim 2022). That borrowed knowledge is why a model can learn skin cancer from thousands, not millions, of pictures.

---

# Methods

---

## The task: seven kinds of skin spot, one that we cannot miss

Our data is DermaMNIST, a tidy version of the HAM10000 dermatoscopy collection with seven lesion types, including melanoma (Tschandl 2018). The catch is severe imbalance: harmless moles make up about two-thirds of every set, and melanoma is a small slice -- which is precisely the setup where a lazy model can look accurate while catching zero cancers.

---

## How we built it, and how we graded it honestly

We trained the network three ways -- from scratch, pretrained, and pretrained with augmentation -- and then audited the best one the way a regulator would. We did not grade on accuracy alone; we tracked melanoma recall, the share of true melanomas actually caught, because on this data that single number decides whether the tool is safe.

---

# Results

---

## A better starting point beat a fancier model

The three training setups landed almost on top of each other, and that is the honest, humbling lesson: on a dataset this small the ceiling is set by the data, not by how clever the model is. A better starting point (transfer learning) matters more than any fancy trick you bolt on afterward.

---

## The confusion matrix tells the real story

Overall accuracy is one number; the confusion matrix shows which mistakes the model actually makes. The diagonal is correct and everything off it is an error, and one column swallows almost everything -- the model funnels nearly every spot, cancer or not, into the common harmless-mole class.

---

## Screening lives or dies on the rare classes

Break accuracy down class by class and the average falls apart. The model is near-perfect on the one common class and near the floor on every rare one, including melanoma, highlighted in pink. A tool that works this differently across classes is not one you would ever deploy to screen a patient.

---

## The one number that matters: melanoma recall

Here is the sentence that should stop the room. Our model reached sixty-eight percent overall accuracy and still caught only ten of the two hundred twenty-three real melanomas in the test set. It missed two hundred thirteen cancers -- proof that a healthy-looking accuracy number can hide a screening tool that is dangerous.

---

## The cases it got wrong are the ones that matter

The most revealing slide in any talk is the model's mistakes. Look at how many true melanomas it labeled as harmless moles: every one of those is a false negative, the exact error a screening tool exists to prevent, and the reason a dermatologist would want a second look at all of them.

---

# Being honest

---

## A toy model, and why that is exactly the point

This model is far too small and simple to trust on a real patient: sixty-four-pixel grayscale thumbnails throw away the color and fine texture that dermatologists and real systems rely on, and its melanoma recall is nowhere near safe (Wei 2024). A real tool would train on full-resolution color images, deliberately tune its threshold toward high sensitivity, and be validated across many hospitals first. The point was never the score -- it was learning the screening mindset: never miss the dangerous case, and never let a shiny number fool you.

---

## References, and thank you

Seven papers grounded this project, from the dermatologist-level results that started the field to the plain-language guide on why sensitivity matters. Thank you for watching -- go build something you can honestly evaluate.
