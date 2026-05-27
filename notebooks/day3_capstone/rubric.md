# Capstone presentation rubric

3 minutes per pair. 5 points, 1 each. The bar is understanding and honesty, not a leaderboard.

| # | Point | What earns it |
|---|-------|---------------|
| 1 | **Built it** | The notebook runs end to end. You can show the model making a prediction. |
| 2 | **Evaluated honestly** | You report test-set (not training) accuracy, with a sensible metric and baseline. |
| 3 | **Found a failure mode** | You show at least one case the model got wrong and explain why it might have. |
| 4 | **Defended a decision** | Each partner names one design choice they personally made and says why. |
| 5 | **Both contributed** | Each partner can answer a question about the other's part of the work. |

## What each score looks like

**1/5** "We ran the starter notebook and got 82%." (It ran. Nothing else demonstrated.)

**3/5** "We added data augmentation and test accuracy went from 0.82 to 0.86. Here are
three images it still gets wrong; two of them are blurry." (Built, evaluated honestly,
found a failure mode. No clear defended decision or shared understanding shown.)

**5/5** "We unfroze the last ResNet block and lowered the learning rate to 1e-4, which
took test accuracy from 0.82 to 0.88. We chose to weight the loss because the classes
were 4:1 imbalanced [partner A explains]. The model fails on lateral-view X-rays, which
are rare in training [partner B explains and shows two]. If we had more time we'd collect
more lateral views." (All five.)

## Note on Claude
Using Claude is expected and encouraged. It does not cost you points. What costs points is
not being able to explain what your code does. If Claude wrote it, read it, understand it,
own it.
