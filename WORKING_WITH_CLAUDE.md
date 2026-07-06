# Working with Claude

You each have Claude Pro. Used well, it's like having an expert sitting next to you. Used
badly, it does your thinking for you and you learn nothing. Here's how to use it well.

## Your setup

Two browser tabs, side by side:
- **Colab** (your notebook, where the code runs)
- **claude.ai** (your pair programmer)

That's it. No installs.

## The loop (do this every time you're stuck)

1. **Copy the code that broke, and the *whole* error message.** Not a screenshot, the text.
2. Paste it to Claude and say what you were trying to do: *"I'm training a model and got this
   error. What's wrong, and why?"*
3. **Read the explanation.** Ask a follow-up if any part is fuzzy: *"what's a tensor shape?"*
4. Apply the fix, then re-run.
5. **The rule: you must be able to explain the fix in your own words.** If you can't, you're not
   done, ask Claude to explain it more simply.

## Ask good questions

| Weak ask | Strong ask |
|----------|-----------|
| "it doesn't work" | "this cell gives `ValueError: shapes (32,3) and (64,) ...`, here's the code, why?" |
| "fix my code" | "why does my accuracy stay at 50%? what should I check first?" |
| "make a model" | "explain, step by step, what this model cell is doing" |

More context = a better answer. Paste the code, the error, and what you already tried.

## What Claude is great at here

- Explaining an error in plain English
- Suggesting **one** change to try (then you measure if it helped)
- Translating jargon ("what's a learning rate?")
- Debugging shape mismatches, the most common beginner bug

## What to watch out for

- **Claude can be confidently wrong** (you'll learn *why* on Day 2, it's called
  *hallucination*). Don't paste code you don't understand and hope. Read it, test it.
- **Don't let it do all the thinking.** If you ask it to write the whole thing, you'll be lost
  when it breaks, and you won't be able to answer "why did you do that?" in your presentation.

## The one rule that matters

**You must be able to explain every line of your code.** That's what gets graded, and it's the
actual skill. Claude is there to help you understand faster, not to understand *for* you.
