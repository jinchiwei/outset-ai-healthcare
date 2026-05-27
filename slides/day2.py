"""Day 2 slide deck: LLMs and multimodal medical AI.

Build:  python -m slides.day2
"""
from pathlib import Path

from slides.theme import (
    new_deck, title_slide, content_slide, section_divider, code_slide, save,
    TURQUOISE, DEEPPINK, AMBER, BLUEVIOLET,
)


def build():
    prs = new_deck()

    title_slide(prs, title="AI in Healthcare",
                subtitle="Day 2: LLMs and multimodal medical AI  ·  Outset, July 7 2026",
                accent=DEEPPINK)
    content_slide(prs, title="Yesterday / today", bullets=[
        "Yesterday: one big neural net, end to end, on eye images",
        "Today: combine three different signals on chest X-rays",
        "And meet the foundation model that ties them together",
    ])

    # --- What is an LLM ---------------------------------------------------- #
    section_divider(prs, label="What is an LLM?", accent=TURQUOISE)
    content_slide(prs, title="The bridge from yesterday", bullets=[
        "The Vision Transformer split an image into patches, used attention",
        "An LLM splits text into tokens, uses attention",
        "Same machinery. Patches become word-pieces.",
    ])
    content_slide(prs, title="Tokens, context, completion", bullets=[
        "Tokenize: break text into pieces ('cardio', 'megaly')",
        "Context: the model reads all the tokens at once",
        "Completion: it predicts what comes next, one token at a time",
    ])
    content_slide(prs, title="Why text matters in medicine", bullets=[
        "Every scan comes with a radiologist's written report",
        "Patient history, notes, prior findings: all text",
        "An LLM can turn messy free text into structured data",
    ])
    content_slide(prs, title="The catch: hallucination", bullets=[
        "LLMs are confident even when wrong",
        "They will happily structure nonsense",
        "In medicine, a confident wrong answer is dangerous. Always verify.",
    ], accent=AMBER)

    # --- The multimodal stack --------------------------------------------- #
    section_divider(prs, label="Three signals, one table", accent=BLUEVIOLET)
    content_slide(prs, title="Today's data: chest X-rays", bullets=[
        "Open-i: real chest X-rays with real radiologist reports",
        "Task: does this patient have cardiomegaly (an enlarged heart)?",
        "We have three kinds of signal to use.",
    ])
    content_slide(prs, title="Signal 1: image features", bullets=[
        "Handcrafted numbers from the X-ray: brightness, contrast, texture",
        "This is 'radiomics', how radiologists have quantified images for years",
        "Production tool is PyRadiomics; we use scikit-image, same idea",
    ])
    content_slide(prs, title="Signal 2: text features", bullets=[
        "An LLM read each report and pulled out structured findings",
        "effusion? opacity? severity? -> yes/no/level",
        "We pre-ran this with the Anthropic API so you don't need a key",
    ])
    content_slide(prs, title="Signal 3: demographics", bullets=[
        "Age, sex, smoking history",
        "Already just numbers, no conversion needed",
    ])
    content_slide(prs, title="The big idea", bullets=[
        "Image -> numbers. Text -> numbers. Demographics -> numbers.",
        "Stack them side by side into one wide table",
        "One row per patient. Everything becomes a tabular row.",
    ], accent=TURQUOISE)
    code_slide(prs, title="One row per patient", code=(
        "intensity_mean  glcm_contrast  llm_effusion  age  smoker  | label\n"
        "    121.4           0.51            1          67     1     |   1\n\n"
        "image features | text features | demographics  ->  predict"
    ))

    # --- TabPFN ------------------------------------------------------------ #
    section_divider(prs, label="TabPFN", accent=DEEPPINK)
    content_slide(prs, title="A foundation model for tables", bullets=[
        "Pretrained on millions of synthetic tables",
        "You 'fit' (it just looks at your examples) and 'predict', in seconds",
        "No training loop, no tuning. The 2020s way to model a table.",
    ])

    # --- Honesty ----------------------------------------------------------- #
    section_divider(prs, label="Is this a fair test?", accent=AMBER)
    content_slide(prs, title="The text features looked amazing", bullets=[
        "Image + demographics only: about 0.64",
        "Add the text features: about 0.89",
        "A huge jump. Too huge. Why?",
    ])
    content_slide(prs, title="Because the report names the answer", bullets=[
        "The report literally says 'cardiomegaly'",
        "The LLM's 'cardiomegaly_present' matches the label ~80% of the time",
        "We're not predicting, we're reading the answer off the report",
        "This is data leakage. It's everywhere in clinical AI.",
    ], accent=DEEPPINK)
    content_slide(prs, title="The real lesson", bullets=[
        "What counts as a fair input vs. a peek at the label?",
        "A note written after the diagnosis is not a fair input",
        "Catching this is a core skill in medical AI",
    ])

    # --- Lab + close ------------------------------------------------------- #
    section_divider(prs, label="Lab: build the stack", accent=BLUEVIOLET)
    content_slide(prs, title="Your turn", bullets=[
        "Load the feature table, assemble image + text + demographics",
        "Fit TabPFN, check the accuracy",
        "Run the ablation: drop the text, see what happens",
        "Stuck on a TODO? Ask Claude, then make sure you get the answer.",
    ])
    content_slide(prs, title="Tomorrow: capstone", bullets=[
        "Two paradigms now in your toolkit: end-to-end DL, and feature + foundation model",
        "Tomorrow you pick a problem and build it, with Claude as your pair programmer",
    ])

    out = Path(__file__).resolve().parents[1] / "slides/build/day2.pptx"
    return save(prs, out)


if __name__ == "__main__":
    print("wrote", build())
