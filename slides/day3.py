"""Day 3 slide deck: capstone. Short -- the day is mostly build time.

Build:  python -m slides.day3
"""
from pathlib import Path

from slides.theme import (
    new_deck, title_slide, content_slide, section_divider, save,
    TURQUOISE, DEEPPINK, AMBER, BLUEVIOLET,
)


def build():
    prs = new_deck()

    title_slide(prs, title="AI in Healthcare",
                subtitle="Day 3: Capstone  ·  Outset, July 8 2026", accent=AMBER)
    content_slide(prs, title="Today is build day", bullets=[
        "Pick a project, build it in pairs, present for 3 minutes at 4:30",
        "You have two paradigms in your toolkit now",
        "And Claude as a real pair programmer",
    ])
    content_slide(prs, title="What you can do now", bullets=[
        "Day 1: end-to-end deep learning, transfer learning, the ladder",
        "Day 2: handcrafted features + an LLM + a foundation model on tables",
        "Today: put it to work on a problem you choose",
    ])

    section_divider(prs, label="Three kits, or pitch your own", accent=DEEPPINK)
    content_slide(prs, title="Option 1: Pneumonia (chest X-ray)", bullets=[
        "Binary: does this X-ray show pneumonia?",
        "Runnable baseline included. Level up to full-res RSNA if you're bold.",
    ], accent=TURQUOISE)
    content_slide(prs, title="Option 2: Skin lesion triage", bullets=[
        "7-class dermatoscopy, including melanoma",
        "Level up: full HAM10000, then run it on a phone photo of a mole",
    ], accent=TURQUOISE)
    content_slide(prs, title="Option 3: Choose your own", bullets=[
        "Any MedMNIST dataset: retina, blood, pathology, organs...",
        "Or pitch something else to me right now",
    ], accent=TURQUOISE)

    section_divider(prs, label="How to win", accent=BLUEVIOLET)
    content_slide(prs, title="The rubric rewards honesty, not a leaderboard", bullets=[
        "1. It runs end to end",
        "2. You evaluated on test data, honestly",
        "3. You found a failure mode and can explain it",
        "4. Each partner defends one design decision",
        "5. Both of you understand the whole thing",
    ])
    content_slide(prs, title="Using Claude well", bullets=[
        "Use it freely. It will not cost you points.",
        "But read what it writes. Understand it. Own it.",
        "If you can't explain a line, you're not done with that line.",
    ], accent=AMBER)

    section_divider(prs, label="Build. 3:15 and 4:00 check-ins.", accent=DEEPPINK)
    content_slide(prs, title="What now, after today?", bullets=[
        "Kaggle has dozens of medical imaging competitions, free",
        "Try the level-up datasets in your kit's README",
        "The MICCAI and fastai communities are welcoming",
        "You built real models this week. Keep going.",
    ])

    out = Path(__file__).resolve().parents[1] / "slides/build/day3.pptx"
    return save(prs, out)


if __name__ == "__main__":
    print("wrote", build())
