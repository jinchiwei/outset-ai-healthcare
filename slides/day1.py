"""Day 1 slide deck: from pixels to vision transformers.

Build:  python -m slides.day1   (writes slides/build/day1.pptx)
"""
from pathlib import Path

from slides.theme import (
    new_deck, title_slide, content_slide, section_divider, code_slide, save,
    TURQUOISE, DEEPPINK, AMBER, BLUEVIOLET,
)


def build():
    prs = new_deck()

    title_slide(
        prs,
        title="AI in Healthcare",
        subtitle="Day 1: From pixels to vision transformers  ·  Outset, July 6 2026",
        accent=TURQUOISE,
    )
    content_slide(prs, title="Today", bullets=[
        "Lecture: what is an image, and what is 'learning'?",
        "Lab: build five models on real eye scans, watch each one improve",
        "Share-back, and a bridge to tomorrow",
    ])
    content_slide(prs, title="How to get help", bullets=[
        "Everyone runs the same notebook. Fill in the # TODO blanks.",
        "Stuck? Ask me, or ask Claude. Both are fair game.",
        "One rule: always be able to explain what your code does.",
        "There is no 'behind'. Go at your pace.",
    ])

    # --- Why does this matter? --------------------------------------------- #
    section_divider(prs, label="Why does this matter?", accent=DEEPPINK)
    content_slide(prs, title="AI already reads medical images", bullets=[
        "Diabetic retinopathy screening, deployed in clinics in India and Thailand",
        "Mammogram triage, FDA-cleared and in use",
        "Sepsis early-warning in hospital EHRs",
        "These are real systems treating real patients right now.",
    ])
    content_slide(prs, title="...and it goes wrong in real ways", bullets=[
        "Shortcut learning: the model reads the ruler, not the tumor",
        "Distribution shift: your hospital is not the training hospital",
        "Overconfidence: '90% sure' that means what, exactly?",
        "Knowing how these fail is the actual skill.",
    ])
    content_slide(prs, title="Today's data: the eye", bullets=[
        "APTOS-2019: color photos of the retina (the back of the eye)",
        "Our task: referable diabetic retinopathy. Does this eye need a doctor? Yes/no.",
        "This is the actual screening task running in clinics today",
        "You will train models to make that call.",
    ], accent=AMBER)

    # --- What is an image? ------------------------------------------------- #
    section_divider(prs, label="What is an image?", accent=AMBER)
    content_slide(prs, title="An image is a grid of numbers", bullets=[
        "Each pixel is a brightness value",
        "A grayscale image is one grid of numbers",
        "Your eye sees a photo; the computer sees a spreadsheet",
    ])
    content_slide(prs, title="Color = three grids stacked", bullets=[
        "Red, Green, Blue: one grid each",
        "Stack them and you get full color",
        "So a color image is really three spreadsheets on top of each other",
    ])
    code_slide(prs, title="In code, it's just an array", code=(
        "image.shape == (224, 224, 3)\n"
        "             height  width  channels\n\n"
        "image[0, 0]  ->  [12, 38, 7]   # the top-left pixel, as R,G,B"
    ))
    content_slide(prs, title="3D images: voxels", bullets=[
        "A CT or MRI scan is a stack of image slices",
        "Each 3D pixel is called a voxel",
        "Same idea, one more dimension. We'll stick to 2D today.",
    ])
    content_slide(prs, title="Augmentation: free extra data", bullets=[
        "Flip it, rotate it, brighten it: still the same eye",
        "The model sees more variety, memorizes less",
        "We'll see this in the notebook.",
    ])

    # --- What is learning? ------------------------------------------------- #
    section_divider(prs, label="What is 'learning'?", accent=BLUEVIOLET)
    content_slide(prs, title="A classifier is a function", bullets=[
        "Input: a pile of numbers (the image)",
        "Output: a label (refer to a doctor, or not)",
        "'Learning' = adjusting the function until its answers match the truth",
    ])
    content_slide(prs, title="The ladder we'll build", bullets=[
        "1. Logistic regression  (the simplest thing that works)",
        "2. MLP  (a small neural net)",
        "3. CNN  (sees spatial structure)",
        "4. ResNet  (transfer learning)",
        "5. Vision Transformer  (attention on patches)",
    ])
    content_slide(prs, title="1. Logistic regression", bullets=[
        "Flatten the image into one long row of numbers",
        "Draw a straight-line boundary: refer or not",
        "Treats every pixel as independent. No idea pixels form shapes.",
    ])
    content_slide(prs, title="2. MLP (multi-layer perceptron)", bullets=[
        "Stack layers with non-linearities -> curved boundaries",
        "More flexible than logreg",
        "Still flattens the image. Still blind to space.",
    ])
    content_slide(prs, title="3. CNN", bullets=[
        "Slides small filters across the image",
        "Learns edges, blobs, textures, on its own",
        "Finally understands that neighboring pixels belong together",
    ], accent=TURQUOISE)
    content_slide(prs, title="4. ResNet (transfer learning)", bullets=[
        "A 50-layer net already trained on a million everyday photos",
        "We freeze it and train just a small new final layer",
        "Reuses general 'vision' for our specific task. Huge shortcut.",
    ], accent=TURQUOISE)
    content_slide(prs, title="5. Vision Transformer", bullets=[
        "Chop the image into patches",
        "Let the patches 'pay attention' to each other",
        "Same machinery as the language models you've used.",
    ], accent=TURQUOISE)

    # --- Lab --------------------------------------------------------------- #
    section_divider(prs, label="Lab: build the ladder", accent=DEEPPINK)
    content_slide(prs, title="How the lab works", bullets=[
        "Open the notebook, run the setup cell first",
        "Fill in each # TODO, run the cell, see the accuracy",
        "Stop at the reflection prompts. Predict before you run.",
        "Stuck on a TODO? Ask Claude, then make sure you get the answer.",
    ])
    content_slide(prs, title="Using Claude well (watch me)", bullets=[
        "I'll take one TODO and show you how I'd ask Claude",
        "Notice: I read the answer, I don't just paste it",
        "I check it makes sense before trusting it",
        "That's the whole skill. Now your turn.",
    ], accent=AMBER)

    # --- Share-back + bridge ----------------------------------------------- #
    section_divider(prs, label="What you just built", accent=BLUEVIOLET)
    content_slide(prs, title="The leaderboard", bullets=[
        "Flat-pixel models (logreg, MLP): around 50%",
        "CNN: a little better. It sees structure.",
        "ResNet and ViT: the big jump. Transfer learning wins.",
        "The lesson: borrowed knowledge beats from-scratch on small data.",
    ])
    content_slide(prs, title="The bridge to tomorrow", bullets=[
        "The ViT split the image into patches, embedded each, used attention",
        "Swap patches for words and you have a Large Language Model",
        "Same machinery. Different input.",
        "Tomorrow: we use one to read radiology reports.",
    ], accent=DEEPPINK)
    code_slide(prs, title="Same machinery, different input", code=(
        "IMAGE   [patch][patch][patch] -> embed -> attention -> refer?\n"
        "TEXT     'the'  'cat'  'sat'  -> embed -> attention -> next word\n\n"
        "the second line is an LLM."
    ))
    content_slide(prs, title="See you tomorrow", bullets=[
        "You trained five models on real medical images today",
        "Tomorrow: language models, and combining image + text + data",
    ])

    out = Path(__file__).resolve().parents[1] / "slides/build/day1.pptx"
    return save(prs, out)


if __name__ == "__main__":
    p = build()
    print("wrote", p)
