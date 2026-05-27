"""Smoke test for the slide theme module."""
from slides.theme import (
    new_deck, title_slide, content_slide, section_divider, code_slide, TURQUOISE,
)


def test_theme_builds():
    prs = new_deck()
    title_slide(prs, title="Test", subtitle="Smoke", accent=TURQUOISE)
    content_slide(prs, title="Bullets", bullets=["a", "b", "c"])
    section_divider(prs, label="Break")
    code_slide(prs, title="Code", code="image.shape == (224, 224, 3)")
    assert len(prs.slides) == 4
    # 16:9 geometry
    assert prs.slide_width == 12192000  # 13.333 in in EMU


if __name__ == "__main__":
    test_theme_builds()
    print("ok")
