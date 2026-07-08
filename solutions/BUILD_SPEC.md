# Solution build spec — hand-authored freeform decks + worked notebooks

Per-group deliverables for the Day-3 capstone "answer keys". Audience: HS sophomores who barely
know Python but all have Claude Code. Teach **judgment** (what to build, which tool, why), not syntax.
Every code line in notebooks is commented. Slides are **very descriptive** (full sentences).

Each group folder `solutions/<group>/` already has: `run_experiment.py`, `figures/*.png`,
`figures/raw/*.csv`, `results.json`, `literature.md`. You produce:

1. `intro_figures.py` → 2-3 bespoke matplotlib figures redrawing the foundational refs in
   `literature.md` (attribution string in each, e.g. "adapted from Esteva 2017"). Use the shared
   style: `sys.path.insert(0, "../_common"); import solfig as sf` (theme "bone", brand palette).
   Save with `sf.save(fig, HERE, "intro_<name>")` + `sf.save_raw(...)`. See `g1_crispr/intro_figures.py`.
2. `build_notebook.py` → `solution.ipynb`: the worked answer key. Runs top-to-bottom, reproduces the
   real results, EVERY code line commented, rich HS-accessible markdown with background (cite refs by
   [n]), the core idea, results, and honest limits. Copy the structure of `g1_crispr/build_notebook.py`.
   Validate headless before committing (see below).
3. A **hand-authored freeform deck** via build-pptx (see recipe). Markdown at `slides/<group>.md`,
   sidecar `slides/<group>.md.layout.json`, output `slides/<group>.pptx` (+ .pdf). ~14-18 slides.
   Arc: title → Background section (with your intro figures + what real scientists do, citing refs) →
   Methods section (the task, the data, the key idea) → Results section (your real figures + numbers
   from results.json) → Limits section (from literature.md positioning) → References → closing.

Then **commit just your group** (`git add solutions/<group> && git commit`) so slides land incrementally.

## Environment
- Python: `/data/rauschecker1/jkw/envs/outset/bin/python` (has torch, sklearn, python-pptx, markdown, nbclient).
- Repo: `~/outset-ai-healthcare`. build-pptx skill: `~/.claude/skills/build-pptx` (build.py).
- Set `export HF_TOKEN=<your-hf-token> HF_HUB_DISABLE_XET=1` when running experiments.

## Deck recipe (PROVEN — follow exactly)

1. Write `slides/<group>.md`: YAML frontmatter (`title`, `eyebrow: "Outset · AI in Healthcare · Day 3 · Group N"`,
   `subtitle`, `name: "Jinchi Wei"`, `org: "Outset"`, `date: "2026-07-08"`), then `---`-separated slides.
   `# H1` = a section divider (Background / Methods / Results / Limits). `## H2` = a content slide title.
   Put a one-line lede paragraph under the H2 (chrome renders it). Reference figures relatively:
   `![alt](figures/foo.png)` — note figures live one level up, so use `figures/...` won't resolve; instead
   COPY the figures you need into `slides/figures/` first (`mkdir -p slides/figures && cp figures/*.png slides/figures/`).
2. Generate the sidecar scaffold + hand-author it:
   `cd ~/.claude/skills/build-pptx && python build.py --input <abs>/slides/<group>.md --output <abs>/slides/<group>.pptx --plan-only --shake`
   Then edit `slides/<group>.md.layout.json`:
   - Set top-level `"theme": "bone"` (MUST match the figures' canvas, or figures show a gray box).
   - For each `freeform` slide set `params._provenance = "agent"` and write `params.code` (the body).
   - For `section-divider` slides keep `params.accent_hex` (Background→#40E0D0, Methods→#FF1493,
     Results→#F0C840, Limits/Discussion→#8A2BE2).
3. Render + QA: `python build.py --input ...md --output ...pptx --qa` (from the skill dir).
   It writes `<pptx>_qa/slide-*.png`. **READ every PNG**, check brand + no overflow/crowding, fix the
   sidecar `code`, re-render. Repeat until clean. The build ABORTS if any content slide is still
   `_provenance:"composer"` — every content slide must be hand-authored.

### Freeform sandbox API (available names inside `params.code`)
Chrome (title, lede, accent bar, footer) is auto-drawn — your `code` only draws the BODY, inside the
region `body_l, body_top, body_w, body_h` (floats, inches). NO imports, no try/with, no dunders.

- `_add_text(slide, text, *, left, top, width, height, size=18, color_rgb=INK_RGB, font=SANS_FONT, bold=False, italic=False, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP)`
- `_fit_image(slide, 'figures/foo.png', *, left, top, max_w, max_h)`  — relative path; scales to fit.
- `_add_rect(slide, *, left, top, width, height, fill_rgb=None)`
- `_add_card(slide, *, label, body, left, top, width, height, accent_rgb)`  — titled surface card.
- Colors (RGBColor, ready to use): `accent_rgb, INK_RGB, WHITE_RGB, MUTED_RGB, TURQUOISE_RGB, DEEPPINK_RGB, AMBER_RGB, BLUEVIOLET_RGB, PAPER_RGB, SURFACE_RGB, CANVAS_BG_RGB`. `THEME_RGBS` = list. `_rgb('#hex')` for custom. `ON_DARK` = bool.
- Fonts: `MONO_FONT` (titles/numbers/labels), `SANS_FONT` (prose). Enums: `PP_ALIGN`, `MSO_ANCHOR`.
- **Gotcha:** `color_rgb` must be an RGBColor (use the provided constants or `_rgb('#..')`), NEVER a tuple — a tuple silently error-chips the slide.

### Good freeform patterns (compose bespoke per slide — don't clone one layout)
- **Figure hero:** `_fit_image(slide, 'figures/x.png', left=body_l, top=body_top+0.1, max_w=body_w, max_h=body_h-0.2)` + a small `_add_text` caption below.
- **Figure + aside:** image on left `max_w=body_w*0.60`; `_add_card(...)` or stacked `_add_text` bullets on the right (`left=body_l+body_w*0.63`).
- **Big stat:** huge `_add_text('0.77', size=70, color_rgb=accent_rgb, font=MONO_FONT, bold=True)` + a label + a sentence of context; optionally a figure beside it.
- **Prose:** 2-3 short `_add_text` paragraphs (size 16-18, SANS_FONT), generous spacing. Descriptive full sentences.
- **Bullets:** loop y-positions, each a bold MONO label chip + SANS body via two `_add_text` calls.

Reference: the working smoke deck is at `~/outset-ai-healthcare/solutions/_common/` docs, and the day
decks `~/outset-ai-healthcare/slides/day3.md` + `day3.md.layout.json` are real hand-authored examples.

## Notebook validation
`/data/rauschecker1/jkw/envs/outset/bin/python -c "import nbformat; from nbclient import NotebookClient; nb=nbformat.read('solutions/<group>/solution.ipynb',4); NotebookClient(nb, kernel_name='outset', resources={'metadata':{'path':'solutions/<group>'}}).execute(); print('OK')"`
(Register kernel once if needed; the `outset` kernel is already installed.) Remove any `catboost_info/`.

## Brand lock (non-negotiable)
Geist Mono titles/numbers, Geist body, palette turquoise→deeppink→amber→blueviolet. Author name
"Jinchi Wei" (never "Jin"). No em dashes in student-facing text (use " -- "). Theme "bone" everywhere.
