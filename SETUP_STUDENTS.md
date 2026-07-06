# Getting set up (students)

Pick one. **If you're not sure, use Option 1 (Colab).** It needs nothing but a web browser.

---

## Option 1 — Google Colab (easiest, nothing to install)

You need: a web browser and a free Google account. That's it. No Python, no downloads.

1. Go to the notebook link your instructor shares (or open the `.ipynb` from GitHub in Colab).
2. Top menu: **Runtime → Change runtime type → T4 GPU → Save** (makes it much faster).
3. **Runtime → Run all.** The first cell installs everything automatically. Done.

That's the whole setup. Work right in the browser.

---

## Option 2 — One-file install on your own laptop

For running offline / without Colab. You do **not** need Python, conda, git, or Jupyter
installed first, the installer brings its own. Download **one file** and double-click it.

**Windows** — download and double-click:
`install/INSTALL_windows.bat`
(If SmartScreen warns: **More info → Run anyway**.)

**macOS** — download and double-click:
`install/INSTALL_mac.command`
(If macOS blocks it: **right-click → Open → Open**.)

It downloads the course to your Desktop, installs a private Python plus all the libraries, and
opens the notebooks in your browser. **First run takes a few minutes** (it's downloading a lot);
after that it's instant. Next time, open the `outset-ai-healthcare` folder on your Desktop and
double-click `START_HERE_windows.bat` / `START_HERE_mac.command`.

> Direct download links (right-click → Save link as, if needed):
> - Windows: `https://raw.githubusercontent.com/jinchiwei/outset-ai-healthcare/main/install/INSTALL_windows.bat`
> - macOS: `https://raw.githubusercontent.com/jinchiwei/outset-ai-healthcare/main/install/INSTALL_mac.command`

---

## Which should I use?

| | Colab | Local install |
|---|-------|---------------|
| Works the same on Windows / Mac / Chromebook | Yes (it's just a browser) | Two installers, one per OS |
| Need to install anything? | No | No (installer brings Python) |
| Works on a locked-down **school laptop** | Yes | Maybe not (installs can be blocked) |
| Internet required | Yes (all the time) | Yes to install; then works offline |
| Free GPU | Yes (T4) | Uses your CPU (fine, tuned to be fast) |

**Recommendation for the class: everyone uses Colab.** It's identical on every operating
system and needs no installation, so it sidesteps the two biggest headaches with a room full
of different laptops: mixed Windows/Mac setup, and school-managed machines that block installs.
Keep the local install for a student who specifically wants to tinker offline later.
