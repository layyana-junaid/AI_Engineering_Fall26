# Submission checklist (handout section 19) - status

Done in this repo:
- [x] Hand-calculated attention (Fig 1, `outputs/logs/hand_calculation.txt`)
- [x] Causal mask shown (Fig 2)
- [x] Multi-head shape trace (Fig 3)
- [x] Positional encoding evidence (Fig 4)
- [x] Architecture diagram (`outputs/figures/architecture_diagram.png`)
- [x] Model summary / parameter count (Fig 5)
- [x] Training/validation curves (Fig 6)
- [x] Attention visualization from trained model (Fig 7)
- [x] >= 5 generation samples (`outputs/samples/eval_samples.txt`)
- [x] Three ablations (+ position, generation strategy) with noise estimate (Fig 9)
- [x] Parameter count and training configuration documented (README, report section 5)
- [x] CLI inference (Fig 10a), API (Fig 10b)
- [x] Tests, Colab notebook, README, `.gitignore`, no secrets
- [x] Report draft (`report/report.md`)

YOU still have to do (cannot be done from here):
- [ ] Create the public GitHub repo, push, and put the URL in README / report / notebook (`<YOUR_REPO_URL>`, `REPO_URL`)
- [ ] Deploy on Render (see README) and test the live URL; screenshot for Fig 10 if you want a live one
- [ ] Open the Colab notebook on a GPU runtime and run it once (re-running training there is optional; the committed results came from CPU)
- [ ] Publish the report as a Medium article (paste `report/report.md`, upload figures from `outputs/figures/`)
- [ ] Record the 5-10 min demo (`report/demo_script.md`) and prepare class slides
- [ ] Add your teammates' names / split of work; read `report/viva_prep.md`
- [ ] Re-read the report: every claim must be something you can explain
