# IEEE LaTeX paper project

Main file: `main.tex`

Compile with a standard LaTeX distribution:

```text
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

The author block lists Heng Zhang, School of Electronic Information, Wuhan
University, using the template's `\thanks{...}` affiliation format.

Experimental scope:

- CIFAR-10 with PreActResNet-18 and ResNet-18, plus CIFAR-100 transfer
- held-out detection on multiple collapse trajectories and non-event controls
- natural random-start CO stress tests at epsilon 16/255
- five-seed immediate/no-replay, replay, and deferred intervention controls
- unified five-seed FastAdv+ and GradAlign baselines
- full 10,000-image CIFAR-10 PGD-10/PGD-50/multi-restart evaluation
- fixed first-1,000-image AutoAttack-standard evaluation for seven checkpoints
- offline threshold, warning-horizon, and CO-definition sensitivity tables

Audit documents: `AUDIT.md`, `CHANGELOG.md`, and `NOT_DONE.md`.
