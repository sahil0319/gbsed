# Report and presentation

## Building the report

```bash
export PATH="/Library/TeX/texbin:$PATH"   # macOS MacTeX
latexmk -pdf main.tex
```

Produces `main.pdf` (46 pages). `latexmk -C` cleans build artefacts.

## Before submitting — fill in every `<<...>>`

Search `main.tex` for `<<` . The placeholders are:

| Location | Item |
|---|---|
| Preamble | Two team member names + all three student IDs |
| Preamble | Supervisor name and designation |
| Preamble | Defense date |
| Appendix A | Weekly meeting table |
| Appendix B | GitHub commit/LoC metrics per member |
| Appendix B | Which member owned which feature |

The first author entry is pre-filled as `Md. Sahil Mahdi / 220041149` —
**verify it**, it was inferred from a filename rather than told to us.

`PRESENTATION.md` has the same `<<Name>>` / `<<ID>>` placeholders.

## Files

| file | contents |
|---|---|
| `main.tex` | the report, 9 chapters + 2 appendices |
| `citations.bib` | 15 references |
| `figures/` | 9 figures (see below) |
| `PRESENTATION.md` | slide-by-slide content, timing plan, anticipated Q&A |
| `iutbscthesis.cls`, `*.sty` | unmodified from the provided template |

## Figures

| file | where used | what it shows |
|---|---|---|
| `architecture.png` | Ch.4 | full pipeline; green = transmitter, blue = our network contribution, red = receiver |
| `scenario.png` | Ch.6 | SUMO geometry + measured delivery vs separation |
| `sg_original.png` / `sg_reconstructed.png` | Ch.6 | a five-actor graph before and after transmission |
| `matched_budget.png` | Ch.6 | fixed-budget comparison; overlapping delivery lines are the control |
| `rate_semantics.png` | Ch.6 | the 96× result |
| `payload_size.png` | Ch.6 | payload sizes, log scale |
| `loss_resilience.png` | Ch.6 | one frame with one chunk lost; the collision-risk relation survives in the reconstructed graph |
| `superseded_metrics.png` | Ch.6 | the original figure, kept as evidence of two defects |

All are regenerable — see `../overview.md` §5.

## Note on the last figure

`superseded_metrics.png` is included **deliberately**, as evidence in the
section documenting the corrected defects. It shows three panels plotting the
identical curve, and a rise at `CAV_Good` that is not a recovery but `Baseline`
replotted under a duplicate configuration. Do not replace it with a corrected
version — the point is what it looked like before.
