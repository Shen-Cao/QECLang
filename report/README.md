# Research report

`main.tex` includes short thematic sections and a public-URL bibliography. It describes the semantic design, finite binary identities, analysis contracts, and open implementation milestones. It does not claim a complete universal QEC implementation or a new asymptotically good code construction.

Build from this directory with a TeX installation containing the packages in `main.tex`:

```sh
mkdir -p ../build/report
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=../build/report main.tex
(cd ../build/report && BIBINPUTS=../../report: bibtex main)
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=../build/report main.tex
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=../build/report main.tex
```

Generated files belong in the ignored build directory. Bibliographic links remain valid independently of any local reference cache.
