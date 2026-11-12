# Build the codebase guide.
#   make          -> openrc_guide.pdf
#   make clean    -> remove LaTeX build artefacts
#   make figures  -> regenerate docs/images
PDF = openrc_guide.pdf

$(PDF): openrc_guide.tex
	latexmk -pdf -interaction=nonstopmode -halt-on-error $<

figures:
	cd .. && python3 scripts/generate_readme_figures.py

clean:
	latexmk -c   # -c keeps the PDF; use -C to remove it too
	rm -f *.aux *.log *.out *.toc *.fls *.fdb_latexmk build.log

.PHONY: figures clean
