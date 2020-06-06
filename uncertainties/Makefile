all: uncertainties.pdf

%.pdf: %.tex refs.bib
	pdflatex $*
	bibtex $*
	pdflatex $*
	bash -c " ( grep Rerun $*.log && pdflatex $* ) || echo noRerun "
	bash -c " ( grep Rerun $*.log && pdflatex $* ) || echo noRerun "
