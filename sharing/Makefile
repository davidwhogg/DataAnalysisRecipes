.SUFFIXES: .pdf .tex

LATEX       = pdflatex
RM          = rm -rf

TMP_SUFFS   = pdf aux bbl blg log dvi ps eps out ent
SUFF        = pdf

RM_TMP      = $(foreach d, ${TEX_FILES}, ${RM} $(foreach suff, ${TMP_SUFFS}, ${d}.${suff}))
CHECK_RERUN = grep Rerun $*.log

DOCS        = $(foreach f, ${TEX_FILES}, ${f}.${SUFF})

# Enter the name(s) of the .tex file(s) that you want to compile.
TEX_FILES = release
DIRS      =

# You shouldn't need to edit below here.
default: subdirs ${DOCS}

.tex.pdf:
	${LATEX} $*.tex
	( ${CHECK_RERUN} && ${LATEX} $*.tex ) || echo "Done."
	( ${CHECK_RERUN} && ${LATEX} $*.tex ) || echo "Done."

subdirs: force_look
	$(foreach d, ${DIRS}, (echo "Looking into ${d}:"; cd ${d}; ${MAKE} ${MFLAGS}) )

force_look:
	true

clean:
	$(RM_TMP)
	$(foreach d, ${DIRS}, (echo "Cleaning ${d}:"; cd ${d}; $(MAKE) clean) )
