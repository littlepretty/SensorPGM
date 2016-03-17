UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S), Linux)
	OPEN := xdg-open
endif
ifeq ($(UNAME_S), Darwin)
	OPEN := open
endif

view: project_report
	$(OPEN) report.pdf

project_report: report.tex
	pdflatex --enable-write18 report.tex
	pdflatex --enable-write18 report.tex
	rm *.aux *.log *.out *.toc
