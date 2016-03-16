UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S), Linux)
	OPEN := xdg-open
endif
ifeq ($(UNAME_S), Darwin)
	OPEN := open
endif

view: project_report
	$(OPEN) project_report.pdf

project_report: project_report.tex
	pdflatex --enable-write18 project_report.tex
	pdflatex --enable-write18 project_report.tex
	rm *.aux *.log *.out *.toc
