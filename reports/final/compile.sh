rm *.aux; \
rm *.out; \
rm *.bbl; \
rm *.toc; \ 
rm *.figbib; \ 
rm *.figbib.blg; \ 
rm *.blg; \ 
rm *.log; \ 

pdflatex report; \
bibtex report; \
pdflatex report; \
bibtex report.figbib \
pdflatex report; \
bibtex report; \
pdflatex report; \
bibtex report.figbib \

pdflatex report; \
pdflatex report; \
pdflatex report; \
open report.pdf
