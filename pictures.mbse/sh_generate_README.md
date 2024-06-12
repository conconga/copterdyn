#!/bin/bash

cat README.md.no_TOC | pandoc --from markdown  --toc -s  --to markdown - > README.md

