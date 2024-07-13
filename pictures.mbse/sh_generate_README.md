#!/bin/bash

cat README.md.no_TOC | pandoc --from markdown+grid_tables --toc -s --number-sections --to gfm - > README.md

