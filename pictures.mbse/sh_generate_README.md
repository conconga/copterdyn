#!/bin/bash

cat README.md.before_pandoc | pandoc --from markdown+grid_tables --toc -s --number-sections --to gfm - > README.md

