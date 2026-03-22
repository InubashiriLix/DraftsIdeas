#!/usr/bin/env bash
set -e

if [ $# -lt 1 ]; then
    echo "Usage: $0 input.md [output.pdf]"
    exit 1
fi

INPUT="$1"

if [ ! -f "$INPUT" ]; then
    echo "Error: file '$INPUT' not found."
    exit 1
fi

BASENAME="$(basename "$INPUT" .md)"
OUTPDF="${2:-$BASENAME.pdf}"
OUTTEX="${BASENAME}.tex"

pandoc "$INPUT" \
    -s \
    --template=template.tex \
    --pdf-engine=xelatex \
    -V papersize=a4 \
    -V fontsize=11pt \
    -o "$OUTPDF"

echo "Generated: $OUTPDF"
