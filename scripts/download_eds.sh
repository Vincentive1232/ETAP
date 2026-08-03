#!/usr/bin/env bash

# Script to download and extract EDS dataset sequences
# Usage: ./download_eds_data.sh [output_directory]

set -euo pipefail

command -v wget >/dev/null || { echo "Error: wget is required" >&2; exit 1; }
command -v tar >/dev/null || { echo "Error: tar is required" >&2; exit 1; }

OUTPUT_DIR="$(pwd)/data/eds"
if [ "${1:-}" != "" ]; then
    OUTPUT_DIR="$1"
fi

mkdir -p "$OUTPUT_DIR"
echo "Downloading datasets to: $OUTPUT_DIR"

BASE_URL="https://download.ifi.uzh.ch/rpg/eds/dataset"

SEQUENCES=(
    "01_peanuts_light"
    "02_rocket_earth_light"
    "08_peanuts_running"
    "14_ziggy_in_the_arena"
)

for seq in "${SEQUENCES[@]}"; do
    echo "======================================================="
    echo "Processing sequence: $seq"
    
    mkdir -p "$OUTPUT_DIR/$seq"

    if [ -s "$OUTPUT_DIR/$seq/events.h5" ]; then
        echo "Already complete, skipping: $OUTPUT_DIR/$seq/events.h5"
        continue
    fi
    
    URL="$BASE_URL/$seq/$seq.tgz"
    TGZ_FILE="$OUTPUT_DIR/$seq/$seq.tgz"
    
    echo "Downloading from: $URL"
    wget -c "$URL" -O "$TGZ_FILE"
    
    echo "Checking archive..."
    tar -tzf "$TGZ_FILE" >/dev/null

    echo "Extracting..."
    tar -xzf "$TGZ_FILE" -C "$OUTPUT_DIR/$seq"

    if [ ! -s "$OUTPUT_DIR/$seq/events.h5" ]; then
        echo "Error: extraction did not create $OUTPUT_DIR/$seq/events.h5" >&2
        echo "Archive kept for inspection: $TGZ_FILE" >&2
        exit 1
    fi
    
    echo "Removing archive..."
    rm "$TGZ_FILE"
    
    echo "Done with $seq"
    echo ""
done

echo "All sequences have been downloaded and extracted."
echo "Data is available in: $OUTPUT_DIR"

echo "======================================================="
echo "Summary of downloaded data:"
for seq in "${SEQUENCES[@]}"; do
    echo "$seq:"
    find "$OUTPUT_DIR/$seq" -maxdepth 1 -type f -printf "  %f  %s bytes\n" | sort
    echo ""
done
