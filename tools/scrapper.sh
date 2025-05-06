#!/bin/bash

# Ensure a text file argument is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <filename>"
    exit 1
fi

FILE="$1"

# Check if the file exists
if [ ! -f "$FILE" ]; then
    echo "Error: File '$FILE' not found!"
    exit 1
fi

# Read the file line by line and execute the command for each link
while IFS= read -r link; do
    # Skip empty lines
    if [ -n "$link" ]; then
        echo "Processing: $link"
        poetry run python Urs.py -c "$link" 0
    fi
done < "$FILE"