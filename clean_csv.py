#!/usr/bin/env python3

import csv
import sys

def remove_quotes_from_content(content):
    return content.replace('"', '')

def clean_csv(input_file, output_file):
    with open(input_file, 'r', newline='') as f:
        content = f.read()
        content = remove_quotes_from_content(content)

    lines = list(csv.reader(content.splitlines()))

    cleaned_lines = []
    for line in lines:
        # Assuming maximum 2 fields per line (as per your example)
        while len(line) > 2:
            line[0] = line[0] + ',' + line.pop(1)
        cleaned_lines.append(line)

    with open(output_file, 'w', newline='\n') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(cleaned_lines)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: script_name input_file output_file")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    clean_csv(input_file, output_file)
