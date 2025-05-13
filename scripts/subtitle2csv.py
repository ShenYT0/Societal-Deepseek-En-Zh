""""""
import csv

import logging
from tqdm import tqdm
from pathlib import Path
from typing import Sequence, Optional


def get_data_from_file(file: Path, counter_start: int):
    # Parse file with soup
    with open(file, 'r') as f:
        subtitles = [subtitle.strip() for subtitle in f.readlines() if subtitle.strip()]
        subtitle_texts = [subtitles[i] for i in range(2, len(subtitles), 3)]  # Every three line is the text
    # Extract data
    data = [{'id': counter_start + i,
             'title': file.stem,
             'text': text}
            for i, text in enumerate(subtitle_texts)]
    counter_start += len(data)
    return data, counter_start


def main(input_dir: Path, output_csv: Path) -> None:
    # Logger config
    logging.basicConfig(level=logging.INFO)

    # Get data from files
    corpus = []
    counter_start = 0
    for file in input_dir.glob('*.srt'):
        data, counter_start = get_data_from_file(file, counter_start)
        corpus.extend(data)

    # Output to csv
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['id', 'title', 'text']

        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        [writer.writerow(row) for row in corpus]


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=Path, default=Path("../data/clean/tech/subtitle"))
    parser.add_argument("--output_csv", type=Path, default=Path("../data/clean/tech/subtitle.csv"))
    args = parser.parse_args()

    main(args.input_dir, args.output_csv)
